import torch
from torch.nn.functional import normalize
import os
import glob
import torch.nn as nn
from trak.projectors import BasicProjector, CudaProjector, ProjectionType

def prepare_batch(batch, device):
    for key in batch:
        batch[key] = batch[key].to(device)

def prepare_optimizer_state(model, optimizer_state, device):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1) for n in names])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq

def get_max_saved_index(output_dir: str, prefix="reps") -> int:
    files = [file for file in os.listdir(output_dir) if file.startswith(prefix)]
    index = [int(file.split(".")[0].split("-")[1]) for file in files]

    return max(index) if len(index) > 0 else -1

def get_max_saved_count(output_dir: str, prefix="reps") -> int:
    files = [file for file in os.listdir(output_dir) if file.startswith(prefix)]
    index = [int(file.split(".")[0].split("-")[1]) for file in files]

    return len(index) if len(index) > 0 else -1


def _project(full_grads, projected_grads, projectors, proj_dim, model_id=0):
    full_grads = torch.stack(full_grads).to(torch.float16)
    for i, projector in enumerate(projectors):
        current_projected_grads = projector.project(full_grads, model_id=model_id)

        projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

    return projected_grads

def get_trak_projector(device: torch.device, no_cuda=False):
    if no_cuda:
        return BasicProjector
    try:
        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        import fast_jl
        fast_jl.project_rademacher_8(torch.zeros(8, 1000, device=device), 512, 0, num_sms)
        return CudaProjector
    except:
        return BasicProjector

def _save(projected_grads, output_dirs, proj_dim, count, full_global_ids, save_prefix="grads"):

    for dim in proj_dim:
        if len(projected_grads[dim]) == 0:
            continue
        projected_grads[dim] = torch.cat(projected_grads[dim])
        assert len(full_global_ids) == len(projected_grads[dim]), f"len(full_global_ids): {len(full_global_ids)}, len(projected_grads[dim]): {len(projected_grads[dim])}"
        projected_grads_ids = {ids: grad for ids, grad in zip(full_global_ids, projected_grads[dim])}

        output_dir = output_dirs[dim]
        outfile = os.path.join(output_dir, f"{save_prefix}-{count}.pt")

        torch.save(projected_grads_ids, outfile)
        print(
            f"Saving {outfile}, {projected_grads[dim].shape}", flush=True)
        projected_grads[dim] = []
        full_global_ids = []


def _save_woproj(grads_list, output_dir, count, full_global_ids, save_prefix='grads_woproj'):

    if len(grads_list) == 0:
        return
    assert len(full_global_ids) == len(grads_list), f"len(full_global_ids): {len(full_global_ids)}, len(grads_list): {len(grads_list)}"
    projected_grads_ids = {ids: grad for ids, grad in zip(full_global_ids, grads_list)}

    outfile = os.path.join(output_dir, f"{save_prefix}-{count}.pt")

    torch.save(projected_grads_ids, outfile)
    print(
        f"Saving {outfile}, {len(grads_list)}", flush=True)
    grads_list = []
    full_global_ids = []



def merge_and_normalize_info(output_dir: str, prefix="reps"):
    pattern = os.path.join(output_dir, "**", f"{prefix}-*.pt")
    info = glob.glob(pattern, recursive=True)
    info.sort(key=lambda x: int(x.split(".")[-2].split("-")[-1]))

    merged_data = {}
    for file in info:
        data = torch.load(file)

        merged_data.update(data)

    merged_data = {g:normalize(d, dim=-1) for g, d in merged_data.items()}
    output_file = os.path.join(output_dir, f"all_orig.pt")
    torch.save(merged_data, output_file)
    print(f"Saving the normalized {prefix} (Shape: {len(merged_data)}) to {output_file}.")


def merge_info(output_dir: str, prefix="reps"):
    pattern = os.path.join(output_dir, "**", f"{prefix}-*.pt")
    info = glob.glob(pattern, recursive=True)
    info.sort(key=lambda x: int(x.split(".")[-2].split("-")[-1]))
    merged_data = {}
    for file in info:
        data = torch.load(file)
        merged_data.update(data)


    output_file = os.path.join(output_dir, f"all_unormalized.pt")
    torch.save(merged_data, output_file)
    print(f"Saving the unnormalized {prefix} (Shape: {len(merged_data)}) to {output_file}.")
