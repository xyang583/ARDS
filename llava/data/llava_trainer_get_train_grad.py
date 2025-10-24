import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
from collections import defaultdict

from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from llava.data.collect_grad_reps import _project, get_trak_projector
from llava.data.collect_grad_reps import _save, _save_woproj, get_max_saved_index, get_max_saved_count, prepare_optimizer_state
from llava.utils import get_optimizer_state, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    @torch.no_grad()
    def collect_reps(self, max_samples=None, num_chunks=1, chunk_idx=0, save_prefix=None, selection_strategy="last_token"):
        save_interval = 10
        torch.random.manual_seed(0)

        self.model.train()

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        count = 0
        all_reps = []
        full_global_ids = []
        if save_prefix is None:
            output_dir = os.path.join(self.args.output_dir, "reps", "last_layer_last_token", f"{num_chunks}_{chunk_idx}")
        else:
            output_dir = os.path.join(self.args.output_dir, "reps", save_prefix, f"{num_chunks}_{chunk_idx}")
        os.makedirs(output_dir, exist_ok=True)
        max_index = get_max_saved_index(output_dir, "reps")
        print(f"=> max_index: {max_index}")

        for inputs in tqdm(self.get_train_dataloader(), total=len(self.get_train_dataloader())):
            inputs = self._prepare_inputs(inputs)
            global_ids = inputs.pop("global_ids")
            for k in inputs.keys():
                if torch.is_floating_point(inputs[k]) or torch.is_complex(inputs[k]):
                    inputs[k] = inputs[k].to(dtype)

            count += 1

            if count <= max_index:
                print("skipping count", count)
                continue

            with torch.inference_mode():
                inputs["output_attentions"] = True
                inputs["output_hidden_states"] = True
                loss, outputs = self.compute_loss(self.model, inputs, return_outputs=True)

                hidden_states = outputs.hidden_states
                attentions = outputs.attentions

                batchsize = len(inputs['input_ids'])
                ids = torch.arange(batchsize, device=inputs['input_ids'].device)
                pos = inputs['attention_mask'].sum(dim=1) - 1
                num_image_tokens = (inputs['input_ids'] == -200).sum(dim=1)
                pos += num_image_tokens * 575
                pos = torch.where(pos >= self.args.model_max_length, self.args.model_max_length-1, pos)

                assert all(pos < hidden_states[-1].shape[1]), f"pos: {pos} not < hidden_states[-1].shape[1]: {hidden_states[-1].shape[1]}"

                reps = hidden_states[-1][ids, pos].float()

                selected_tokens = []
                selected_tokens.append(reps)

                last_hidden_states = hidden_states[-1]
                last_layer_attn = attentions[-1].mean(dim=1)

                batch_weighted_aggregation = []
                for sam_pos, sam_attn, sam_hidd in zip(pos, last_layer_attn, last_hidden_states):
                    sam_attn_last_tok = sam_attn[sam_pos, :sam_pos]
                    sam_attn_last_tok = sam_attn_last_tok / sam_attn_last_tok.sum(dim=-1, keepdim=True)
                    sam_hidd_pre_last_tok = sam_hidd[:sam_pos, :]
                    weighted_tokens = sam_hidd_pre_last_tok * sam_attn_last_tok.unsqueeze(-1)
                    weighted_aggregation = torch.sum(weighted_tokens, dim=0)
                    batch_weighted_aggregation.append(weighted_aggregation)

                batch_weighted_aggregation = torch.stack(batch_weighted_aggregation)
                selected_tokens.append(batch_weighted_aggregation)
                selected_tokens = [nn.functional.normalize(x, dim=-1) for x in selected_tokens]
                selected_tokens = torch.cat(selected_tokens, dim=-1)

                reps = selected_tokens

            all_reps.extend([d for d in reps.detach().cpu()])
            full_global_ids.extend(global_ids)

            if count % save_interval == 0:
                _save_woproj(all_reps, output_dir, count, full_global_ids, save_prefix="reps")
                all_reps = []
                full_global_ids = []
                torch.cuda.empty_cache()

            if max_samples is not None and count >= max_samples:
                break

        if len(all_reps):
            _save_woproj(all_reps, output_dir, count, full_global_ids, save_prefix="reps")
            all_reps = []
            full_global_ids = []
            torch.cuda.empty_cache()

        print("Finished")

    @torch.no_grad()
    def collect_loss(self, max_samples=None, num_chunks=1, chunk_idx=0, save_prefix=None):
        save_interval = 160
        torch.random.manual_seed(0)

        self.model.train()

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        count = 0
        all_loss = []
        full_global_ids = []
        if save_prefix is None:
            output_dir = os.path.join(self.args.output_dir, "loss", "standard_training_loss", f"{num_chunks}_{chunk_idx}")
        else:
            output_dir = os.path.join(self.args.output_dir, "loss", save_prefix, f"{num_chunks}_{chunk_idx}")
        os.makedirs(output_dir, exist_ok=True)
        max_index = get_max_saved_index(output_dir, "loss")
        print(f"=> max_index: {max_index}")

        for inputs in tqdm(self.get_train_dataloader(), total=len(self.get_train_dataloader())):
            inputs = self._prepare_inputs(inputs)
            global_ids = inputs.pop("global_ids")
            for k in inputs.keys():
                if torch.is_floating_point(inputs[k]) or torch.is_complex(inputs[k]):
                    inputs[k] = inputs[k].to(dtype)
            count += 1

            if count <= max_index:
                print("skipping count", count)
                continue
            with torch.inference_mode():
                loss = self.compute_loss(self.model, inputs)


            all_loss.append(loss.item())
            del loss
            torch.cuda.empty_cache()
            full_global_ids.extend(global_ids)

            if count % save_interval == 0:
                _save_woproj(all_loss, output_dir, count, full_global_ids, save_prefix="loss")
                all_loss = []
                full_global_ids = []
                torch.cuda.empty_cache()

            if max_samples is not None and count >= max_samples:
                break

        if len(all_loss):
            _save_woproj(all_loss, output_dir, count, full_global_ids, save_prefix="loss")
            all_loss = []
            full_global_ids = []
            torch.cuda.empty_cache()

        print("Finished")


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)


            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

        elif self.args.lora_enable:
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
            os.makedirs(output_dir, exist_ok=True)

            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

            config_use_cache = self.model.config.use_cache
            self.model.config.use_cache = True

            state_dict = get_peft_state_maybe_zero_3(
                self.model.named_parameters(), self.args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters()
            )
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                self.model.save_pretrained(output_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))

            torch.cuda.empty_cache()
            optimzier_state = get_optimizer_state(self.model.named_parameters(), trainer=self)
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                try:
                    optimzier_dict = {
                        "state": optimzier_state,
                        "param_groups": self.optimizer.state_dict()['optimizer_state_dict']['param_groups'],
                    }
                except:
                    optimzier_dict = {
                        "state": optimzier_state,
                        "param_groups": self.optimizer.state_dict()['base_optimizer_state']['param_groups'],
                    }
                torch.save(optimzier_dict, os.path.join(output_dir, 'optimizer.bin'))
            del state_dict, non_lora_state_dict, optimzier_state
            torch.cuda.empty_cache()

            self.model.config.use_cache = config_use_cache
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
