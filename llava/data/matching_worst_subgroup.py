import argparse
import os
import json

import torch
import random
import torch.nn.functional as F


argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training')

argparser.add_argument('--gradient_path', type=str, default="{} ckpt{}",
                       help='The path to the gradient file')
argparser.add_argument('--ckpts', type=int, nargs='+',
                       help="Checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                       help="checkpoint weights")
argparser.add_argument('--target_task_names', type=str,
                       nargs='+', help="The name of the target tasks")
argparser.add_argument('--validation_gradient_path', type=str,
                       default="{} ckpt{}", help='The path to the validation gradient file')
argparser.add_argument('--validation_loss_path', type=str)
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')

argparser.add_argument('--train_files', type=str, default=None,
                       help='The path of the training file that corresponds to the score file')
argparser.add_argument('--validation_files', type=str, default=None)
argparser.add_argument('--max_samples', type=int, nargs='+', default=None,
                       help='The maximum number of samples')
argparser.add_argument('--percentage', type=float, nargs='+', default=None,
                       help='The percentage of the data to be selected')
argparser.add_argument('--seed', type=int, default=3)
argparser.add_argument('--validation_sampling', type=str, choices=["top", "random", "all"], default='random')
argparser.add_argument('--validation_score', type=str, choices=["max", "mean"], default='max')

def count_lines(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        line_count = 0
        for line in file:
            line_count += 1
    return line_count


def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor, validation_info_global_ids, subgroup_loss_weight_dict, map_globalid_subgroup):
    from collections import defaultdict
    cosine_similarities = torch.matmul(training_info, validation_info.transpose(0, 1))
    subgroup_val_posids = defaultdict(list)
    for pos_idx, val_gids in enumerate(validation_info_global_ids):
        belong_subgroup = map_globalid_subgroup[val_gids]
        subgroup_val_posids[belong_subgroup].append(pos_idx)

    group_scores = []
    for group_name in subgroup_loss_weight_dict.keys():
        group_sample_pos_ids = subgroup_val_posids[group_name]
        group_cosine_sim = cosine_similarities[:, group_sample_pos_ids]
        if args.validation_score == "mean":
            group_avg_sim = group_cosine_sim.mean(dim=1, keepdim=True)
        elif args.validation_score == "max":
            group_avg_sim = group_cosine_sim.max(dim=1, keepdim=True)[0]
        else:
            raise NotImplementedError

        group_scores.append(group_avg_sim)

    group_scores = torch.hstack(group_scores)

    weighted_influence_scores = group_scores * torch.tensor(list(subgroup_loss_weight_dict.values())).reshape(1, -1)
    weighted_influence_scores = weighted_influence_scores.sum(-1)

    return weighted_influence_scores

def remove_samples(training_info, validation_info_global_ids, globalid_data_dict):
    filterd_globalids = globalid_data_dict.keys()
    filtered_training_info = {k: v for k, v in training_info.items() if (k not in validation_info_global_ids) and (k in filterd_globalids)}
    filtered_training_info_ids = list(filtered_training_info.keys())
    filtered_training_info_values = list(filtered_training_info.values())
    filtered_training_info_values = torch.vstack(filtered_training_info_values)
    filtered_training_info_values = filtered_training_info_values.to(device).float()

    return filtered_training_info_ids, filtered_training_info_values

def matching(args, target_task_name, globalid_data_dict, map_globalid_subgroup):

    influence_score = 0
    unweight_influence_score = 0

    validation_path = args.validation_gradient_path
    if os.path.isdir(validation_path):
        validation_path = os.path.join(validation_path, "all_orig.pt")
    validation_info_dict = torch.load(validation_path)

    validation_info = list(validation_info_dict.values())
    validation_info_global_ids = list(validation_info_dict.keys())

    if not torch.is_tensor(validation_info):
        validation_info = torch.vstack(validation_info)
    validation_info = validation_info.to(device).float()

    if os.path.isdir(args.validation_loss_path):
        args.validation_loss_path = os.path.join(args.validation_loss_path, "all_orig.pt")
    validation_loss_info_dict = torch.load(args.validation_loss_path)

    assert len(validation_info_dict) == len(validation_loss_info_dict)

    import numpy as np
    from collections import defaultdict
    sub_groups, counts = np.unique(list(map_globalid_subgroup.values()), return_counts=True)
    num_groups = len(sub_groups)
    print(f"\n=> num_groups: {num_groups}")

    subgroup_loss_dict = defaultdict(list)
    for val_gids, val_loss in validation_loss_info_dict.items():
        belong_subgroup = map_globalid_subgroup[val_gids]
        subgroup_loss_dict[belong_subgroup].append(val_loss)


    subgroup_loss_dict_mean = dict()
    for subgroup, val_loss_list in subgroup_loss_dict.items():
        subgroup_loss_dict_mean[subgroup] = np.mean(val_loss_list)

    subgroup_loss_weight_keys = list(subgroup_loss_dict_mean.keys())
    subgroup_loss_weight_values = list(subgroup_loss_dict_mean.values())
    subgroup_loss_weight_softmax = F.softmax(torch.tensor(subgroup_loss_weight_values), dim=-1).tolist()
    subgroup_loss_weight_dict = dict(zip(subgroup_loss_weight_keys, subgroup_loss_weight_softmax))

    gradient_path = args.gradient_path
    if os.path.isdir(gradient_path):
        gradient_path = os.path.join(gradient_path, "all_orig.pt")
    training_info_dict = torch.load(gradient_path)

    training_info_ids, training_info = remove_samples(training_info_dict, validation_info_global_ids, globalid_data_dict)
    print(f"training_info_ids after remove: {len(training_info_ids)}, training_info.shape: {training_info.shape}")

    ckpt_influence_score = calculate_influence_score(
        training_info=training_info,
        validation_info=validation_info, validation_info_global_ids=validation_info_global_ids,
        subgroup_loss_weight_dict=subgroup_loss_weight_dict,
        map_globalid_subgroup=map_globalid_subgroup
    )
    influence_score += ckpt_influence_score
    unweight_influence_score += ckpt_influence_score
    print(f"ckpt_influence_score shape: {ckpt_influence_score.shape}, mean: {ckpt_influence_score.mean():.6f}, max: {ckpt_influence_score.max():.6f}, min: {ckpt_influence_score.min():.6f}")

    print(f"ckpt: All, unweight_influence_score: {unweight_influence_score.mean():.6f}, max: {unweight_influence_score.max():.6f}, min: {unweight_influence_score.min():.6f}")


    output_dir = os.path.join(args.output_path, target_task_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(args.output_path, target_task_name, f"train_file_influence_score.pt")
    torch.save(influence_score, output_file)
    print("Saved influence score to {}".format(output_file))

    return training_info_ids, influence_score



if __name__ == "__main__":
    args = argparser.parse_args()


    device = "cpu"
    list_data_dict = json.load(open(args.train_files, "r"))
    globalid_data_dict = {d["global_id"]: d for d in list_data_dict}
    print(f"globalid_data_dict, len: {len(globalid_data_dict)}")

    list_data_dict_subgroup = json.load(open(args.validation_files, "r"))
    map_globalid_subgroup = {d["global_id"]: d['subgroup'] for d in list_data_dict_subgroup}

    for target_task_name in args.target_task_names:
        training_info_ids, influence_score = matching(args, target_task_name, globalid_data_dict, map_globalid_subgroup)

        assert args.percentage is not None or args.max_samples is not None
        device = "cpu"

        output_path = os.path.join(args.output_path, target_task_name)
        sorted_scores, sorted_index = torch.sort(influence_score, dim=0, descending=True)
        sorted_index = [training_info_ids[i] for i in sorted_index.tolist()]

        sorted_score_file = os.path.join(output_path, f"sorted.csv")
        if not os.path.exists(sorted_score_file):
            with open(sorted_score_file, 'w', encoding='utf-8') as file:
                file.write("file name, index, score\n")
                for score, index in zip(sorted_scores, sorted_index):
                    item = globalid_data_dict[index]
                    if "image" in item:
                        data_from = os.path.dirname(item["image"])
                    else:
                        data_from = "plain text"
                    file.write(f"{data_from}, {index}, {round(score.item(), 6)}\n")

        assert len(influence_score) == len(training_info_ids), f"len(influence_score): {len(influence_score)} != len(list_data_dict): {len(training_info_ids)}"
        total_samples = len(list_data_dict)
        if args.percentage is not None:
            for pert in args.percentage:
                args.max_samples = int(pert * total_samples)
                data_amount_name = f"p{pert}"

                final_index_list = sorted_index[:args.max_samples]
                selected_training_files = [globalid_data_dict[index] for index in final_index_list]

                output_file = os.path.join(output_path, f"top_{data_amount_name}.jsonl")
                with open(output_file, 'w') as file:
                    json.dump(selected_training_files, file, indent=4)

                print("Saved selected files to {}, length {}".format(output_file, len(selected_training_files)))
        else:
            for max_sample in args.max_samples:
                data_amount_name = f"num{max_sample}"

                final_index_list = sorted_index[:max_sample]
                selected_training_files = [globalid_data_dict[index] for index in final_index_list]

                output_file = os.path.join(output_path, f"top_{data_amount_name}.jsonl")
                with open(output_file, 'w') as file:
                    json.dump(selected_training_files, file, indent=4)

                print("Saved selected files to {}, length {}".format(output_file, len(selected_training_files)))
