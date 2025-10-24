import numpy as np
import time
import torch

import faiss
# require faiss-gpu=1.8.0
import os

import json
from collections import defaultdict
import click
import json
import random
import os
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import math


format_pool = [
    "Answer the question using a single word or phrase.",
    "Answer with the option's letter from the given choices directly.",
    "Provide a one-sentence caption for the provided image",
    "Provide a short description for this region",
    "Provide the bounding box coordinate of the region this sentence describes",
]

format_abbr = {
    "Answer the question using a single word or phrase.": "fmt_phrase",
    "Answer with the option's letter from the given choices directly.": "fmt_choice",
    "Provide a one-sentence caption for the provided image": "fmt_one_img_caption",
    "Provide a short description for this region": "fmt_desc_region",
    "Provide the bounding box coordinate of the region this sentence describes": "fmt_bbox_region",
    "others_has_image": "fmt_other_has_img",
    "others_plain_text": "fmt_other_plain_text"
}



def main(vector_path, dataset_file, n_components, n, niter):
    vector_base = torch.load(vector_path)
    list_data_dict = json.load(open(dataset_file, "r"))

    map_id_to_dataset = {}
    for item in tqdm(list_data_dict):
        if "image" in item:
            has_image = True
        else:
            has_image = False

        first_turn_qs = item['conversations'][0]['value']
        other_flag = True
        for fmt in format_pool:
            if fmt.lower() in first_turn_qs.lower():
                cur_format = fmt
                other_flag = False
                break

        if other_flag:
            if has_image:
                cur_format = "others_has_image"
            else:
                cur_format = "others_plain_text"

        map_id_to_dataset[item["global_id"]] = format_abbr[cur_format]

    format_reprs = defaultdict(list)
    format_global_ids = defaultdict(list)

    for global_id, repr in vector_base.items():
        format = map_id_to_dataset[global_id]
        format_reprs[format].append(repr)
        format_global_ids[format].append(global_id)

    format_subgroup_sampled = dict()

    for format_name, reprs in format_reprs.items():

        features = torch.stack(reprs)

        global_ids = np.array(format_global_ids[format_name])
        assert len(global_ids) == len(reprs)
        kmeans = faiss.Kmeans(features.shape[1], n_components, niter=niter, verbose=True, spherical=True, gpu=True, seed=42)
        kmeans.train(features.numpy())
        D, I = kmeans.index.search(features.numpy(), 1)
        clusters, counts = np.unique(I, return_counts=True)
        sorted_idx = np.argsort(-counts)
        sorted_idx = sorted_idx[counts[sorted_idx] > 2]
        n_per_cluster = n // len(sorted_idx)
        sampled_global_ids = []
        for i in range(len(sorted_idx)):

            indices = np.where(I == clusters[sorted_idx[i]])[0]

            if len(indices) > n_per_cluster:
                indices_random = np.random.choice(indices, n_per_cluster, replace=False)
                n -= n_per_cluster
            else:
                indices_random = indices
                n -= len(indices)

            sampled_global_ids.append(list(global_ids[indices_random]))

        # if n > 0:
        #     print(f"K-means: {n} samples left to sample from clusters with size <= 2")
        #     clusters_to_sample = clusters[np.where(counts <= 2)[0]]
        #     indices = np.where(np.isin(I, clusters_to_sample))[0]
        #     sampled_indices.append(np.random.choice(indices, n, replace=False))


        format_subgroup_sampled[format_name] = sampled_global_ids


    # if save_path is not None:
    save_path = os.path.join(os.path.dirname(vector_path), f"kmeans.json")
    with open(save_path, "w") as f:
        json.dump(format_subgroup_sampled, f, indent=2)
        print(f"save to {save_path}")

