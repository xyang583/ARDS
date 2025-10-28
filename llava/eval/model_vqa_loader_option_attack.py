import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import re
import numpy as np

from collections import defaultdict
import math
import itertools

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.eval.model_vqa_science_option_attack import adv_permutation_attack
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class CustomDataset(Dataset):
    def __init__(self, questions, annotations, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.annos = annotations
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        question_id = str(line['question_id'])
        label = self.annos[question_id]['answer']
        image_file = line["image"]
        qs = line["text"]

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = self.format_example(qs)

        all_new_input_ids = []
        all_new_qs, all_new_labels = adv_permutation_attack(qs, label, args)
        for qs in all_new_qs:
            new_input_ids = self.format_example(qs)
            all_new_input_ids.append(new_input_ids)
        return input_ids, label, image_tensor, image.size, all_new_input_ids, all_new_qs, all_new_labels

    def format_example(self, qs):
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        if "Answer with the option's letter from the given choices directly" not in qs:
            if qs.endswith('\n'):
                qs = qs + "Answer with the option's letter from the given choices directly."
            else:
                qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, labels, image_tensors, image_sizes, all_new_input_ids, all_new_qs, all_new_labels = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, labels, image_tensors, image_sizes, all_new_input_ids, all_new_qs, all_new_labels
def create_data_loader(questions, annotations, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, annotations, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def model_infer(model, tokenizer, input_ids, image_tensor, image_sizes, args):
    input_ids = input_ids.to(device='cuda', non_blocking=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    annotations = json.load(open(args.annotation_file))
    annotations = {annotation['question_id']: annotation for annotation in annotations['questions']}

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, annotations, args.image_folder, tokenizer, image_processor, model.config, num_workers=args.num_workers)

    cors = []
    for (input_ids, label, image_tensor, image_sizes, all_new_input_ids, all_new_qs, all_new_labels), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        label = label[0]

        ori_outputs = model_infer(model, tokenizer, input_ids, image_tensor, image_sizes, args)

        corr = 1
        attack_results = []
        for perm_i, (new_input_ids, new_qs, new_label) in enumerate(zip(all_new_input_ids[0], all_new_qs[0], all_new_labels[0])):
            outputs = model_infer(model, tokenizer, new_input_ids.unsqueeze(0), image_tensor, image_sizes, args)
            attack_results.append({
                "perm_i": perm_i,
                "prompt": new_qs,
                "label": new_label,
                "text": outputs,
            })
            if outputs != new_label:
                corr = 0
                break
        cors.append(corr)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": ori_outputs,
                                   "label": label,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {},
                                   "attack_results": attack_results}) + "\n")
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--annotation-file", type=str, default="./playground/data/eval/seed_bench/SEED-Bench.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument('--options', nargs='+', default=["A", "B", "C", "D", "E"])
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
