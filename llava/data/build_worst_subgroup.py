import click
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import shortuuid
from collections import defaultdict
from PIL import Image
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.eval.model_vqa_science_option_attack import adv_permutation_attack
from llava.train.train import preprocess_multimodal, preprocess_v1


import json
import re
from copy import copy
from tqdm import tqdm
import click

from multiprocessing import Queue, Process, Manager

import random
from torchvision import transforms

def pil_to_tensor(image):
    transform = transforms.ToTensor()
    return transform(image)

def tensor_to_pil(tensor):
    transform = transforms.ToPILImage()
    return transform(tensor)

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000

    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step)
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step)

    return image_tensor_cd


option_mapping_qwer = {
    "A": "Q",
    "B": "W",
    "C": "E",
    "D": "R",
    "E": "T",
    "F": "Y",
    "G": "U"}

option_mapping_snvf = {
    "A": "S",
    "B": "N",
    "C": "V",
    "D": "F",
    "E": "I",
    "F": "G",
    "G": "H"}

def replace_options(text, mapping="original", use_bracket=False):
    def map_option(match):
        original = match.group(1)
        if mapping == "original":
            mapped_text = original
        elif mapping == "qwer":
            mapped_text = option_mapping_qwer[original]
        elif mapping == "snvf":
            mapped_text = option_mapping_snvf[original]
        else:
            raise NotImplementedError

        if use_bracket:
            mapped_text_format = "\n" + f"({mapped_text})"
        else:
            mapped_text_format = "\n" + mapped_text + "."

        return  mapped_text_format

    return re.sub(r'\n\b([A-G])\.', map_option, text)


def replace_ans(text, mapping="original", use_bracket=False):
    def map_option(match):
        original = match.group(1)

        if mapping == "original":
            mapped_text = original
        elif mapping == "qwer":
            mapped_text = option_mapping_qwer[original]
        elif mapping == "snvf":
            mapped_text = option_mapping_snvf[original]
        else:
            raise NotImplementedError
        if use_bracket:
            mapped_text_format = f"({mapped_text})"
        else:
            mapped_text_format = mapped_text
        return mapped_text_format
    return re.sub(r'\b([A-G])', map_option, text)

def transform_conversations(question, label, mapping, que_use_bracket=False, ans_use_bracket=False):
    new_question = replace_options(question, mapping=mapping, use_bracket=que_use_bracket)
    new_label = replace_ans(label, mapping=mapping, use_bracket=ans_use_bracket)
    return new_question, new_label


def format_prompt(line, question, model, args, format_name):
    if isinstance(question, dict):
        qs = question['value'].replace('<image>', '').strip()
    else:
        qs = question.replace('<image>', '').strip()
    cur_prompt = qs

    if 'image' in line:
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        cur_prompt = '<image>' + '\n' + cur_prompt

    if "Answer with the option's letter from the given choices directly" not in qs and format_name == "fmt_choice":
        qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt, cur_prompt


def model_infer(line, prompt, model, image_processor, tokenizer, args, image_noise=False):

    if 'image' in line:
        image_file = line["image"]
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')

        if image_noise:
            image_tensor = pil_to_tensor(image)
            image_noisy = add_diffusion_noise(image_tensor, 300)
            image = tensor_to_pil(image_noisy)

        image_tensor = process_images([image], image_processor, model.config)[0]
        images = image_tensor.unsqueeze(0).half().cuda()
        image_sizes = [image.size]
    else:
        images = None
        image_sizes = None


    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs

def model_infer_loss(line, model, image_processor, tokenizer, args, image_noise=False):
    # reuse
    #
    # recalculate
    if 'image' in line:
        has_image = True
        image_file = line["image"]
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        if image_noise:
            image_tensor = pil_to_tensor(image)
            image_noisy = add_diffusion_noise(image_tensor, 300)
            image = tensor_to_pil(image_noisy)
        image_tensor = process_images([image], image_processor, model.config)[0]
        images = image_tensor.unsqueeze(0).half().cuda()
        image_sizes = [image.size]

        convs = [line["conversations"]]
        args.is_multimodal = True
        args.mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
        sources = preprocess_multimodal(
            convs,
            args)
    else:
        has_image = False
        image_sizes = None
        crop_size = image_processor.crop_size
        images = torch.zeros(3, crop_size['height'], crop_size['width']).cuda()
        sources = [line["conversations"]]

    data_dict = preprocess_v1(
        sources,
        tokenizer,
        has_image=has_image)

    input_ids=data_dict["input_ids"].cuda()
    labels=data_dict["labels"].cuda()


    with torch.inference_mode():
        outputs = model(
            input_ids,
            images=images,
            labels=labels)
        loss = outputs.loss

    return loss.item()


def worker(format_name, items, i, result_queue, args):
    print(f"Initialize worker {i}...")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    for line in tqdm(items):
        if format_name == "fmt_choice":
            line_res = []
            for i in range(0, len(line['conversations']), 2):
                question = line['conversations'][i]['value']
                label = line['conversations'][i+1]['value']

                prompt, cur_prompt = format_prompt(line, question, model, args, format_name=format_name)
                ori_outputs = model_infer(line, prompt, model, image_processor, tokenizer, args)

                new_question, new_label = transform_conversations(question, label, mapping="qwer", que_use_bracket=False, ans_use_bracket=False)
                args.options = ["Q", "W", "E", "R", "T"]
                all_new_qs, all_new_labels = adv_permutation_attack(new_question, new_label, args)

                perturb_results = []
                for new_qs, new_ls in zip(all_new_qs, all_new_labels):
                    new_prompt, new_cur_prompt = format_prompt(line, new_qs, model, args, format_name=format_name)
                    new_outputs = model_infer(line, new_prompt, model, image_processor, tokenizer, args)
                    if new_outputs != new_ls:
                        perturb_results.append({"new_qs": new_qs, "new_ls": new_ls, "new_prompt": new_prompt, "new_outputs": new_outputs})
                        break

                line_res.append({"ori_outputs": ori_outputs,
                                 "perturb_results": perturb_results})

        elif format_name != "fmt_other_plain_text":
            ori_loss = model_infer_loss(line, model, image_processor, tokenizer, args, image_noise=False)
            new_loss = model_infer_loss(line, model, image_processor, tokenizer, args, image_noise=True)
            loss_change = abs(ori_loss - new_loss)
            line_res = [{"ori_loss": ori_loss, "new_loss": new_loss, "loss_change": loss_change}]
        else:
            perturb_results = []
            line_res = []

        result_queue.put({
            "line": line,
            "line_res": line_res
        })

def main(args):
    if args.seed is not None:
        random.seed(args.seed)

    list_data_dict = json.load(open(args.data_path, "r"))
    list_data_dict = {item['global_id']: item for item in list_data_dict}

    format_subgroup_sampled = json.load(open(args.subgroup_data_path, "r"))

    if args.score_method == "worst_group":
        format_data_dict = defaultdict(list)
        for format_name, subgroups in format_subgroup_sampled.items():
            for sub_group in subgroups:
                items = [list_data_dict[global_id] for global_id in sub_group]
                format_data_dict[format_name].append(items)

        format_data_results = defaultdict(list)
        for format_name, subgroups in format_data_dict.items():
            for items in subgroups:
                all_result = []
                nr_records = len(items)
                nr_split = math.ceil(nr_records / args.num_workers)
                result_queue = Queue(1000)
                procs = []

                for i in range(args.num_workers):
                    start = i * nr_split
                    end = min(start + nr_split, nr_records)
                    if start > end:
                        continue
                    split_records = items[start:end]
                    proc = Process(
                        target=worker,
                        args=(format_name, split_records, i, result_queue, args)
                    )
                    print('process:%d, start:%d, end:%d' % (i, start, end))
                    proc.start()
                    procs.append(proc)

                for i in tqdm(range(nr_records)):
                    result = result_queue.get()
                    all_result.append(result)

                for p in procs:
                    p.join()

                format_data_results[format_name].append(all_result)

        format_data_results_selected = defaultdict(list)
        for format_name, res_subgroups in format_data_results.items():
            for res_subgroup in res_subgroups:
                res_subgroups_wanted = []
                if format_name == "fmt_choice":
                    for res in res_subgroup:
                        for conv_res in res['line_res']:
                            if len(conv_res['perturb_results']):
                                res_subgroups_wanted.append(res)
                                break
                elif format_name != "fmt_other_plain_text":
                    res_subgroups_wanted = sorted(res_subgroup, key=lambda x: x['line_res'][0]['loss_change'], reverse=True)

                format_data_results_selected[format_name].append(res_subgroups_wanted)

        save_dir = os.path.dirname(args.subgroup_data_path)
        subgroup_top = args.subgroup_top
        worst_group_samples = []
        for format_name, res_subgroups_s in format_data_results_selected.items():
            if format_name in ['fmt_bbox_region', 'fmt_desc_region', 'fmt_phrase', 'fmt_other_has_img', 'fmt_one_img_caption']:
                for gp_idx, res_subgroup_s in enumerate(res_subgroups_s):
                    for item in res_subgroup_s[:subgroup_top]:

                        item_org = {
                            'id': item['line']['id'],
                            'image': item['line']['image'],
                            'conversations': item['line']['conversations'],
                            'global_id': item['line']['global_id'],
                            'subgroup': f"{format_name}_{gp_idx}"
                        }
                        worst_group_samples.append(item_org)

            elif format_name in ['fmt_choice']:
                for gp_idx, res_subgroup_s in enumerate(res_subgroups_s):
                    for item in res_subgroup_s[:subgroup_top]:
                        item_org = {
                            'id': item['line']['id'],
                            'image': item['line']['image'],
                            'conversations': item['line']['conversations'],
                            'global_id': item['line']['global_id'],
                            'subgroup': f"{format_name}_{gp_idx}"
                        }
                        worst_group_samples.append(item_org)

                        for conv_res in item['line_res']:
                            if len(conv_res['perturb_results']):
                                item_org2 = {
                                    'id': f"{item['line']['id']}_qwer_permu",
                                    'image': item['line']['image'],
                                    'conversations': [
                                        {'from': 'human', 'value': conv_res['perturb_results'][0]['new_qs']+"Answer with the option's letter from the given choices directly."},
                                        {'from': 'gpt', 'value': conv_res['perturb_results'][0]['new_ls']},
                                    ],
                                    'global_id': f"{item['line']['global_id']}_qwer_permu",
                                    'subgroup': f"{format_name}_{gp_idx}"
                                }

                                worst_group_samples.append(item_org2)

        with open(args.save_path, "w") as f:
            json.dump(worst_group_samples, f, indent=2)
        print(f"Save to {args.save_path}")

    else:
        raise NotImplementedError




if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--subgroup_data_path", type=str, default=None)
    parser.add_argument("--loss_data_path", type=str, default=None)
    parser.add_argument("--proxy_data_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument('--options', nargs='+', default=["A", "B", "C", "D", "E"])
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--subgroup_top", type=int, default=50)
    parser.add_argument("--sample_number", type=int, default=None)
    parser.add_argument("--score_method", type=str, default="worst_group", choices=["worst_group"])
    parser.add_argument("--seed", type=int, default=3)

    args = parser.parse_args()
    main(args)
