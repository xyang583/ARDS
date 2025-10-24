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

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_pred_idx(prediction, options):
    if prediction in options:
        return options.index(prediction)
    else:
        return -1


def combine_options(text, ori_options, new_options, args=None):

    def map_option(match):
        original_cont = match.group(2).strip()
        pos_index = ori_options.index(original_cont)
        new_option_cont = new_options[pos_index]

        return match.group(0).replace(original_cont, new_option_cont)

    option_space = ",".join(args.options)
    return re.sub(rf'(\n[{option_space}])\.(.*?)(?=\n[{option_space}]|$)', map_option, text, flags=re.DOTALL)

def adv_permutation_attack(question, label, args=None):
    if isinstance(question, dict):
        question = question['value'].replace('<image>', '').strip()
    else:
        question = question.replace('<image>', '').strip()
    question = question.replace("Answer with the option's letter from the given choices directly.", "")

    option_space = ",".join(args.options)
    options = re.findall(rf'(\n[{option_space}])\.(.*?)(?=\n[{option_space}]|$)', question, flags=re.DOTALL)
    option_letters = [opt[0].strip() for opt in options]
    option_texts = [opt[1].strip() for opt in options]

    num_choices = len(option_letters)
    total_perms = math.factorial(num_choices)
    perm_list = list(itertools.permutations(range(num_choices)))

    new_questions = []
    new_labels = []
    for permutation in perm_list:
        new_choices = [option_texts[i] for i in permutation]
        new_question = combine_options(question, option_texts, new_choices, args)
        new_label = option_letters[new_choices.index(option_texts[option_letters.index(label)])]

        new_questions.append(new_question)
        new_labels.append(new_label)
    return new_questions, new_labels

def format_prompt(line, question, model, args):
    if isinstance(question, dict):
        qs = question['value'].replace('<image>', '').strip()
    else:
        qs = question
    cur_prompt = qs

    if 'image' in line:
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        cur_prompt = '<image>' + '\n' + cur_prompt

    if args.single_pred_prompt:
        qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

        cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt, cur_prompt

def model_infer(line, prompt, model, image_processor, tokenizer, args):

    if 'image' in line:
        image_file = line["image"]
        image = Image.open(os.path.join(args.image_folder, image_file))
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
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    results = {'correct': [], 'incorrect': []}
    cors = []

    for i, line in enumerate(tqdm(questions)):
        if 'image' not in line and args.eval_img:
            continue

        idx = line["id"]
        question = line['conversations'][0]
        label = line['conversations'][1]['value']

        prompt, cur_prompt = format_prompt(line, question, model, args)
        ori_outputs = model_infer(line, prompt, model, image_processor, tokenizer, args)

        corr = 1
        attack_results = []
        all_new_qs, all_new_labels = adv_permutation_attack(question, label, args)
        for perm_i, (new_qs, new_label) in enumerate(zip(all_new_qs, all_new_labels)):
            new_prompt, new_cur_prompt = format_prompt(line, new_qs, model, args)
            outputs = model_infer(line, new_prompt, model, image_processor, tokenizer, args)
            attack_results.append({
                "perm_i": perm_i,
                "label": new_label,
                "text": outputs,
            })
            if outputs != new_label:
                corr = 0
                break
        cors.append(corr)

        ans_id = shortuuid.uuid()
        analysis = {"question_id": idx,
                   "prompt": cur_prompt,
                   "answer_id": ans_id,
                   "text": ori_outputs,
                   "label": label,
                   "model_id": model_name,
                   "metadata": {},
                   "is_multimodal": 'image' in line,
                   "attack_results": attack_results}

        if ori_outputs == label:
            results['correct'].append(analysis)
        else:
            results['incorrect'].append(analysis)

        ans_file.write(json.dumps(analysis) + "\n")
        ans_file.flush()
    ans_file.close()
    correct = len(results['correct'])
    total = len(results['correct']) + len(results['incorrect'])

    multimodal_correct = len([x for x in results['correct'] if x['is_multimodal']])
    multimodal_incorrect = len([x for x in results['incorrect'] if x['is_multimodal']])
    multimodal_total = multimodal_correct + multimodal_incorrect

    print(f'\nTotal: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%')
    print(f'Total w/ im: {multimodal_total}, Correct: {multimodal_correct}, IMG-Accuracy: {multimodal_correct / multimodal_total * 100:.2f}%')
    if args.eval_img:
        imgs_cors = cors
    else:
        assert len(cors) == len(questions)
        imgs_cors = [corr for x, corr in zip(questions, cors) if 'image' in x]
    print(f'\nTotal: {len(cors)}, Correct: {np.sum(cors)}, Attack Accuracy: {np.mean(cors) * 100:.2f}%')
    print(f'Total w/ im: {len(imgs_cors)}, Correct: {np.sum(imgs_cors)}, Attack IMG-Accuracy: {np.mean(imgs_cors) * 100:.2f}%')


def read_log_cal_acc(args):
    answers_file = os.path.expanduser(args.answers_file)
    ans_file = open(answers_file, "r")
    lines = [json.loads(line.strip()) for line in ans_file.readlines()]

    results = {'correct': [], 'incorrect': []}
    cors = []

    for i, line in enumerate(tqdm(lines)):
        pred_text = line['text']
        answer = line['parsed_ans']
        label = line['label']
        attack_results = line['attack_results']

        if answer == label:
            results['correct'].append(line)
        else:
            results['incorrect'].append(line)

        corr = 1
        for attack_res in attack_results:
            new_text = attack_res['text']
            parsed_ans = attack_res['parsed_ans']
            new_label = attack_res['label']

            if parsed_ans != new_label:
                corr = 0
                break

        cors.append(corr)
    correct = len(results['correct'])
    total = len(results['correct']) + len(results['incorrect'])

    multimodal_correct = len([x for x in results['correct'] if x['is_multimodal']])
    multimodal_incorrect = len([x for x in results['incorrect'] if x['is_multimodal']])
    multimodal_total = multimodal_correct + multimodal_incorrect

    clean_acc = multimodal_correct / multimodal_total

    print(f'\nTotal: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%')
    print(f'Total w/ im: {multimodal_total}, Correct: {multimodal_correct}, IMG-Accuracy: {clean_acc * 100:.2f}%')
    assert len(cors) == len(lines)
    imgs_cors = [corr for x, corr in zip(lines, cors) if x['is_multimodal']]
    attack_acc = np.mean(imgs_cors)
    print(f'\nTotal: {len(cors)}, Correct: {np.sum(cors)}, Attack Accuracy: {np.mean(cors) * 100:.2f}%')
    print(f'Total w/ im: {len(imgs_cors)}, Correct: {np.sum(imgs_cors)}, Attack IMG-Accuracy: {attack_acc * 100:.2f}%')

    return clean_acc, attack_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument('--options', nargs='+', default=["A", "B", "C", "D", "E"])
    args = parser.parse_args()

    eval_model(args)
    read_log_cal_acc(args)
