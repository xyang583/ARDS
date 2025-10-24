# written by yangxu

import json
import os
import random
import argparse
from glob import glob

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question-file', type=str)
    return parser.parse_args()


def convert_gqa_to_llava(question_file):
    with open(question_file) as file:
        questions = json.load(file)

    questions_llava_format = []
    for qid, cnts in questions.items():
        ques = cnts['question']
        image = cnts['imageId']
        item = {
            "question_id": qid,
            "image": f"{image}.jpg",
            "text": f"{ques}\nAnswer the question using a single word or phrase.",
            "category": "default",
        }

        questions_llava_format.append(item)

    return questions_llava_format


def save_question_list(question_list, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for item in question_list:
            f.write(json.dumps(item) + '\n')
    print(f"Save to {save_path}")


if __name__ == "__main__":
    args = get_args()


    if os.path.isdir(args.question_file):
        question_files = glob(f"{args.question_file}/*.json")
        for question_file in question_files:
            print(f"load from {question_file}")
            converted_question_list = convert_gqa_to_llava(question_file)
            print(f"len of converted_question_list: {len(converted_question_list)}")

            save_path = os.path.join(args.question_file,  f"llava_gqa_{os.path.basename(question_file)}")
            save_question_list(converted_question_list, save_path)


    elif os.path.isfile(args.question_file):
        question_file = args.question_file
        print(f"load from {question_file}")
        converted_question_list = convert_gqa_to_llava(question_file)
        print(f"len of converted_question_list: {len(converted_question_list)}")

        save_path = os.path.join(os.path.dirname(question_file), f"llava_gqa_{os.path.basename(question_file)}")
        save_question_list(converted_question_list, save_path)


    else:
        raise RuntimeError("Cannot find the file")
