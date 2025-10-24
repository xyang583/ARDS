import os
import json
import argparse
import numpy as np
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--result-upload-file", type=str)
    return parser.parse_args()


def eval_single(result_file, eval_only_type=None):
    results = {}
    for line in open(result_file):
        row = json.loads(line)
        results[row['question_id']] = row

    type_counts = {}
    correct_counts = {}
    for question_data in data['questions']:

        if eval_only_type is not None and question_data['data_type'] != eval_only_type: continue
        data_type = question_data['question_type_id']
        type_counts[data_type] = type_counts.get(data_type, 0) + 1
        try:
            question_id = int(question_data['question_id'])
        except:
            question_id = question_data['question_id']
        if question_id not in results:
            correct_counts[data_type] = correct_counts.get(data_type, 0)
            continue

        row = results[question_id]
        if row['text'] == question_data['answer']:
            correct_counts[data_type] = correct_counts.get(data_type, 0) + 1

    total_count = 0
    total_correct = 0
    for data_type in sorted(type_counts.keys()):
        accuracy = correct_counts.get(data_type, 0) / type_counts[data_type] * 100
        if eval_only_type is None:
            print(f"{ques_type_id_to_name[data_type]}: {accuracy:.2f}%")

        total_count += type_counts[data_type]
        total_correct += correct_counts.get(data_type, 0)

    total_accuracy = total_correct / total_count * 100
    if eval_only_type is None:
        print(f"Total: {total_count}, Total accuracy: {total_accuracy:.2f}%")
    else:
        print(f"Total: {total_count}, {eval_only_type} accuracy: {total_accuracy:.2f}%")

    return results


def eval_single_adv(result_file, eval_only_type=None):
    results = {}
    for line in open(result_file):
        row = json.loads(line)
        results[row['question_id']] = row

    type_counts = {}
    correct_counts = {}
    cors = defaultdict(list)
    for question_data in data['questions']:
        if eval_only_type is not None and question_data['data_type'] != eval_only_type: continue
        data_type = question_data['question_type_id']
        type_counts[data_type] = type_counts.get(data_type, 0) + 1
        try:
            question_id = int(question_data['question_id'])
        except:
            question_id = question_data['question_id']
        if question_id not in results:
            correct_counts[data_type] = correct_counts.get(data_type, 0)
            continue

        row = results[question_id]
        if row['text'] == question_data['answer']:
            correct_counts[data_type] = correct_counts.get(data_type, 0) + 1

        if "attack_results" in row:
            corr = 1
            for attack_res in row['attack_results']:
                new_text = attack_res['text']
                new_label = attack_res['label']

                if new_text != new_label:
                    corr = 0
                    break

            cors[data_type].append(corr)

    if len(cors):
        for data_type in sorted(type_counts.keys()):
            accuracy = np.mean(cors[data_type]) * 100
            if eval_only_type is None:
                print(f"{ques_type_id_to_name[data_type]}: {accuracy:.2f}%")

        total_cors = np.concatenate([v for v in cors.values()])
        total_accuracy = np.mean(total_cors) * 100
        if eval_only_type is None:
            print(f"Total: {len(total_cors)}, Total Attack accuracy: {total_accuracy:.2f}%")
        else:
            print(f"Total: {len(total_cors)}, {eval_only_type} Attack accuracy: {total_accuracy:.2f}%")

    return results

if __name__ == "__main__":
    args = get_args()
    data = json.load(open(args.annotation_file))
    ques_type_id_to_name = {id:n for n,id in data['question_type'].items()}

    results = eval_single(args.result_file)
    eval_single(args.result_file, eval_only_type='image')
    eval_single(args.result_file, eval_only_type='video')

    os.makedirs(os.path.dirname(args.result_upload_file), exist_ok=True)
    with open(args.result_upload_file, 'w') as fp:
        for question in data['questions']:
            qid = question['question_id']
            if qid in results:
                result = results[qid]
            else:
                result = results[int(qid)]
            fp.write(json.dumps({
                'question_id': qid,
                'prediction': result['text']
            }) + '\n')

    print("\n\nattack accuracy")
    adv_results = eval_single_adv(args.result_file)
    eval_single_adv(args.result_file, eval_only_type='image')
    eval_single_adv(args.result_file, eval_only_type='video')
