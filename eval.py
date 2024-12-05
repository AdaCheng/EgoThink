import os
import json
import argparse
import datetime

import torch
import numpy as np

from models import get_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import disable_caching
disable_caching()

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="blip2-7b")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=1)

    # datasets
    # parser.add_argument("--dataset-names", type=str, default=None)
    parser.add_argument('--annotation_path', type=str, default="/EgoThink/Activity/annotations.json")
    # result_path
    parser.add_argument("--answer_path", type=str, default="/answer/Activity")
    # parser.add_argument("--planning", action="store_true")

    args = parser.parse_args()
    return args


def eval_sample_dataset(dataset, dataset_name, max_sample_num=50, seed=0):
    if max_sample_num == -1:
        return dataset
    return sample_dataset(dataset, dataset_name, max_sample_num, seed)

def load_dataset(args):
    annotation_path = args.annotation_path
    with open(annotation_path, 'r') as f:
        dataset = json.load(f)
    for i, d in enumerate(dataset):
        image_filename = d['image_path'][0].split('/')[-1]
        dataset[i]['images'] = os.path.join(os.path.dirname(annotation_path), 'images', image_filename)
    return dataset

def get_generation_args(dataset_name):
    if dataset_name in ['assistance', 'navigation']:
        return {
            'max_new_tokens': 300,
            'planning': True
        }
    else:
        return {
            'max_new_tokens': 30,
            'planning': False
        }


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model_names = args.model_name.split(',')  

    # primary_abilities = args.primary_ability.split(',')
    # if primary_abilities[0] == 'all':
    #     primary_abilities = abilities.keys()
    # secondary_abilities = args.secondary_ability.split(',')
    time = datetime.datetime.now().strftime("%m%d-%H%M")
    # path: /${dataset}/annotations.json 
    dataset_name = os.path.dirname(args.annotation_path).split('/')[-1]

    for model_name in model_names:
        print(f"Running inference: {model_name}")

        if 'blip2' in model_name.lower() or 'llava' in model_name.lower():
            batch_size = 1
        else:
            batch_size = args.batch_size
        model = get_model(model_name, device=torch.device('cuda'))
        # time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # answer_path = f"{args.answer_path}/{args.model_name}"
        
        dataset = load_dataset(args)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})

        model_answers = []
        ref_answers = []
        question_files = []
        q_id = 0
        for batch in tqdm(dataloader, desc=f"Running inference: {model_name} on  {dataset_name}"):
            questions = batch['question']
            # print(questions)
            images = batch['images']
            if args.batch_size == 1:
                output = model.generate(images[0], questions[0], **get_generation_args(dataset_name))
                outputs = [output]
            else:
                outputs = model.batch_generate(images, questions, **get_generation_args(dataset_name))
            for i, (question, answer, pred) in enumerate(zip(batch['question'], batch['answer'], outputs)):
                # answer_dict={'question': questions, 'prediction': pred,
                # 'gt_answers': answer}
                # model_answers.append(answer_dict)
                model_answers.append({
                    'question_id': q_id,
                    'model_id': model_name,
                    'choices':[{'index': 0, "turns": [pred]}]
                })
                ref_answers.append({
                    'question_id': q_id,
                    'model_id': 'ground_truth',
                    'choices':[{'index': 0, "turns": [answer]}]
                })
                question_files.append({
                    'question_id': q_id,
                    'turns': [question]
                })
                q_id += 1
        torch.cuda.empty_cache()
        del model



        result_folder = args.answer_path
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        model_answer_folder = os.path.join(result_folder, 'model_answer')
        if not os.path.exists(model_answer_folder):
            os.makedirs(model_answer_folder)
        with open(os.path.join(model_answer_folder, f"{model_name}.jsonl"), 'w') as f:
            for pred in model_answers:
                f.write(json.dumps(pred) + '\n')
        
        ref_answer_folder = os.path.join(result_folder, 'reference_answer')
        if not os.path.exists(ref_answer_folder):
            os.makedirs(ref_answer_folder)
        with open(os.path.join(ref_answer_folder, "ground_truth.jsonl"), 'w') as f:
            for ref in ref_answers:
                f.write(json.dumps(ref) + '\n')
        
        with open(os.path.join(result_folder,  "question.jsonl"), 'w') as f:
            for q in question_files:
                f.write(json.dumps(q) + '\n')

        

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)