import os
import json
import argparse
import datetime
import warnings
import torch
import numpy as np

import sys
import os
sys.path.append('./models')

from models import get_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import disable_caching
disable_caching()

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="BLIP2")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=1)

    # datasets
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument('--annotation_path', type=str, required=True)
    # result_path
    parser.add_argument("--answer_path", type=str, default="./tiny_answers")
    parser.add_argument("--data_mode", type=str, default="VIDEO", help="You can chooce PICTURE or VIDEO")
    # parser.add_argument("--planning", action="store_true")

    args = parser.parse_args()
    return args


def load_dataset(args):
    annotation_path = args.annotation_path
    with open(annotation_path, 'r') as f:
        dataset = json.load(f)
    for i, d in enumerate(dataset):
        if len(d['image_path']) != 0:
            image_filename = d['image_path'][0].split('/')[-1]
        else:
            image_filename = ""
            warnings.warn("No image in image dataset path", UserWarning)
        dataset[i]['images'] = os.path.join(os.path.dirname(annotation_path), 'images', image_filename)
    return dataset

def get_generation_args(dataset_name):
    print(f"---------function dataset name {dataset_name}------------")
    if dataset_name == "tg":
        return {
            'max_new_tokens': 128,
            'planning': False,
            'tg': True,
            'rm_critique': False,
            'rm_feedback': False,
            'hp_h2m':False
        }
    elif dataset_name == "rm_critique":
        return {
            'max_new_tokens': 128,
            'planning': False,
            'tg': False,
            'rm_critique': True,
            'rm_feedback': False,
            'hp_h2m':False
        }
    elif dataset_name == "rm_feedback":
        return {
            'max_new_tokens': 128,
            'planning': False,
            'tg': False,
            'rm_critique': False,
            'rm_feedback': True,
            'hp_h2m':False
        }
    elif dataset_name in ['assistance', 'navigation']:
        return {
            'max_new_tokens': 300,
            'planning': True,
            'tg': False,
            'rm_critique': False,
            'rm_feedback': False,
            'hp_h2m':False
        }
    elif dataset_name == 'hp_h2m':
        return {
            'max_new_tokens': 300,
            'planning': False,
            'tg': False,
            'rm_critique': False,
            'rm_feedback': False,
            'hp_h2m':True
        }
    elif dataset_name == 'hp_m2l':
        return {
            'max_new_tokens': 300,
            'planning': False,
            'tg': False,
            'rm_critique': False,
            'rm_feedback': False,
            'hp_h2m':False,
            'hp_m2l':True
        }
    else:
        return {
            'max_new_tokens': 30,
            'planning': False,
            'tg': False,
            'rm_critique': False,
            'rm_feedback': False,
            'hp_h2m': False
        }

def get_dataset_name(annotation_path):
    folder_name = os.path.dirname(annotation_path).split('/')[-1]
    file_name = annotation_path.split('/')[-1]
    dataset_name = folder_name
    if folder_name == 'rm':
        if 'critique' in file_name:
            dataset_name = 'rm_critique'
        elif 'feedback' in file_name:
            dataset_name = 'rm_feedback'
        else:
            raise ValueError("Invalid rm file name.")
    elif folder_name == 'hp':
        if 'h2m' in file_name:
            dataset_name = 'hp_h2m'
        elif 'm2l' in file_name:
            dataset_name = 'hp_m2l'
        else:
            raise ValueError("Invalid hp file name.")
    elif folder_name == 'vqa':
        dataset_name = 'vqa'
    elif folder_name == 'tg':
        dataset_name = 'tg'
    elif folder_name == 'og':
        dataset_name = 'og'
    elif folder_name == 'fg':
        dataset_name = 'fg'
    else:
        raise ValueError("Invalid file name.")
    return dataset_name

def get_video_path(dataset_name):
    # video_folder = "/apdcephfs_cq10/share_1150325/csj/videgothink/goalstep_val_clean"
    video_folder = "/apdcephfs_cq10/share_1150325/csj/videgothink/goalstep_val_hp_video"
    if 'vqa' in dataset_name:
        video_folder = video_folder
    elif 'rm' in dataset_name:
        video_folder = video_folder + "/rm"
    elif 'hp' in dataset_name:
        video_folder = video_folder
    elif 'tg' in dataset_name:
        video_folder = video_folder + "/tg"
    elif 'og' in dataset_name:
        video_folder = video_folder + "/og"
    elif 'fg' in dataset_name:
        video_folder = video_folder + "/fg"
    else:
        raise ValueError("Invalid video path.")
    return video_folder


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model_names = args.model_name.split(',')  

    # primary_abilities = args.primary_ability.split(',')
    # if primary_abilities[0] == 'all':
    #     primary_abilities = abilities.keys()
    # secondary_abilities = args.secondary_ability.split(',')
    time = datetime.datetime.now().strftime("%m%d-%H%M")
    # path: /${dataset}/annotations.json 
    # if "hp_m2l" in args.answer_path:
    #     dataset_name = 'hp_m2l'
    # elif "hp_h2m" in args.answer_path:
    #     dataset_name = 'hp_h2m'
    # else:
    #     dataset_name = 'Other'
    dataset_name = args.dataset_name

    print(f"dataset name: {dataset_name}")
    print(f"Running inference on {dataset_name}")

    for model_name in model_names:
        print(f"Running inference: {model_name}")

        if 'blip2' in model_name.lower() or 'llava' in model_name.lower():
            batch_size = 1
        else:
            batch_size = args.batch_size
        device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        model = get_model(model_name, device=device)
        # time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # answer_path = f"{args.answer_path}/{args.model_name}"
        
        dataset = load_dataset(args)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})

        model_answers = []
        ref_answers = []
        question_files = []
        q_id = 0
        for batch in tqdm(dataloader, desc=f"Running inference: {model_name} on  {dataset_name}"):
            questions = batch['question']
            # print(questions)
            images = batch['images']
            if args.data_mode == "VIDEO":
                video_path = get_video_path(dataset_name)
                print(f"video path: {video_path}")
                images = [ video_path + "/" + x for x in batch['video_path']]

            print(f"args.data_mode:{args.data_mode} , dataset_name: {dataset_name}, Images:{images}")     
            if args.batch_size == 1:
                try:
                    output = model.generate(images[0], questions[0], **get_generation_args(dataset_name))
                    print(output)
                    outputs = [output]
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error: {e}")
                    outputs = ['']
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