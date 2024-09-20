import os
import json
import argparse
import datetime
from tqdm import tqdm

from openai import OpenAI
import requests
import base64

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--inference_type", type=str, default="blind")

    # datasets
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument("--answer_path", type=str, required=True)

    args = parser.parse_args()

    if args.model_name == "gpt-4-tubro" and (args.inference_type != "caption" or args.inference_type != "narration"):
        raise ValueError(f"gpt-4-turbo does not support multi-image inference :(")

    return args
    
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def textual_inference(prompt, CLIENT, model="gpt-4o"):
    output = CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt}
        ]
    )
    return output.choices[0].message.content

def multi_image_inference(prompt, image_list, key, model="gpt-4o"):
    headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {key['api_key']}"
    }
    content = [{
        "type": "text",
        "text": prompt
    }]
    for path in tqdm(image_list, desc="Encoding Frames", leave=False):
        content.append({
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{encode_image(path)}"
            }
        })
    payload = {
        "model": model,
        "messages": [ { 
            "role": "user", 
            "content": content
        }],
        "max_tokens": 2205
    }
    response = requests.post(f"{key['api_base']}/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

def main(args):
    gpt_key =  {
        "api_key": args.api_key,
        "api_base": args.api_base 
    }
    gpt = OpenAI(api_key=gpt_key['api_key'], base_url=gpt_key['api_base'])

    task = []
    with open(args.dataset_path, "r") as f:
        task = json.load(f)
    
    model_answers = []
    ref_answers = []
    question_files = []
    q_id = 0

    for entry in tqdm(task, desc=f"Running {args.model_name} on dataset {os.path.basename(args.dataset_path).split('.')[0]}"):
        question = ""
        answer = ""
        pred = ""

        if args.inference_type == "caption":
            question = entry["caption_prompt"]
            answer = entry["answer"]
            pred = textual_inference(question, gpt, model=args.model_name)
        elif args.inference_type == "frames":
            question = entry["frames_prompt"]
            answer = entry["answer"]
            pred = multi_image_inference(question, entry["frame_list"], gpt_key, model=args.model_name)
        elif args.inference_type == "frames_caption":
            question = entry["frames_caption_prompt"]
            answer = entry["answer"]
            pred = multi_image_inference(question, entry["frame_list"], gpt_key, model=args.model_name)
        else:
            raise ValueError(f"After a cubersome search, we cannot locat the inference type {args.inference_type} :(")
        
        model_answers.append({
            "question_id" : q_id,
            "model_id" : args.model_name,
            "choices" : [{"index" : 0, "turns" : [pred]}]
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
        
    result_folder = args.answer_path
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    model_answer_folder = os.path.join(result_folder, 'model_answer')
    if not os.path.exists(model_answer_folder):
        os.makedirs(model_answer_folder)
    with open(os.path.join(model_answer_folder, f"{args.model_name}_{args.inference_type}.jsonl"), 'w') as f:
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