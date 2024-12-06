import os
import json
import argparse
import datetime
from tqdm import tqdm

from openai import AzureOpenAI
import openai
import requests
import base64
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class LLM_API:
    def __init__(self, model, base_url=None, temperature=0.0, stop=None):
        self.model = model
        self.temperature = temperature
        self.n_repeat = 1
        self.stop = stop

        if "gpt" in model.lower():
            self.api_key = ""
            self.client = AzureOpenAI(
                    api_key=self.api_key,  
                    api_version="2024-02-01",
                    azure_endpoint = ""
                    )
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        else:
            assert base_url is not None
            self.client = OpenAI(api_key="ss", base_url=base_url)

    def request_general(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=self.temperature,
            stop=self.stop,
        )
        return response.choices[0].message.content
    
    def request_vision(self, img_dir, prompt):
        vision_messages = [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(image)}"
            }
        } for image in [img_dir]]

        content = [{
            "type": "text",
            "text": prompt,
        }]
        for message in vision_messages:
            content.append(message)
        all_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content},
            ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            stream=False,
            temperature=self.temperature,
            stop=self.stop,
        )
        return response.choices[0].message.content

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

def generate_item(args, item):  
    q_id = item["q_id"]
    question = item["question"]
    images = item["images"]
    answer = item["answer"]

    prompt = "Please let your answer be as short as possible. Question: {question} Short answer:".format(question=question)

    max_retries = 5  # 最大重试次数
    retry_delay = 2  # 重试之间的延时，单位为秒
    attempt = 0  # 当前尝试次数
    while True:
        try:
            output = llm_api.request_vision(images, prompt)
            if "Short Answer: " in output:
                output = output.split("Short Answer: ")[1]
            print(output)
            break
        except Exception as e:
            # print(e)
            if attempt >= max_retries:
                print(e)
                output = "error."
                break
            time.sleep(retry_delay)
            attempt += 1
    return output, question, answer, q_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT Inference on a dataset")

    # models
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")

    # datasets
    parser.add_argument('--annotation_path', type=str, default="./EgoThink/Activity/annotations.json")
    parser.add_argument("--answer_path", type=str, default="./answer/Activity")

    args = parser.parse_args()

    llm_api = LLM_API('gpt-4o')

    dataset = load_dataset(args)

    for i, item in enumerate(dataset):
        item["q_id"] = i + 1
    model_answers = []
    ref_answers = []
    question_files = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_item = {executor.submit(generate_item, args, item): item for item in dataset}

    # 等待每个任务完成并处理结果
    for future in tqdm(as_completed(future_to_item),  total=len(future_to_item), desc=f"Running {args.model_name} on task {args.annotation_path}"):
        item = future_to_item[future]
        try:
            output, question, answer, q_id  = future.result()
            print(question)
        except Exception as e:
            print(f"处理项目 {item} 时发生错误: {e}")

        model_answers.append({
            "question_id" : q_id,
            "model_id" : args.model_name,
            "choices" : [{"index" : 0, "turns" : [output]}]
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
    with open(os.path.join(model_answer_folder, f"{args.model_name}.jsonl"), 'w') as f:
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