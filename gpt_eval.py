import os
import json
import argparse
import datetime
from tqdm import tqdm

from openai import AzureOpenAI
import openai
import requests
import base64
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT Inference on a dataset")

    # models
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")

    # datasets
    parser.add_argument('--annotation_path', type=str, default="./EgoThink/Activity/annotations.json")
    parser.add_argument("--answer_path", type=str, default="./answer/rm_critique")

    args = parser.parse_args()

    llm_api = LLM_API('gpt-4o')

    dataset = load_dataset(args)
    model_answers = []
    ref_answers = []
    question_files = []
    q_id = 0
    for item in tqdm(dataset, desc=f"Running {args.model_name} on benchmark {args.annotation_path}"):
        q_id += 1
        question = item["question"]
        images = item["images"]
        answer = item["answer"]

        prompt = "Please let your answer be as short as possible. Question: {question} Short answer:".format(question=question)

        try:
            output = llm_api.request_vision(images, prompt)
        except Exception as e:
            print(e)
            output = "error."
        if "Short Answer: " in output:
            output = output.split("Short Answer: ")[1]
        print(output)
        
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