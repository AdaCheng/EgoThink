from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils_new import process_vision_info
from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

class TestQwen2VL:
    def __init__(self, device):
        self.device = device
        model_path = "/disks/disk1/share/models/Qwen2-VL-7B-Instruct"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map= torch.device("cuda" if torch.cuda.is_available() else "cpu")# device,#"auto",
        )
        # default: Load the model on the available device(s) : no flash-attn 2 version 
        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        # self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        # )

        # default processer
        # self.processor = AutoProcessor.from_pretrained(model_path)
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    @torch.no_grad()
    def generate(self, video_path, question, max_new_tokens=256, planning=False, tg=False, rm_critique=False, rm_feedback=False, hp_h2m=False):
        if tg == True:
            question = question + " Please provide the starting second and ending second for that interval."
        elif rm_critique:
            question = "Imagine you are the camera wearer (I) who recorded the video. Please directly answer yes or no to determin whether the task is completed or not. Question: {} Short answer:".format(question)
        elif rm_feedback:
            question = "Imagine you are the camera wearer (I) who recorded the video. The video contains an uncompleted task. Please identify the essential completion signals in my observations that indicate the task is not completed by me. Please directly generate the rationale as short as possible. \nQuestion: {} \nShort Answer:".format(question)
        elif hp_h2m:
            question = "Imagine you are the camera wearer (I) who recorded the video. Given the high-level goal (e.g., 'making dumpling') and the current progress video, you need to predict the next mid-level step (e.g., fold dumplings on a cutting board) to achieve the goal. Please directly generate the next one step as short as possible. Question: {} Short answer:".format(question)
        else:
            question = "Imagine you are the camera wearer (I) who recorded the video. Please directly answer the question as short as possible. Question: {} Short answer:".format(question)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = TestQwen2VL(device)
    video = "/disks/disk1/share/EgoThink/data/czx_avc_0_15s.mp4"
    question = "Describe the content of the video."
    print(model.generate(video, question, max_new_tokens=128))

    # # Messages containing a video and a text query
    # model_path = "/disks/disk1/share/models/Qwen2-VL-7B-Instruct"
    # video_path = "/disks/disk1/share/EgoThink/data/czx_avc_0_15s.mp4"
    # question = "Describe the content of the video."

    # # default: Load the model on the available device(s)
    # # model = Qwen2VLForConditionalGeneration.from_pretrained(
    # #     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    # # )

    # # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )

    # # default processer
    # processor = AutoProcessor.from_pretrained(model_path)


    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestQwen2VL(device)
    video = "data/tg/images/12.mp4"
    # question = "Please give me the time period when I can take the dough out."
    question = "In which video clip does the taking the dough out step occur? Please provide the starting and ending times for that step."
    print(model.generate(video, question))

