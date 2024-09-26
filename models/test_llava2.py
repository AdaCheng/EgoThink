import argparse
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel, StoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import warnings, os, shutil
# from llava import LlavaMPTForCausalLM, LlavaLlamaForCausalLM, conv_templates, SeparatorStyle
from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from PIL import Image
import pdb, re
import requests
from PIL import Image
from io import BytesIO
import re
import cv2
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

def image_parser(image_file):
    out = image_file.split(',')
    return out

def get_image_path_from_video(video_path):
    lis = video_path.split("/")
    # image_folder = '/'.join(lis[:-2] + [lis[-1] + '_img', lis[-1].split('.')[0]])
    image_folder = video_path.replace("/video/", "/images/").replace(".mp4", "/")
    print(image_folder)
    images = os.listdir(image_folder)
    return os.path.join(image_folder, images[-1])

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def get_model_name(model_path):
    # get model name
    if model_path.endswith("/"):
        model_path = model_path[:-1]
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        model_name = model_paths[-2] + "_" + model_paths[-1]
    else:
        model_name = model_paths[-1]
    
    return model_name


def get_conv(model_name, planning=False):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    if planning:
        conv_mode += "_planning"
    conv = conv_templates[conv_mode].copy()
    return conv

def get_image(image):
    if type(image) is str:
        try:
            return Image.open(image).convert("RGB")
        except Exception as e:
            print(f"Fail to read image: {image}")
            exit(-1)
    elif type(image) is Image.Image:
        return image
    else:
        raise NotImplementedError(f"Invalid type of Image: {type(image)}")

class TestLLaVA2:
    def __init__(self, size='7b', device=None):
        if size == '7b':
            model_path = ""
            model_name = get_model_name(model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, "", model_name, device_map="auto")
        elif size == '7b-sharegpt4v':
            model_path = ""
            model_name = "llava-v1.5-7b"
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name, device_map="auto")
        elif size == '13b':
            model_path = ""
            model_name = get_model_name(model_path)
            # print(model_name, model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name, device_map="auto")
        elif size == '13b-llama2':
            model_path = ""
            model_name = get_model_name(model_path)
            # print(model_name, model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name, device_map="auto")
        elif size == '1.5-7b':
            model_path = "/disks/disk1/share/models/llava-v1.5-7b"
            model_name = get_model_name(model_path)
            model_base = None
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, model_name, device_map="auto")
        elif size == '1.5-13b':
            model_path = "/disks/disk1/share/models/llava-v1.5-13b"
            model_name = get_model_name(model_path)
            model_base = None
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, model_name, device_map="auto")
        self.model_name = model_name
        # model_path="liuhaotian/LLaVA-Lightning-MPT-7B-preview"
        # model_name = get_model_name(model_path)
        # self.tokenizer, self.model, self.image_processor, self.context_len = load_model(model_path, model_name)
        self.model_path = model_path
        self.image_process_mode = "Resize" # Crop, Resize, Pad
        # print('Device: ', device)


        # if device is not None:
        #     self.move_to_device(device)
        # import pdb; pdb.set_trace()
    
    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = self.model.device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        # vision_tower = self.model.get_model().vision_tower[0]
        # vision_tower.to(device=self.device, dtype=self.dtype)
        # self.model.to(device=self.device, dtype=self.dtype)
    
    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=30, planning=False, tg=False, rm_critique=False, rm_feedback=False, hp_h2m=False):
        if rm_critique:
            question = "Imagine you are the camera wearer (I) who recorded the video. Please directly answer yes or no to determin whether the task is completed or not. Question: {} Short answer:".format(question)
        elif rm_feedback:
            question = "Imagine you are the camera wearer (I) who recorded the video. The video contains an uncompleted task. Please identify the essential completion signals in my observations that indicate the task is not completed by me. Please directly generate the rationale as short as possible. \nQuestion: {} \nShort Answer:".format(question)
        elif hp_h2m:
            question = "Imagine you are the camera wearer (I) who recorded the video. Given the high-level goal (e.g., 'making dumpling') and the current progress video, you need to predict the next mid-level step (e.g., fold dumplings on a cutting board) to achieve the goal. Please directly generate the next one step as short as possible. Question: {} Short answer:".format(question)
        else:
            question = "Imagine you are the camera wearer (I) who recorded the video. Please directly answer the question as short as possible. Question: {} Short answer:".format(question)

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        qs = question
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = get_conv(self.model_name, planning=planning).copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if not image.endswith("jpg"):
            image = get_image_path_from_video(image)

        # if image[0].endswith("jpg"):
        image_files = image_parser(image)
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)
        # else:
        #     cap = cv2.VideoCapture(image)
        #     fps = cap.get(cv2.CAP_PROP_FPS)
        #     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        #     ret, frame = cap.read()
        #     cap.release()
        #     images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))]
        #     image_sizes = [images[0].size]
        #     images_tensor = process_images(
        #         images,
        #         self.image_processor,
        #         self.model.config
        #     ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        temperature = 0.2

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs


    

if __name__ == "__main__":
    model = TestLLaVA2("1.5-7b")
    image = "data/vqa/video/34.mp4"
    qs = "What am I doing?"
    output = model.generate(image, qs)
    print(output)
