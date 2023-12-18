import torch
from transformers import CLIPImageProcessor, CLIPVisionModel, StoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import warnings, os, shutil
# from llava import LlavaMPTForCausalLM, LlavaLlamaForCausalLM, conv_templates, SeparatorStyle
from llava.model import LlavaLlamaForCausalLM, LlavaMPTForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from PIL import Image
import pdb, re
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
    # if "llava" in model_name.lower():
    #     if "v1" in model_name.lower():
    #         template_name = "llava_v1"
    #     elif "mpt" in model_name.lower():
    #         template_name = "mpt_multimodal"
    #     else:
    #         template_name = "multimodal"
    # elif "mpt" in model_name:
    #     template_name = "mpt_text"
    # elif "koala" in model_name: # Hardcode the condition
    #     template_name = "bair_v1"
    # elif "v1" in model_name:    # vicuna v1_1/v1_2
    #     template_name = "vicuna_v1_1"
    # else:
    #     template_name = "v1"
    # # return conv_templates[template_name].copy()
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1_2"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    # conv_mode = 'plain'
    if planning:
        conv_mode += "_planning"
    # print("Using conv mode: ", conv_mode)
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
            model_path = ""
            model_name = get_model_name(model_path)
            model_base = ""
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, model_name, device_map="auto")
        elif size == '1.5-13b':
            model_path = ""
            model_name = get_model_name(model_path)
            model_base = ""
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
    def generate(self, image, question, max_new_tokens=30, planning=False):
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

        image = get_image(image)
        images_tensor = process_images(
            [image],
            self.image_processor,
            self.model.config
        ).to(dtype=torch.float16).cuda(0)
        # print(prompt)
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda(0)
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True,
                temperature=0.001,
                top_p=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        if not planning:
            outputs = outputs.split("\n")[-1]
        return outputs
    
if __name__ == "__main__":
    model = TestLLaVA2()
    image = "data/Activity/images/3_153.jpg"
    qs = "What am I doing?"
    output = model.generate(image, qs)
    print(output)
