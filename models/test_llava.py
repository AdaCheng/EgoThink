import torch
from transformers import CLIPImageProcessor, CLIPVisionModel, StoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import warnings, os, shutil
# from llava import LlavaMPTForCausalLM, LlavaLlamaForCausalLM, conv_templates, SeparatorStyle
from llava.model import LlavaLlamaForCausalLM, LlavaMPTForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from . import get_image
import pdb


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


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


def get_conv(model_name):
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
    print("Using conv mode: ", conv_mode)
    conv = conv_templates[conv_mode].copy()
    return conv


# def prompt_question(question):
#     return f"Please answer the following question and keep your answer short and concise: {question}"


def load_model(model_path, model_name, dtype=torch.float16, device='cpu'):
    # get tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if 'llava' in model_name.lower():
        if 'mpt' in model_name.lower():
            model = LlavaMPTForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    elif 'mpt' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)

    # get image processor
    image_processor = None
    if 'llava' in model_name.lower():
        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=dtype)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = model.get_model().vision_tower[0]
        if vision_tower.device.type == 'meta':
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True).to(device=device)
            model.get_model().vision_tower[0] = vision_tower
        else:
            vision_tower.to(device=device, dtype=dtype)
        
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    model.to(device=device)

    return tokenizer, model, image_processor, context_len

# def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto"):
#     kwargs = {"device_map": device_map}

#     if load_8bit:
#         kwargs['load_in_8bit'] = True
#     elif load_4bit:
#         kwargs['load_in_4bit'] = True
#         kwargs['quantization_config'] = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type='nf4'
#         )
#     else:
#         kwargs['torch_dtype'] = torch.float16

#     if 'llava' in model_name.lower():
#         # Load LLaVA model
#         if 'lora' in model_name.lower() and model_base is None:
#             warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
#         if 'lora' in model_name.lower() and model_base is not None:
#             lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
#             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#             print('Loading LLaVA from base model...')
#             model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
#             token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
#             if model.lm_head.weight.shape[0] != token_num:
#                 model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
#                 model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

#             print('Loading additional LLaVA weights...')
#             if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
#                 non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
#             else:
#                 # this is probably from HF Hub
#                 from huggingface_hub import hf_hub_download
#                 def load_from_hf(repo_id, filename, subfolder=None):
#                     cache_file = hf_hub_download(
#                         repo_id=repo_id,
#                         filename=filename,
#                         subfolder=subfolder)
#                     return torch.load(cache_file, map_location='cpu')
#                 non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
#             non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
#             if any(k.startswith('model.model.') for k in non_lora_trainables):
#                 non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
#             model.load_state_dict(non_lora_trainables, strict=False)

#             from peft import PeftModel
#             print('Loading LoRA weights...')
#             model = PeftModel.from_pretrained(model, model_path)
#             print('Merging LoRA weights...')
#             model = model.merge_and_unload()
#             print('Model is loaded...')
#         elif model_base is not None:
#             # this may be mm projector only
#             print('Loading LLaVA from base model...')
#             if 'mpt' in model_name.lower():
#                 if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
#                     shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
#                 tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
#                 cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
#                 model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
#             else:
#                 tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#                 cfg_pretrained = AutoConfig.from_pretrained(model_path)
#                 model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

#             mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
#             mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
#             model.load_state_dict(mm_projector_weights, strict=False)
#         else:
#             if 'mpt' in model_name.lower():
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#                 model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
#             else:
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#                 model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
#     else:
#         # Load language model
#         if model_base is not None:
#             # PEFT model
#             from peft import PeftModel
#             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#             model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
#             print(f"Loading LoRA weights from {model_path}")
#             model = PeftModel.from_pretrained(model, model_path)
#             print(f"Merging weights")
#             model = model.merge_and_unload()
#             print('Convert to FP16...')
#             model.to(torch.float16)
#         else:
#             use_fast = False
#             if 'mpt' in model_name.lower():
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#                 model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
#             else:
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#                 model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

#     image_processor = None

#     if 'llava' in model_name.lower():
#         mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
#         mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
#         if mm_use_im_patch_token:
#             tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
#         if mm_use_im_start_end:
#             tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
#         model.resize_token_embeddings(len(tokenizer))

#         vision_tower = model.get_vision_tower()
#         if not vision_tower.is_loaded:
#             vision_tower.load_model()
#         vision_tower.to(device='cuda', dtype=torch.float16)
#         image_processor = vision_tower.image_processor

#     if hasattr(model.config, "max_sequence_length"):
#         context_len = model.config.max_sequence_length
#     else:
#         context_len = 2048

#     return tokenizer, model, image_processor, context_len



class TestLLaVA:
    def __init__(self, size='7b', device=None):
        if size == '7b':
            model_path = ""
            model_name = get_model_name(model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, "", model_name, device_map="auto")
        elif size == '13b':
            model_path = ""
            model_name = get_model_name(model_path)
            # print(model_name, model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, model_name, device_map="auto")
        elif size == '1.5-7b':
            model_path = ""
            model_name = get_model_name(model_path)
            model_base = ""
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, model_name, device_map="auto")

        # model_path="liuhaotian/LLaVA-Lightning-MPT-7B-preview"
        # model_name = get_model_name(model_path)
        # self.tokenizer, self.model, self.image_processor, self.context_len = load_model(model_path, model_name)
        self.model_path = model_path
        self.conv = get_conv(model_name)
        self.image_process_mode = "Resize" # Crop, Resize, Pad
        # print('Device: ', device)


        if device is not None:
            self.move_to_device(device)
        # import pdb; pdb.set_trace()
    
    def refresh_conv(self):
        model_name = get_model_name(self.model_path)
        self.conv = get_conv(model_name)
        
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
    def generate(self, image, question, max_new_tokens=30):
        image = get_image(image)
        conv = self.conv.copy()
        text = question + '\n<image>'
        text = (text, image, self.image_process_mode)
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        # prompt = f"{prompt} Answer: "
        # import pdb; pdb.set_trace()
        # print(prompt)
        output = self.do_generate([prompt], [image], stop_str=stop_str, dtype=self.dtype, max_new_tokens=max_new_tokens)[0]

        return output

    @torch.no_grad()
    def pure_generate(self, image, question, max_new_tokens=30):
        image = get_image(image)
        conv = self.conv.copy()
        prompt = question
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        output = self.do_generate([prompt], [image], stop_str=stop_str, dtype=self.dtype, max_new_tokens=max_new_tokens)[0]

        return output
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=30):
        images, prompts = [], []
        for image, question in zip(image_list, question_list):
            image = get_image(image)
            conv = self.conv.copy()
            text = question + '\n<image>'
            text = (text, image, self.image_process_mode)
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            prompts.append(prompt)
            images.append(image)
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        outputs = self.do_generate(prompts, images, stop_str=stop_str, dtype=self.dtype, max_new_tokens=max_new_tokens)

        return outputs

    @torch.no_grad()
    def do_generate(self, prompts, images, dtype=torch.float16, temperature=0.2, max_new_tokens=30, stop_str=None, keep_aspect_ratio=False):
        if keep_aspect_ratio:
            new_images = []
            for image, prompt in zip(images, prompts):
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = self.image_processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values'][0]
                new_images.append(image.to(self.model.device, dtype=dtype))
                # replace the image token with the image patch token in the prompt (each occurrence)
                cur_token_len = (image.shape[1]//14) * (image.shape[2]//14)
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * cur_token_len
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token, 1)
            images = new_images
        else:
            images = self.image_processor(images, return_tensors='pt')['pixel_values']
            images = images.to(self.model.device, dtype=dtype)
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * 256    # HACK: 256 is the max image token length hacked
            if getattr(self.model.config, 'mm_use_im_start_end', False):
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            prompts = [prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token) for prompt in prompts]

        stop_idx = None
        if stop_str is not None:
            stop_idx = self.tokenizer(stop_str).input_ids
            if len(stop_idx) == 1:
                stop_idx = stop_idx[0]
            else:
                stop_idx = None

        input_ids = self.tokenizer(prompts).input_ids
        batch_size = len(input_ids)
        min_prompt_size = min([len(input_id) for input_id in input_ids])
        max_prompt_size = max([len(input_id) for input_id in input_ids])
        for i in range(len(input_ids)):
            padding_size = max_prompt_size - len(input_ids[i])
            # input_ids[i].extend([self.tokenizer.pad_token_id] * padding_size)
            input_ids[i] = [self.tokenizer.pad_token_id] * padding_size + input_ids[i]
        
        output_ids = []
        get_result = [False for _ in range(batch_size)]
        for i in range(max_new_tokens):
            if i == 0:
                # pdb.set_trace()
                out = self.model(
                    torch.as_tensor(input_ids).to(self.model.device),
                    use_cache=True,
                    images=images)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                # pdb.set_trace()
                out = self.model(input_ids=token,
                            use_cache=True,
                            attention_mask=torch.ones(batch_size, past_key_values[0][0].shape[-2] + 1, device=self.model.device),
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[:, -1]
            if temperature < 1e-4:
                token = torch.argmax(last_token_logits, dim=-1)
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            token = token.long().to(self.model.device)

            output_ids.append(token)
            for idx in range(len(token)):
                if token[idx] == stop_idx or token[idx] == self.tokenizer.eos_token_id:
                    get_result[idx] = True
            if all(get_result):
                break
        
        output_ids = torch.cat(output_ids, dim=1).long()
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if stop_str is not None:
            for i in range(len(outputs)):
                pos = outputs[i].rfind(stop_str)
                if pos != -1:
                    outputs[i] = outputs[i][:pos]
        
        return outputs