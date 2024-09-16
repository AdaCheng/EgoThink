import math
from typing import List, Optional
import json
import torch
import torchvision

from threading import Thread
from copy import deepcopy
from PIL import Image
from transformers import AutoProcessor, Qwen2PreTrainedModel, Qwen2ForCausalLM, TextIteratorStreamer
from processing_mplugowl3 import mPLUGOwl3Processor
from image_processing_mplugowl3 import mPLUGOwl3ImageProcessor
from configuration_mplugowl3 import mPLUGOwl3Config
# from .modeling_navit_siglip import SiglipVisionTransformer
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer
from x_sdpa import ScaleDotProductAttention
from modeling_hyper_qwen2 import HyperQwen2ForCausalLM
from torch import nn


class mPLUGOwl3PreTrainedModel(Qwen2PreTrainedModel):
    config_class = mPLUGOwl3Config


class mPLUGOwl3Model(mPLUGOwl3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = HyperQwen2ForCausalLM(config)
        self.vision_model = self.init_vision_module()
        self.vision_dim = self.vision_model.embed_dim
        self.embed_dim = self.language_model.config.hidden_size
        self.vision2text_model = nn.Linear(self.vision_dim, self.embed_dim)
        self.processor = None

        self.terminators = ['<|im_end|>', '<|endoftext|>']

    def init_vision_module(self):
       
        self.config.vision_config._attn_implementation = self.config.vision_config._attn_implementation
        model = SiglipVisionTransformer(self.config.vision_config)

        setattr(model, 'embed_dim', model.embeddings.embed_dim)
        setattr(model, 'patch_size', model.embeddings.patch_size)
        return model


    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.language_model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.language_model.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def forward_image(self, pixel_values):
        if pixel_values is None:
            return None
        dtype = self.language_model.model.embed_tokens.weight.dtype
        with torch.inference_mode():
            image_embeds = self.vision_model(pixel_values.to(dtype), output_hidden_states=True).hidden_states[-2]
        
        if self.vision2text_model is not None:
            image_embeds = self.vision2text_model(image_embeds)
        else:
            pass
     
        return image_embeds

    def forward(self, pixel_values=None, **kwargs):
        image_embeds = self.forward_image(pixel_values)
        
        return self.language_model(
            image_embeds=image_embeds,
            **kwargs
        )
    
    def _decode(self, input_ids, image_embeds, media_offset, tokenizer, attention_mask, decode_text=False, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        output = self.language_model.generate(
            input_ids=input_ids,
            image_embeds=image_embeds,
            media_offset=media_offset,
            pad_token_id=0,
            eos_token_id=terminators,
            attention_mask=attention_mask,
            **kwargs
        )
        
        output = output[:,input_ids.shape[1]:]
        if decode_text:
            return self._decode_text(output, tokenizer)
        return output

    def _decode_stream(self, input_ids, image_embeds, media_offset, tokenizer, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_kwargs = {
            'input_ids': input_ids,
            'image_embeds': image_embeds,
            'media_offset': media_offset,
            'pad_token_id': 0,
            'eos_token_id': terminators,
            'streamer': streamer
        }
        generation_kwargs.update(kwargs)

        thread = Thread(target=self.language_model.generate, kwargs=generation_kwargs)
        thread.start()
    
        return streamer

    def _decode_text(self, result_ids, tokenizer):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text

    def init_processor(self, tokenizer):
        ip = mPLUGOwl3ImageProcessor(image_size=384)
        self.processor = mPLUGOwl3Processor(image_processor=ip, tokenizer=tokenizer)
        processor = self.processor
        return processor

    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        media_offset=None,
        attention_mask=None,
        tokenizer=None,
        return_vision_hidden_states=False,
        stream=False,
        decode_text=False,
        **kwargs
    ):
        assert input_ids is not None

        with torch.inference_mode():
            image_embeds = self.forward_image(pixel_values)

            if stream:
                result = self._decode_stream(input_ids=input_ids, image_embeds=image_embeds, media_offset=media_offset, tokenizer=tokenizer, **kwargs)
            else:
                result = self._decode(input_ids=input_ids, image_embeds=image_embeds, media_offset=media_offset, tokenizer=tokenizer, attention_mask=attention_mask, decode_text=decode_text, **kwargs)

        if return_vision_hidden_states:
            return result, image_embeds
        
        return result

    def chat(
        self,
        images,
        videos,
        msgs,
        tokenizer,
        processor=None,
        vision_hidden_states=None,
        max_new_tokens=2048,
        min_new_tokens=0,
        sampling=True,
        max_inp_length=8192,
        system_prompt='',
        stream=False,
        max_slice_nums=None,
        use_image_id=None,
        **kwargs
    ):
        print(msgs)
        if processor is None:
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)
            processor = self.processor
        

        inputs = processor(
            prompts_lists, 
            input_images_lists, 
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            return_tensors="pt", 
            max_length=max_inp_length
        ).to(self.device)

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }
            
        if min_new_tokens > 0:
            generation_config['min_new_tokens'] = min_new_tokens

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )

        inputs.pop("image_sizes")
        with torch.inference_mode():
            res = self.generate(
                **inputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                decode_text=True,
                **generation_config
            )
        
        if stream:
            def stream_gen():
                for text in res:
                    for term in self.terminators:
                        text = text.replace(term, '')
                    yield text
            return stream_gen()

        else:
            if batched:
                answer = res
            else:
                answer = res[0]
            return answer

