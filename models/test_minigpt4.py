import torch
import os
import re
import json
import argparse
from argparse import Namespace
from collections import defaultdict
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub
from minigpt4.common.registry import registry
from transformers import StoppingCriteriaList
from . import get_image

CFG_PATH = 'Your path to minigpt4_eval.yaml'


def prompt_template(question):
    return f"Question: {question}\nShort answer:"

def postprocess(output: str):
    return output.strip().split('\n')[0].strip()


class TestMiniGPT4:
    def __init__(self, device=0):
        conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

        print('Initializing Chat')
        cfg = Config(CFG_PATH)

        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)

        self.CONV_VISION = conv_dict[model_config.model_type]

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        stop_words_ids = [[835], [2277, 29937]]
        stop_words_ids = [torch.tensor(ids).to(device=device) for ids in stop_words_ids]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        self.chat = Chat(model, vis_processor, device=device, stopping_criteria=stopping_criteria)

    
    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=30, **kwargs):
        chat_state = self.CONV_VISION.copy()
        img_list = []
        if image is not None:
            image = get_image(image)
            self.chat.upload_img(image, chat_state, img_list)
        self.chat.encode_img(img_list)
        self.chat.ask(prompt_template(question), chat_state)
        llm_message = self.chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=max_new_tokens)[0]
        return llm_message
        # return postprocess(llm_message)
    
    @torch.no_grad()
    def pure_generate(self, image, question, max_new_tokens=30):
        chat_state = self.CONV_VISION.copy()
        img_list = []
        if image is not None:
            image = get_image(image)
            self.chat.upload_img(image, chat_state, img_list)
        llm_message = self.chat.direct_answer(question, img_list=img_list, max_new_tokens=max_new_tokens)[0]

        return postprocess(llm_message)

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=30):
        image_list = [get_image(image) for image in image_list]
        chat_list = [self.CONV_VISION.copy() for _ in range(len(image_list))]
        question_list = [prompt_template(question) for question in question_list]
        batch_outputs = self.chat.batch_answer(image_list, question_list, chat_list, max_new_tokens=max_new_tokens)
        return [postprocess(o) for o in batch_outputs]