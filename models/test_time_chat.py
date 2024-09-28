import argparse
import os
import random
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchshow as ts
from timechat.common.config import Config
from timechat.common.dist_utils import get_rank
from timechat.common.registry import registry
from timechat.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle, conv_llava_llama_2
import decord
import cv2
import time
import subprocess
from decord import VideoReader
from timechat.processors.video_processor import ToTHWC, ToUint8, load_video
decord.bridge.set_bridge('torch')

# imports modules for registration
from timechat.datasets.builders import *
from timechat.models import *
from timechat.processors import *
from timechat.runners import *
from timechat.tasks import *

import random as rnd
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import gradio as gr

class TestTimeChat:
    def __init__(self, device=None):
        args = argparse.Namespace()
        args.cfg_path = "/disks/disk1/share/EgoThink/models/timechat/timechat.yaml"
        args.options = None
        args.gpu_id = 0
        cfg = Config(args)

        DIR="/disks/disk1/share/models/TimeChat/timechat"
        MODEL_DIR=f"{DIR}/timechat_7b.pth"
        print(cfg.model_cfg)

        self.model_config = cfg.model_cfg
        self.model_config.device_8bit = args.gpu_id
        self.model_config.ckpt = MODEL_DIR
        self.model_cls = registry.get_model_class(self.model_config.arch)
        # model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
        self.model = self.model_cls.from_config(self.model_config).to(device)
        self.model.eval()

        self.vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
        self.vis_processor = registry.get_processor_class(self.vis_processor_cfg.name).from_config(self.vis_processor_cfg)

        # chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        self.chat = Chat(self.model, self.vis_processor, device=device)
    
    def generate(self, video_path, question, max_new_tokens=200, planning=False, tg=False, rm_critique=False, rm_feedback=False, hp_h2m=False):
        # video, _ = load_video(
        #     video_path=video_path,
        #     n_frms=32,
        #     sampling ="uniform", return_msg = True
        # )
        img_list = []
        chat_state = conv_llava_llama_2.copy()
        chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        msg = self.chat.upload_video_without_audio(
            video_path=video_path, 
            conv=chat_state,
            img_list=img_list, 
            n_frms=96,
        )

        # text_input = "You are given a cooking video from the YouCook2 dataset. Please watch the video and extract a maximum of 10 significant cooking steps. For each step, determine the starting and ending times and provide a concise description. The format should be: 'start time - end time, brief step description'. For example, ' 90 - 102 seconds, spread margarine on two slices of white bread'."
        if tg == True:
            question = question + " Please provide the starting and ending times for that step."
        elif rm_critique:
            question = "Imagine you are the camera wearer (I) who recorded the video. Please directly answer yes or no to determin whether the task is completed or not. Question: {} Short answer:".format(question)
        elif rm_feedback:
            question = "Imagine you are the camera wearer (I) who recorded the video. The video contains an uncompleted task. Please identify the essential completion signals in my observations that indicate the task is not completed by me. Please directly generate the rationale as short as possible. \nQuestion: {} \nShort Answer:".format(question)
        elif hp_h2m:
            question = "Imagine you are the camera wearer (I) who recorded the video. Given the high-level goal (e.g., 'making dumpling') and the current progress video, you need to predict the next mid-level step (e.g., fold dumplings on a cutting board) to achieve the goal. Please directly generate the next one step as short as possible. Question: {} Short answer:".format(question)
        else:
            question = "Imagine you are the camera wearer (I) who recorded the video. Please directly answer the question as short as possible. Question: {} Short answer:".format(question)
        
        text_input = question
        print(text_input)

        self.chat.ask(text_input, chat_state)

        num_beams = 1 #args.num_beams
        temperature = 1.0 #args.temperature
        llm_message = self.chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=num_beams,
                                temperature=temperature,
                                max_new_tokens=300,
                                max_length=2000)[0]
        print(llm_message)
        return llm_message

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestTimeChat(device)
    video = "videos/0.mp4"
    # question = "Please give me the time period when I can take the dough out."
    question = "In which video clip does the taking the dough out step occur? Please provide the starting and ending times for that step."
    print(model.generate(video, question))
