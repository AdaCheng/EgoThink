import argparse
import torch
from lego.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_SOUND_TOKEN
from lego.conversation import SeparatorStyle
from lego import conversation as conversation_lib
from lego.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, load_image_square, postprocess_output
from lego.model.builder import CONFIG, load_pretrained_model
from video_llama.processors.video_processor import load_video
from video_llama.models.ImageBind.data import load_and_transform_audio_data
from lego.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN, \
                           DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN, DEFAULT_SOUND_PATCH_TOKEN, DEFAULT_SOUND_START_TOKEN, DEFAULT_SOUND_END_TOKEN

class TestGroundingGPT:
    def __init__(self, device=None):
        self.device = device
        self.model, self.tokenizer, self.image_processor, self.video_transform, self.context_len = load_pretrained_model("/disks/disk1/share/models/GroundingGPT")
    
    def generate(self, video_path, question, max_new_tokens, planning=False, tg=False, rm_critique=False, rm_feedback=False, hp_h2m=False):
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

        video = load_video(
                video_path = video_path,
                n_frms = self.model.config.max_frame,
                height = 224,
                width = 224,
                sampling ="uniform", return_msg = False)
        video_tensor = self.video_transform(video).unsqueeze(0).to(CONFIG.device, dtype=torch.bfloat16)
        conv = conversation_lib.default_conversation.copy()
        roles = conv.roles
        if self.model.config.mm_use_im_start_end:
            question = DEFAULT_VIDEO_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * CONFIG.video_token_len + DEFAULT_VIDEO_END_TOKEN + '\n' + question
        else:
            question = DEFAULT_VIDEO_TOKEN + '\n' + question
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=None,
                videos=video_tensor,
                sounds=None,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=max_new_tokens, 
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        return outputs

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestGroundingGPT(device)
    video_path = "videos/vqa/2.mp4"
    question = "describe the video"
    response = model.generate(video_path, question, 256)
    print(response)