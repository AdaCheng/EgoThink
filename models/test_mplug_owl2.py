import torch
from PIL import Image
from transformers import TextStreamer
import os

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def get_image_path_from_video(video_path):
    lis = video_path.split("/")
    # image_folder = '/'.join(lis[:-2] + [lis[-1] + '_img', lis[-1].split('.')[0]])
    image_folder = video_path.replace("/video/", "/images/").replace(".mp4", "/")
    images = os.listdir(image_folder)
    return os.path.join(image_folder, images[-1])


class TestMplugOwl2:
    def __init__(self, device):
        self.device = device
        model_path = "/disks/disk1/share/models/mplug-owl2-llama2-7b"
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, self.model_name, load_8bit=False, load_4bit=False, device=device)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256, planning=False, tg=False, rm_critique=False, rm_feedback=False, hp_h2m=False):
        if not image.endswith("jpg"):
            image = get_image_path_from_video(image)

        if rm_critique:
            question = "Imagine you are the camera wearer (I) who recorded the video. Please directly answer yes or no to determin whether the task is completed or not. Question: {} Short answer:".format(question)
        elif rm_feedback:
            question = "Imagine you are the camera wearer (I) who recorded the video. The video contains an uncompleted task. Please identify the essential completion signals in my observations that indicate the task is not completed by me. Please directly generate the rationale as short as possible. \nQuestion: {} \nShort Answer:".format(question)
        elif hp_h2m:
            question = "Imagine you are the camera wearer (I) who recorded the video. Given the high-level goal (e.g., 'making dumpling') and the current progress video, you need to predict the next mid-level step (e.g., fold dumplings on a cutting board) to achieve the goal. Please directly generate the next one step as short as possible. Question: {} Short answer:".format(question)
        else:
            question = "Imagine you are the camera wearer (I) who recorded the video. Please directly answer the question as short as possible. Question: {} Short answer:".format(question)
        conv = conv_templates["mplug_owl2"].copy()
        roles = conv.roles

        image = Image.open(image).convert('RGB')
        max_edge = max(image.size)
        image = image.resize((max_edge, max_edge))

        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        

        inp = DEFAULT_IMAGE_TOKEN + question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        temperature = 0.7
        max_new_tokens = 512

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        return outputs
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestMplugOwl2(device)
    image = "data/vqa/video/2.mp4"
    question = "Describe the image?"
    print(model.generate(image, question))