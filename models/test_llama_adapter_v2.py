import torch
from . import get_BGR_image, DATA_DIR
from . import llama_adapter_v2 as llama
import cv2
from PIL import Image
import os

llama_dir = "/disks/disk1/share/models/llama_adapter"
model_path = "BIAS-7B"
 # llama_adapter_v2_BIAS-7B.pth, llama_adapter_v2_LORA-BIAS-7B.pth

def get_image_path_from_video(video_path):
    lis = video_path.split("/")
    # image_folder = '/'.join(lis[:-2] + [lis[-1] + '_img', lis[-1].split('.')[0]])
    image_folder = video_path.replace("/video/", "/images/").replace(".mp4", "/")
    print(image_folder)
    images = os.listdir(image_folder)
    return os.path.join(image_folder, images[-1])

class TestLLamaAdapterV2:
    def __init__(self, device=None) -> None:
        # choose from BIAS-7B, LORA-BIAS-7B
        model, preprocess = llama.load(model_path, llama_dir, device, max_seq_len=256, max_batch_size=16)
        model.eval()
        self.img_transform = preprocess
        self.model = model
        self.device = device

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256, planning=False, tg=False, rm_critique=False, rm_feedback=False, hp_h2m=False):
        if not image.endswith("jpg"):
            image = get_image_path_from_video(image)
        # if image.endswith("jpg"):
        imgs = [get_BGR_image(image)]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        # else:
            # cap = cv2.VideoCapture(image)
            # fps = cap.get(cv2.CAP_PROP_FPS)
            # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            # ret, frame = cap.read()
            # cap.release()
            # images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))]
            # imgs = [self.img_transform(x) for x in images]
            # imgs = torch.stack(imgs, dim=0).to(self.device)

        if rm_critique:
            question = "Imagine you are the camera wearer (I) who recorded the video. Please directly answer yes or no to determin whether the task is completed or not. Question: {} Short answer:".format(question)
        elif rm_feedback:
            question = "Imagine you are the camera wearer (I) who recorded the video. The video contains an uncompleted task. Please identify the essential completion signals in my observations that indicate the task is not completed by me. Please directly generate the rationale as short as possible. \nQuestion: {} \nShort Answer:".format(question)
        elif hp_h2m:
            question = "Imagine you are the camera wearer (I) who recorded the video. Given the high-level goal (e.g., 'making dumpling') and the current progress video, you need to predict the next mid-level step (e.g., fold dumplings on a cutting board) to achieve the goal. Please directly generate the next one step as short as possible. Question: {} Short answer:".format(question)
        else:
            question = "Imagine you are the camera wearer (I) who recorded the video. Please directly answer the question as short as possible. Question: {} Short answer:".format(question)

        prompts = [llama.format_prompt(question, planning=planning)]
        results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens)
        result = results[0].strip()

        return result

    @torch.no_grad()
    def pure_generate(self, image, question, max_new_tokens=256):
        imgs = [get_BGR_image(image)]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [question]
        results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens)
        result = results[0].strip()

        return result

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256):
        imgs = [get_BGR_image(img) for img in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [llama.format_prompt(question) for question in question_list]
        results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens)
        results = [result.strip() for result in results]

        return results
