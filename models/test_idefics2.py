import requests
from PIL import Image
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
import torch
import cv2

class TestIdefics2:
    def __init__(self, device=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Idefics2Processor.from_pretrained("/disks/disk1/share/models/idefics2-8b")
        self.model = Idefics2ForConditionalGeneration.from_pretrained("/disks/disk1/share/models/idefics2-8b")
        self.model.to(device)

    def generate_img(self, image, question, max_new_tokens=256, planning=False):
        img = Image.open(image)
        images = [img]
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ],
        }]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=images, text=text, return_tensors="pt").to(self.device)
        generated_text = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.processor.batch_decode(generated_text, skip_special_tokens=True)[0]
        return generated_text
    
    def generate(self, video, question, max_new_tokens=256, planning=False, frame_num=8):
        cv2.VideoCapture(video)
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        imgs = []
        for i in range(frame_num):
            cap.set(cv2.CAP_PROP_POS_FRAMES, (i + 1) * frame_count // (frame_num + 1))
            ret, frame = cap.read()
            imgs.append(frame)
        cap.release()
        images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in imgs]
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }]
        for i in range(len(images)):
            messages[i]["content"].append({"type": "image"})
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=images, text=text, return_tensors="pt").to(self.device)
        generated_text = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_text = self.processor.batch_decode(generated_text, skip_special_tokens=True)[0]
        return generated_text

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestIdefics2(device)
    image = "data/Activity/images/1.jpg"
    question = "describe the image"
    response = model.generate_img(image, question)
    print(response)

