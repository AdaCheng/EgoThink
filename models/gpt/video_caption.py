import cv2
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from openai import OpenAI
import os
import base64
import requests

PROMPT = ''''
Given a sequence of image which are selected frames from a clip within a video. You are tasked to provide
a comprehensive list of narrations describing what is happening in that clip. You are output should consists 
of a list of sentencce describing what the segments of the clips captures and can tell you about the video. For example
if several segments describe someone opening a jar and pouring the content it into a bowl. Your output should be 
of the form: "person A opens the jar.\nPerson A places the lid and the open jar on the side of the table.\nPerson A grabs 
a bowl from the cabinet behind him.\nPerson A places the bowl onto the counter.\nPerson A pour the content of the jar into 
the bowl". Each line of narration should be a independent clause that highlight the subject that performs an action and the 
action performed by the subject. Note if they are multiple suject, you may use person A or B. When you see an action is performed
with something but are unsure what the thing is, you may refer to it as #unsure. When outputting your replies, you do not 
need to output any description to what you are going to output, your chain of thought, or any report of anomalies; plain text out is fine.
'''

class Video:
    def __init__(self, path):
        try:
            self.path = path
            self.video = cv2.VideoCapture(self.path)
            self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
            self.total_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_time = self.total_frame / self.fps
        except:
            print("An error have occured")
    
    def get_frame(self, timestamp, save=False, path=None):
        self.video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = self.video.read()

        if ret:
            # Display the frame
            if save == False:
                return frame
            else:
                try:
                    cv2.imwrite(os.path.join(path, f"{timestamp}.jpeg"), frame)
                except:
                    print("Failed to save")
        else:
            print("Failed to retrieve frame at the specified time.")

    def show_frame(self, timestamp):
        self.video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = self.video.read()
        if ret:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()
        else:
            print("Failed to retrieve frame at the specified time.")

    def save_frames(self, start_t, end_t, k, path):
        if end_t <= start_t or end_t > self.total_time:
            raise ValueError()
            return
        
        increment = round((end_t - start_t) / k)
        timestamps = [round(start_t + (i * increment), 2) for i in range(k)]

        for ts in tqdm(timestamps, desc=f"Saving {k} frames to {path}", leave=False):
            self.get_frame(ts, save=True, path=path)
    
    def del_video(self):
        self.video.release()

def video_to_clip(annotation, video_path, save_path):
    video = Video(video_path)
    for i in tqdm(range(len(annotation)), "Processing Clips"):
        os.mkdir(os.path.join(save_path, f"_{i}"))
        video.save_frames(annotation[i]['start_time'], annotation[i]['end_time'], 
                          annotation[i]['end_time']-annotation[i]['start_time'], 
                          save_path + f"//_{i}")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def multi_image_inference(prompt, image_list, key):
    headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {key['api_key']}"
    }
    
    content = [{
        "type": "text",
        "text": prompt
    }]

    for path in tqdm(image_list, desc="Encoding Frames", leave=False):
        content.append({
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{encode_image(path)}"
            }
        })

    payload = {
        "model": "gpt-4o",
        "messages": [ { 
            "role": "user", 
            "content": content
        }],
        "max_tokens": 1105
    }

    # try:
    response = requests.post(f"{key['api_base']}/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']
    # except:
    #     return "An error has occured"

def video_inference(prompt, clip_folder_path, key):
    return [multi_image_inference(prompt, video_segment_list, key) for video_segment_list in tqdm(clip_folder_path)] 

