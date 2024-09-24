import json
import os
from tqdm import tqdm

save_dir="/disks/disk1/share/EgoThink/videos/h2m/sample_save/"
h2m_path = "/disks/disk1/share/EgoThink/videos/h2m/inference_pipeline/formal_goalstep_hp_h2m.json"
m2l_path = "/disks/disk1/share/EgoThink/videos/h2m/inference_pipeline/formal_goalstep_hp_m2l(1).json"
h2m = []
m2l = []

with open(h2m_path, "r") as f:
    h2m = json.load(f)
with open(m2l_path, "r") as f:
    m2l = json.load(f)

def get_caption_annotation_path(data, save_dir):
    vid_path = data['file']
    vid_dir = save_dir
    vid_name = os.path.splitext(os.path.basename(data['file']))[0]
    folder_name = vid_name + "-" + str(round(data['start_time'])) + "-" + str(round(data['end_time']))
    folder_dir = os.path.join(vid_dir, folder_name)
    frame_dir =  os.path.join(folder_dir, "frames")
    caption_path = os.path.join(folder_dir, "caption.json")
    
    if not os.path.exists(caption_path):
        raise FileNotFoundError(f"The path {caption_path} cannot be charted after a long search")
    elif not os.path.exists(frame_dir):
        raise FileNotFoundError(f"The path {frame_dir} seems to be disappered")
    elif len(os.listdir(frame_dir)) == 0:
        raise FileNotFoundError(f"The directory {frame_dir} has an overwhelming absence of frames")
    
    return [os.path.join(frame_dir, frame) for frame in os.listdir(frame_dir)], caption_path

def get_egothink_data_h2m(dataset):
    out = []
    for data in tqdm(dataset):
        out.append({
            "video_uid": data['video_uid'],
            "start_time": 0,
            "end_time": data['start_time'],
            "video_path": data['file'],
            "image_path": [],
            "question": f"I want to perform the following task: {data['goal_description']} What is the next sub-step I should take in order to complete this task? ",
            "answer": data['step_description']
        })
    return out

def get_egothink_data_m2l(dataset):
    out = []
    for data in tqdm(dataset):
        out.append({
            "video_uid": data['video_uid'],
            "start_time": 0,
            "end_time": data['start_time'],
            "video_path": data['file'],
            "image_path": [],
            "question": f"I want to perform the following task: {data['step_description']}. What are the actions I should take in order to complete this task? ",
            "answer": data['clip_narration_1'],
            "_answer": data['clip_narration_2']
        })
    return out

with open(os.path.join(save_dir, "h2m.json"), "w") as f:
    json.dump(get_egothink_data_h2m(h2m), f, indent=4)

with open(os.path.join(save_dir, "m2l.json"), "w") as f:
    json.dump(get_egothink_data_m2l(m2l), f, indent=4)