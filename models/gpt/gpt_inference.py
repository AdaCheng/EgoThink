import json
import re
from openai import OpenAI
from video_caption import Video, video_inference, multi_image_inference
from video_caption import PROMPT as VID_CAP_PROMPT
from tqdm import tqdm
import os

GET_DOC  = '''\n\nGiven the dictionary above that describes a function that performs a low-level action, provide formal 
             documentation for the purpose of the function and what each argument is for. The argument in the above 
             dictionary is a sample argument, you may use it to reason what each argument is meant for in the function;
             however, you must provide your documentation devoid of referring to the specific argument provided in the 
             example above. Provide the documentation in string. you may output the function as follows 
             function_name(<arg1>, <arg2>, ...) etc and use this to refer to part of the function when making 
             descriptions. Output your response in a single concise paragraph. Do not provide any line breaks.'''

KEY = {
    "api_key": "API_KEY",
    "api_base": "API_BASE" 
}
CLIENT = OpenAI(api_key=KEY['api_key'], base_url=KEY['api_base'])

def gpt(prompt):
    try:
        output = CLIENT.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt}
            ]
        )
        return output.choices[0].message.content
    except:
        return "error occurred"

def split_list(lst, chunk_size=10):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def prepare_vid_data(data, key):
    if not os.path.exists(data['file']):
        raise FileNotFoundError(f"The path {data['file']} is invalid.")

    # Get Relevent  Directory
    vid_path = data['file']
    vid_dir = os.path.dirname(data['file']).rpartition('videos')[0] + 'videos'
    vid_name = os.path.splitext(os.path.basename(data['file']))[0]
    folder_name = vid_name + str(round(data['start_time'])) + "-" + str(round(data['end_time']))
    folder_dir = os.path.join(vid_dir, folder_name)
    frame_dir =  os.path.join(folder_dir, "frames")
    caption_path = os.path.join(folder_dir, "caption.json")
    
    if not os.path.exists(folder_dir):
        os.mkdir(folder_dir)
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)

        # Generate frames at 2 frames per second
        vid = Video(vid_path)
        vid.save_frames(data['start_time'],
                        data['end_time'],
                        round((data['end_time']  - data['start_time']) / 1),
                        frame_dir)
        vid.del_video()
        del vid
    
    # Generate LLM Caption
    split_frames = split_list(os.listdir(frame_dir))
    for i in range(len(split_frames)):
        for j in range(len(split_frames[i])):
            split_frames[i][j] = os.path.join(frame_dir, split_frames[i][j])

    caption = video_inference(VID_CAP_PROMPT, split_frames, key)
    with open(caption_path, "w") as f:
        json.dump(caption, f, indent=4)

def main():
    h2m_path = "H2M"
    m2l_path = "M2L"
    h2m = []
    m2l = []
    with open(h2m_path, "r") as f:
        h2m = json.load(f)
    with open(m2l_path, "r") as f:
        m2l = json.load(f)

    prepare_data(h2m[0], KEY)

if __name__ == "__main__":
    main()