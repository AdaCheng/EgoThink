import torch
import sys
sys.path.append('/disks/disk1/share/models/mPLUG-Owl3-7B-240728')

from modeling_mplugowl3 import mPLUGOwl3Model
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from decord import VideoReader, cpu

class MPLUG_owl3:
    def __init__(self, model_path, max_num_frames=16, device='cuda', torch_dtype=torch.half):
        self.model_path = model_path
        self.MAX_NUM_FRAMES = max_num_frames
        self.device = device

        # Load model, tokenizer, and processor
        self.model = mPLUGOwl3Model.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch_dtype)
        self.model.eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = self.model.init_processor(self.tokenizer)

    def encode_video(self, video_path):
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > self.MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, self.MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        print('num frames:', len(frames))
        return frames

    def generate_description(self, video_paths, messages, max_new_tokens=100):
        video_frames = [self.encode_video(video_path) for video_path in video_paths]
        inputs = self.processor(messages, images=None, videos=video_frames)
        inputs = inputs.to(self.device)
        inputs.update({
            'tokenizer': self.tokenizer,
            'max_new_tokens': max_new_tokens,
            'decode_text': True,
        })

        return self.model.generate(**inputs)
if __name__ == "__main__":
    # 使用方式
    model_path = '/disks/disk1/share/models/mPLUG-Owl3-7B-240728'
    videos = ['/disks/disk5/private/yyyou/InternLM-XComposer-main/examples/liuxiang.mp4']
    messages = [
        {"role": "user", "content": """<|video|>
    Describe this video."""},
        {"role": "assistant", "content": ""}
    ]

    model = MPLUG_owl3(model_path)
    description = model.generate_description(videos, messages)
    print(description)
