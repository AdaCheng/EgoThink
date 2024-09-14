import av
import numpy as np
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

class VideoLLaVA:
    def __init__(self, model_path, video_path, question):
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(model_path)
        self.processor = VideoLlavaProcessor.from_pretrained(model_path)
        self.video_path = video_path
        self.container = av.open(video_path)
        self.prompt = question

    def read_video_pyav(self, indices):
        frames = []
        self.container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(self.container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def sample_frames(self, num_frames=8):
        total_frames = self.container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
        return indices

    def generate_description(self, num_frames=8, max_length=80):
        indices = self.sample_frames(num_frames)
        clip = self.read_video_pyav(indices)
        inputs = self.processor(text=self.prompt, videos=clip, return_tensors="pt")
        generate_ids = self.model.generate(**inputs, max_length=max_length)
        return self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


if __name__ == "__main__":
    model_path = "/disks/disk1/share/models/Video-LLaVA-7B-hf"
    video_path = "/disks/disk5/private/yyyou/InternLM-XComposer-main/examples/liuxiang.mp4"
    question = "USER: <video>Describe this video. ASSISTANT:"
    model = VideoLLaVA(model_path, video_path, question)
    description = model.generate_description()
    print(description)