from lmdeploy import TurbomindEngineConfig, pipeline, GenerationConfig
from lmdeploy.vl import load_image
from transformers.dynamic_module_utils import get_class_from_dynamic_module

class TestiXC2d5:
    def __init__(self, model_path: str, video_path: str):
        """
        初始化推理类，设置模型和视频路径，初始化 NVIDIA SMI。
        :param model_path: 本地模型路径
        :param video_path: 视频文件路径
        """
        self.model_path = model_path
        self.video_path = video_path
        self.pipe = None

        # 加载视频处理相关的动态模块
        self.load_video = get_class_from_dynamic_module('ixc_utils.load_video', self.model_path)
        self.frame2img = get_class_from_dynamic_module('ixc_utils.frame2img', self.model_path)
        self.Video_transform = get_class_from_dynamic_module('ixc_utils.Video_transform', self.model_path)
        self.get_font = get_class_from_dynamic_module('ixc_utils.get_font', self.model_path)

        # 初始化 pipeline
        engine_config = TurbomindEngineConfig(model_format='awq', cache_max_entry_count=0.1)
        self.pipe = pipeline(self.model_path, backend_config=engine_config)

    def preprocess_video(self):
        """
        加载并预处理视频，生成图片。
        :return: 处理后的图片
        """
        video = self.load_video(self.video_path)
        img = self.frame2img(video, self.get_font())
        return self.Video_transform(img)

    def run_inference(self, query: str, top_k: int = 50, top_p: float = 0.8, temperature: float = 1.0):
        """
        运行推理并生成结果。
        :param query: 推理时的查询
        :param top_k: top_k 参数
        :param top_p: top_p 参数
        :param temperature: temperature 参数
        :return: 推理结果文本
        """
        img = self.preprocess_video()
        gen_config = GenerationConfig(top_k=top_k, top_p=top_p, temperature=temperature)
        sess = self.pipe.chat((query, img), gen_config=gen_config)
        return sess.response.text

# 使用示例
if __name__ == "__main__":
    # model_path = '/disks/disk1/share/models/internlm-xcomposer2d5-7b-4bit'
    video_path = '/data/home/adaaaacheng/csj/videgothink/goalstep_val_clean/151.mp4'
    # 初始化推理类
    inference = VideoInference(model_path, video_path)
    # 执行推理
    query = 'Is there a cupboard in the video?'
    result = inference.run_inference(query)
    print(result)