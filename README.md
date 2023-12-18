# EgoThink (Can Vision-Language Models Think from a First-Person Perspective?)

[**🌐 Homepage**](https://adacheng.github.io/EgoThink/) | [**🤗 Dataset**](https://huggingface.co/datasets/EgoThink/EgoThink) | [**🤗 Paper**](https://huggingface.co/papers/2311.15596) | [**📖 arXiv**](https://arxiv.org/abs/2311.15596) | [**GitHub**](https://github.com/AdaCheng/EgoThink)

## Dataset

## Evaluation

### Add new model

1. create `Test_{new_model}.py` in `/models`.
2. Add the new model in `get_model()` in `/models/__init__.py`.
```sh
# BLIP2-7B
 if model_name == 'blip2-7b':
        from .test_blip2 import TestBlip2
        return TestBlip2(name='blip2_opt', model_type='pretrain_opt6.7b', config_path='/models/blip_configs/blip2_pretrain_opt6.7b.yaml', device=device)
```

### Inference
```sh
# dataset: Activity, Object/existence, etc., currently does not support direct calls from hugging face
# MODEL: models defined in the models file
# DEVICE: GPU id, 0/1/2..., currently only single card can run
python eval.py \
    --model_name $MODEL \
    --annotation_path /${dataset}/annotations.json \
    --answer_path /answer/${dataset} \
    --batch_size 1 \
    --device $DEVICE
```

### Evaluation
```sh
# dataset: Activity, Object/existence, etc.
# EVA_MODELS: a list of models to be evaluated (separated by spaces), for example "llava-13b-llama2 llava-1.5-13b llava-1.5-7b"
# $EVA_JUDGE_MODEL: gpt-4 (default), gpt-3.5-turbo, claude-2, etc.

export OPENAI_API_KEY= 
export ANTHROPIC_API_KEY=
export OPENAI_API_BASE=

python  gen_judgment.py \
    --data-folder data_egothink \
    --bench-name $dataset \
    --mode single \
    --model-list $EVA_MODELS \
    --judge-model $EVA_JUDGE_MODEL 
    --parallel 4
    --judge-file judge_prompts.jsonl
```

### Show Results

```sh
# EVA_MODELS: a list of models to be evaluated (separated by spaces), for example "llava-13b-llama2 llava-1.5-13b llava-1.5-7b"
# $EVA_JUDGE_MODEL: gpt-4 (default), gpt-3.5-turbo, claude-2, etc.
python show_result.py \
    --input-file {data_folder}/{bench-name}/model_judgment/{judge-model}_single.jsonl \
    --judge-model $EVA_JUDGE_MODEL \
    --model-list  $EVA_MODELS \
    --mode single
```

### 

## Contact
- Sijie Cheng: csj23@mails.tsinghua.edu.cn

## Citation

```bibtex
@article{cheng2023can,
  title={Can Vision-Language Models Think from a First-Person Perspective?},
  author={Cheng, Sijie and Guo, Zhicheng and Wu, Jingwen and Fang, Kechen and Li, Peng and Liu, Huaping and Liu, Yang},
  journal={arXiv preprint arXiv:2311.15596},
  year={2023}
}
```

## Acknowledge
Thanks to Xiaolong Wang, Yangyang Yu, Zixin Sun, and Zhaoyang Li for their contributions to data collection and construction.
We appreciate Zeyuan Yang, Szymon Tworkowski, Guan Wang, and Zonghan Yang for their support of API resources; Xinghang Li for his valuable discussion; Siyu Wang for her code base on the annotation system.

Furthermore, we appreciate the developers behind the following projects for their significant contributions to our research: [Ego4D](https://github.com/facebookresearch/Ego4d), [Multi-Modality-Arena](https://github.com/OpenGVLab/Multi-Modality-Arena/tree/main/tiny_lvlm_evaluation), [FastChat](https://github.com/lm-sys/FastChat).
