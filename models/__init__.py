import torch
import numpy as np
from PIL import Image

DATA_DIR = '/VLEAI_bench'

def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip


def get_image(image):
    if type(image) is str:
        try:
            return Image.open(image).convert("RGB")
        except Exception as e:
            print(f"Fail to read image: {image}")
            exit(-1)
    elif type(image) is Image.Image:
        return image
    else:
        raise NotImplementedError(f"Invalid type of Image: {type(image)}")


def get_BGR_image(image):
    image = get_image(image)
    image = np.array(image)[:, :, ::-1]
    image = Image.fromarray(np.uint8(image))
    return image


def get_model(model_name, device=None):
    model_name = model_name.lower()
    # if model_name == 'BLIP2'.lower():
    #     from .test_blip2 import TestBlip2
    #     return TestBlip2(device)
    # el
    if model_name == 'blip2-7b':
        from .test_blip2 import TestBlip2
        return TestBlip2(name='blip2_opt', model_type='pretrain_opt6.7b', config_path='/models/blip_configs/blip2_pretrain_opt6.7b.yaml', device=device)
    elif model_name == 'blip2-13b':
        from .test_blip2 import TestBlip2
        return TestBlip2(name='blip2_t5', model_type='pretrain_flant5xxl', config_path='/models/blip_configs/blip2_pretrain_flant5xxl.yaml', device=device)
    elif model_name == 'mPLUG-Owl'.lower():
        from .test_mplug_owl import TestMplugOwl
        return TestMplugOwl(device)
    elif model_name == 'Otter-Image'.lower():
        from .test_otter_image import TestOtterImage
        return TestOtterImage(name='image', device=device)
    elif model_name == 'Otter-Video'.lower():
        from .test_otter_image import TestOtterImage
        return TestOtterImage(name='video', device=device)
    elif model_name == 'InstructBLIP-7b'.lower():
        from .test_blip2 import TestBlip2
        return TestBlip2(device=device, config_path='/models/blip_configs/blip2_instruct_vicuna7b.yaml', name='blip2_vicuna_instruct', model_type='vicuna7b')
    elif model_name == 'InstructBLIP-13b-flant5'.lower():
        from .test_blip2 import TestBlip2
        return TestBlip2(device=device, config_path='/models/blip_configs/blip2_instruct_flant5xxl.yaml', name='blip2_t5_instruct', model_type='flant5xxl')
    elif model_name == 'InstructBLIP-13b-vicuna'.lower():
        from .test_blip2 import TestBlip2
        return TestBlip2(device=device, config_path=None, name='blip2_vicuna_instruct', model_type='vicuna13b')
    elif model_name == 'VPGTrans'.lower():
        from .test_vpgtrans import TestVPGTrans
        return TestVPGTrans(device)
    elif model_name == 'LLaVA-7b'.lower():
        from .test_llava2 import TestLLaVA2
        return TestLLaVA2(size='7b', device=device)
    elif model_name == 'LLaVA-13b'.lower():
        from .test_llava2 import TestLLaVA2
        return TestLLaVA2(size='13b', device=device)
    elif model_name == 'LLaVA-13b-llama2'.lower():
        from .test_llava2 import TestLLaVA2
        return TestLLaVA2(size='13b-llama2', device=device)
    elif model_name == 'LLaVA-1.5-7b'.lower():
        from .test_llava2 import TestLLaVA2
        return TestLLaVA2(size='1.5-7b', device=device)
    elif model_name == 'LLaVA-1.5-13b'.lower():
        from .test_llava2 import TestLLaVA2
        return TestLLaVA2(size='1.5-13b', device=device)
    elif model_name == 'sharegpt4v'.lower():
        from .test_llava2 import TestLLaVA2
        return TestLLaVA2(size='7b-sharegpt4v', device=device)
    elif model_name == 'LLaMA-Adapter-v2'.lower():
        from .test_llama_adapter_v2 import TestLLamaAdapterV2
        return TestLLamaAdapterV2(device)
    elif model_name == 'minigpt-4'.lower():
        from .test_minigpt4 import TestMiniGPT4
        return TestMiniGPT4(device)
    elif model_name == 'pandagpt-7b':
        from .test_pandagpt import TestPandaGPT
        return TestPandaGPT('7b', device)
    elif model_name == 'pandagpt-13b':
        from .test_pandagpt import TestPandaGPT
        return TestPandaGPT('13b', device)
    elif 'OFv2'.lower() in model_name:
        _, version = model_name.split('_')
        from .test_OFv2 import OFv2
        return OFv2(version, device)
    elif model_name == 'openflamingo':
        from .test_openflamingo import TestOpenFlamingo
        return TestOpenFlamingo(device)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")