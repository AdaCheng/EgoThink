import random
from typing import Optional, Union, Dict, Any, List

from einops import rearrange, repeat
import torch
import math
import PIL.Image
import PIL.ImageSequence
import numpy as np
import PIL
from PIL import Image

from transformers.utils import TensorType, requires_backends, is_torch_dtype, is_torch_device
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers import AutoImageProcessor
from transformers.image_transforms import to_channel_dimension_format
from transformers.image_utils import (
    ImageInput, 
    make_list_of_images, 
    valid_images, 
    is_torch_tensor, 
    is_batched,
    to_numpy_array, 
    infer_channel_dimension_format,
    ChannelDimension
)
from torchvision.ops.boxes import box_area
from torchvision.transforms import functional as F
from torchvision.transforms.transforms import InterpolationMode
from torchvision import transforms

def recursive_converter(converter, value):
    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value += [recursive_converter(converter, v)]
        return new_value
    else:
        return converter(value)

def box_iou(boxes1, area1, boxes2, eps=1e-5):
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union+eps)
    return iou, union

available_anchor_strategy = ['docowl', 'random', 'highest', 'last', 'llava']

grid_dict = {
    'grid_33':[
        (1,1),
        (1,2),(2,1),
        (1,3),(3,1),
        (2,2),(1,4),(4,1),
        (1,5),(5,1),
        (1,6),(6,1),(2,3),(3,2),
        (1,7),(7,1),
        (4,2),(2,4),(1,8),(8,1),
        (3,3),(1,9),(9,1)],
    'grid_squ_3x3':[
        (1,1),(2,2),(3,3)
    ],
    'grid_squ_4':[
        (2,2),(1,3),(1,4),(3,1),(4,1)
    ],
    'grid_squ_6':[
        (2,2),(1,3),(1,4),(3,1),(4,1), (2,3),(3,2)
    ],
    'grid_squ_2':[
        (2,1)
    ],
    'grid_squ_9':[
        (1,1),
        (1,2),(2,1),
        (1,3),(3,1),
        (2,2),(1,4),(4,1),
        (1,5),(5,1),
        (1,6),(6,1),(2,3),(3,2),
        (1,7),(7,1),
        (4,2),(2,4),(1,8),(8,1),
        (3,3),(1,9),(9,1)],
}

cut_prompt_template_dict = {
    'v0': lambda img_token, h, w: f''.join([f"{img_token}" for i in range(h) for j in range(w)]),
    'v1': lambda img_token, h, w: f'Cut to {h} rows {w} columns, '+ ' '.join([f"subimg({i},{j}){img_token}"for i in range(h) for j in range(w)]),
    'v1_global': lambda img_token, h, w: f'Cut to {h} rows {w} columns with a global view, '+ ' '.join([f"subimg({i},{j}){img_token}"for i in range(h) for j in range(w)]+[f"global_view{img_token}"]),
    'v2_global': lambda img_token, h, w: f'Cut to {h} rows {w} columns with a global view\n'+ '\n'.join([' '.join([f"subimg({i},{j}){img_token}" for j in range(w)]) for i in range(h)])+f"\nglobal_view{img_token}",
}

def anchor_rank(anchors, anchors_areas, input_image_size, eps=1e-5):
    # anchors x1 y1 x2 y2

    # image_size: (h, w)
    # xyxy
    input_image_bbox = torch.tensor([0, 0, input_image_size[1], input_image_size[0]]).unsqueeze(0)

    boxes1 = anchors
    boxes2 = input_image_bbox
    boxes3 = anchors.clone()
    # y2
    boxes3[:,3] = input_image_size[0]/input_image_size[1]*anchors[:,2] # 用于算分辨率无关的iou
    
    area1 = anchors_areas
    
    iou, _ = box_iou(boxes1, area1, boxes2)
    iou = iou.squeeze(1)
    shape_iou, _ = box_iou(boxes1, area1, boxes3)
    shape_iou = shape_iou.diag()
    # 优先匹配形状接近 再匹配分辨率接近
    index = torch.argmax(shape_iou*100+iou,dim=0)
    return index

def select_best_resolution(anchors, anchors_areas, input_image_size): # TODO For a futher check
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_size = (input_image_size[1], input_image_size[0])
    possible_resolutions = [(_[2], _[3]) for _ in anchors] # xyxy -> w,h

    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    index = 0
    for i, (width, height) in enumerate(possible_resolutions):
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)
            index = i

    return index

def build_cut_shape_indices(cut_shape):
    # cut_shape: a list of (nh,nw)
    cut_shape_indices = []
    for shape in cut_shape:
        n=shape[0]*shape[1]
        indices = torch.cat([
            repeat(torch.tensor(shape),'l -> n l',n=n),
            torch.arange(n).unsqueeze(1)
        ], dim=1)
        assert indices.shape[0] == n
        assert indices.shape[1] == 3 # nh,nw,idx

        cut_shape_indices.append(indices)
    cut_shape_indices = torch.cat(cut_shape_indices,dim=0).long()
    return cut_shape_indices
    
class AnchorResize(torch.nn.Module):
  
    def __init__(self, image_size, anchors, interpolation=InterpolationMode.BILINEAR, antialias=None, anchor_strategy='docowl'):
        super().__init__()
        self.image_size = image_size
        # xyxy
        self.anchors = torch.tensor(
            [[0, 0, _[1]*image_size[1], _[0]*image_size[0]] 
            for _ in anchors], requires_grad=False
        )
        
        self.anchor_areas = box_area(self.anchors)

        self.interpolation = interpolation
        self.antialias = antialias
        self.anchor_strategy = anchor_strategy
        assert self.anchor_strategy in available_anchor_strategy

    def resize_global(self, img):
        return F.resize(img, self.image_size, self.interpolation, max_size=None, antialias=self.antialias)

    def forward(self, img, skip_resize=False):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if self.anchor_strategy == 'docowl':
            selected_anchor = anchor_rank(self.anchors, self.anchor_areas, (img.size[1], img.size[0]))
        elif self.anchor_strategy == 'random':
            selected_anchor = random.randint(0,len(self.anchors)-1)
        elif self.anchor_strategy == 'highest':
            # 选面积最大的 在这个基础上 尽可能选最方正的
            selected_anchor = torch.argmax(self.anchors[:,2]*self.anchors[:,3]*100-torch.abs(self.anchors[:,2]-self.anchors[:,3]))
        elif self.anchor_strategy == 'last':
            selected_anchor = len(self.anchors)-1
        elif self.anchor_strategy == 'llava':
            selected_anchor = select_best_resolution(self.anchors, self.anchor_areas, (img.size[1], img.size[0]))
        else:
            selected_anchor = None
        assert selected_anchor is not None

        target_size = self.anchors[selected_anchor][2:].tolist() # w,h
        if skip_resize:
            # for debug
            return selected_anchor
        return F.resize(img, [target_size[1],target_size[0]], self.interpolation, max_size=None, antialias=self.antialias), selected_anchor

    def __repr__(self) -> str:
        detail = f"(size={self.image_size}, anchor={self.anchors}, interpolation={self.interpolation.value}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"

class CutMixin:
    def __init__(self, cut_cfg={"anchors": "grid_squ_6", "anchor_strategy": "docowl", "cut_prompt": "v2", "add_global": True, "cut_prob": 1.0}) -> None:
        if cut_cfg is None:
            self.cut_enable = False
            return
        else:
            self.cut_enable = True
        image_size = self.image_size
        anchors = cut_cfg.get('anchors','grid_33')
        anchor_strategy = cut_cfg.get('anchor_strategy','docowl')
        cut_prompt = cut_cfg.get('cut_prompt','v0')
        self.cut_prob = cut_cfg.get('cut_prob', 1.0)
        
        self.force_shape_cut = cut_cfg.get('force_shape_cut', False)
        force_shape_cut_anchors = cut_cfg.get('force_shape_cut_anchors', 'force_shape_cut_anchors')


        self.add_global = cut_cfg.get('add_global', False)
        
        # h,w
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

        if anchors in grid_dict:
            anchors = grid_dict[anchors]
        else:
            anchors = eval(anchors)
        self.anchors = [tuple(_) for _ in anchors]
        self.anchor_max = max([max(_) for _ in self.anchors])
        self.resizer = AnchorResize(image_size=image_size, anchors=anchors, interpolation=InterpolationMode.BICUBIC, anchor_strategy=anchor_strategy)

        if force_shape_cut_anchors in grid_dict:
            force_shape_cut_anchors = grid_dict[force_shape_cut_anchors]
        else:
            force_shape_cut_anchors = eval(force_shape_cut_anchors)
        self.force_shape_cut_anchors = [tuple(_) for _ in force_shape_cut_anchors]
        self.force_shape_cut_anchors_max = max([max(_) for _ in self.force_shape_cut_anchors])
      


        self.old_resizer = transforms.Resize(image_size,interpolation=InterpolationMode.BICUBIC)

        # 把image processor的缩放去掉 只保留后面的变换
        self.image_transform = transforms.Compose(self.image_transform.transforms[1:])
        if self.add_global:
            self.cut_prompt_template = cut_prompt_template_dict[cut_prompt+'_global']
        else:
            self.cut_prompt_template = cut_prompt_template_dict[cut_prompt]

        self.media_tokens = ["<|image|>", "<|video|>"]



    def _process_image(self, images):
        new_images = []
        cut_shape = []
        for image in images:
            raw_image = image
            
            image, selected_anchor = self.resizer(image)
            image_input = self.image_transform(image) # h,w,3 -> 3,h,w
            cut_shape.append((image_input.shape[1]//self.image_size[0], image_input.shape[2]//self.image_size[1])) # cut_h, cut_w
            image_input = rearrange(image_input, 'C (num_h h) (num_w w) -> (num_h num_w) C h w', h=self.image_size[0], w=self.image_size[1])
         
            new_images.append(image_input)
        
            if self.add_global:
                new_images.append(self.image_transform(self.resizer.resize_global(raw_image)).unsqueeze(0))
                cut_shape.append((1,1))

        new_images = torch.cat(new_images,dim=0)
        cut_shape_indices = build_cut_shape_indices(cut_shape)
        return new_images, cut_shape, cut_shape_indices

class mPLUGOwl3BatchFeature(BatchFeature):
    r"""
    Extend from BatchFeature for supporting various image size
    """
    def __init__(self, data: Optional[Dict[str, Any]] = None, tensor_type: Union[None, str, TensorType] = None):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        if tensor_type is None:
            return self
        
        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type)

        def converter(value):
            try:
                if not is_tensor(value):
                    tensor = as_tensor(value)
                    return tensor
            except:  # noqa E722
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )


        for key, value in self.items():
            self[key] = recursive_converter(converter, value)
        return self
            
    def to(self, *args, **kwargs) -> "mPLUGOwl3BatchFeature":
        requires_backends(self, ["torch"])
        import torch

        def cast_tensor(v):
            # check if v is a floating point
            if torch.is_floating_point(v):
                # cast and send to device
                return v.to(*args, **kwargs)
            elif device is not None:
                return v.to(device=device)
            else:
                return v

        new_data = {}
        device = kwargs.get("device")
        # Check if the args are a device or a dtype
        if device is None and len(args) > 0:
            # device should be always the first argument
            arg = args[0]
            if is_torch_dtype(arg):
                # The first argument is a dtype
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                device = arg
            else:
                # it's something else
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")
        # We cast only floating point tensors to avoid issues with tokenizers casting `LongTensor` to `FloatTensor`
        for k, v in self.items():
            new_data[k] = recursive_converter(cast_tensor, v)
        self.data = new_data
        return self


class mPLUGOwl3ImageProcessor(BaseImageProcessor, CutMixin):
    model_input_names = ["pixel_values"]

    def __init__(
            self, 
            image_size,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        CutMixin.__init__(self)

    def preprocess(
            self, 
            images: Union[Image.Image, List[Image.Image]],
            cut_enable=True,
            **kwargs
        ) -> mPLUGOwl3BatchFeature:
        if isinstance(images, Image.Image):
            images_list = [images]
        else:
            images_list = images

        if self.cut_enable and cut_enable:
            image_data, cut_shape, cut_shape_indices = self._process_image(images_list)
        else:
            image_data = [self.image_transform(self.resizer.resize_global(image)) for image in images_list]
            image_data = torch.stack(image_data, dim=0)
            cut_shape = cut_shape_indices = None
            
        return mPLUGOwl3BatchFeature(data={'pixel_values': image_data, 'cut_shape':cut_shape, 'cut_shape_indices':cut_shape_indices})

AutoImageProcessor.register("mPLUGOwl3ImageProcessor", mPLUGOwl3ImageProcessor)
