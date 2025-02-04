import torch
from torchvision import transforms

def _convert(img: torch.Tensor):
    return img.repeat(3, 1, 1)

def _from0_to1(img: torch.Tensor):
    img = (img + torch.abs(img))
    img = img / torch.max(img)
    return img

def _from0_to255(img: torch.Tensor):
    img = (img + torch.abs(img))
    img = img / torch.max(img) * 255
    return img
    
def transform(img):
    """transform Grayscale Image to feed to CLIP Image Encoder"""
    tf = transforms.Compose([
        transforms.ToTensor(),
        _convert,
        _from0_to255,
        transforms.CenterCrop(224),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    return tf(img)

def transform_mask(img):
    """transform Grayscale Image to feed to CLIP Image Encoder"""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224),
    ])
    return tf(img)