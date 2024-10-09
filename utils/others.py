import random

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_captions(image, caption_model, caption_processor, device):
    with torch.no_grad():  # image: tensor, [b,h,w,3], uint8, 0~255
        inputs = caption_processor(image, return_tensors="pt").to(device, torch.float16)
        generated_ids = caption_model.generate(**inputs, max_new_tokens=20)
        generated_text = caption_processor.batch_decode(generated_ids, skip_special_tokens=True)  # [0].strip()

    return generated_text


def inpainting_resize(image, size, interval=32, is_mask=False, square=False):
    # specific resizing method for inpainting
    # resize the shortest side to "size"
    if len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape

    if type(size) == tuple or type(size) == list:
        h_new, w_new = size[0], size[1]
        interval = 1
    elif square:
        h_new, w_new = size, size
    else:
        if w > h:
            s = size / h
            h_new = size
            w_new = int(w * s)
        else:
            s = size / w
            w_new = size
            h_new = int(h * s)

    w_new = w_new // interval * interval
    h_new = h_new // interval * interval

    if is_mask:
        if w_new <= w:
            image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_AREA)
            image[image > 0] = 1
        else:
            image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    else:
        if w_new <= w:
            image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LANCZOS4)

    return image


def get_lpips_score(loss_fn_alex, gt, pred, device):  # gt, pred: [h,w,3]
    gt_img_lpips = lpips.im2tensor(gt * 255).to(device)
    pred_img_lpips = lpips.im2tensor(pred * 255).to(device)
    lpips_ = loss_fn_alex(gt_img_lpips, pred_img_lpips).item()

    return lpips_


def get_clip_score(clip_model, gt, pred, device):  # gt, pred: [h,w,3]
    clip_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                              (0.26862954, 0.26130258, 0.27577711))])

    pred_img_clip = clip_transform(pred)[None].to(device)
    pred_img_clip = F.interpolate(pred_img_clip.to(dtype=torch.float32), [224, 224], mode='bicubic')
    pred_clip_feat = clip_model.visual(pred_img_clip.to(dtype=clip_model.ln_final.weight.dtype))
    pred_clip_feat = pred_clip_feat / pred_clip_feat.norm(dim=-1, keepdim=True)

    ref_img_clip = clip_transform(gt)[None].to(device).to(device, dtype=clip_model.ln_final.weight.dtype)
    ref_img_clip = F.interpolate(ref_img_clip.to(dtype=torch.float32), [224, 224], mode='bicubic')
    ref_clip_feat = clip_model.visual(ref_img_clip.to(dtype=clip_model.ln_final.weight.dtype))
    ref_clip_feat = ref_clip_feat / ref_clip_feat.norm(dim=-1, keepdim=True)

    clip_ = (pred_clip_feat * ref_clip_feat).sum().item()

    return clip_
