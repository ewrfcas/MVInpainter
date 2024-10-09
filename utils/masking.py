import math
import random
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.append('.')
from utils.generate_masks import RandomRectangleMaskGenerator, RandomIrregularMaskGenerator

root_path = "./data/masks"

deepfillv2_mask_path = "./data/masks/deepfillv2_mask/irregular_mask_list.txt"
coco_mask_path = "./data/masks/coco_mask/coco_mask_list.txt"

# used for 512*512 only! used for first version inpainting
rec_mask_gen = RandomRectangleMaskGenerator(margin=0, bbox_min_size=50, bbox_max_size=300, min_times=1, max_times=4)
irr_tick_mask_gen = RandomIrregularMaskGenerator(max_angle=4, min_len=50, max_len=450, min_width=50, max_width=250, min_times=1, max_times=5)
irr_medium_mask_gen = RandomIrregularMaskGenerator(max_angle=4, min_len=10, max_len=250, min_width=20, max_width=100, min_times=10, max_times=30)
irr_thin_mask_gen = RandomIrregularMaskGenerator(max_angle=4, min_len=5, max_len=100, min_width=10, max_width=70, min_times=45, max_times=90)
irr_gens = [irr_thin_mask_gen, irr_medium_mask_gen, irr_tick_mask_gen]

rec_mask_gen2 = RandomRectangleMaskGenerator(margin=0, bbox_min_size=50, bbox_max_size=250, min_times=1, max_times=4)
irr_tick_mask_gen2 = RandomIrregularMaskGenerator(max_angle=4, min_len=50, max_len=300, min_width=50, max_width=150, min_times=1, max_times=4)
irr_medium_mask_gen2 = RandomIrregularMaskGenerator(max_angle=4, min_len=10, max_len=200, min_width=20, max_width=80, min_times=10, max_times=25)
irr_thin_mask_gen2 = RandomIrregularMaskGenerator(max_angle=4, min_len=5, max_len=100, min_width=10, max_width=50, min_times=40, max_times=80)
irr_gens2 = [irr_thin_mask_gen2, irr_medium_mask_gen2, irr_tick_mask_gen2, rec_mask_gen2]


def load_mask_list(file_path):
    mask_list = []
    with open(file_path) as f:
        for line in f:
            mask_list.append(f"{root_path}/{line.strip()}")

    return mask_list


deepfillv2_mask_list = load_mask_list(deepfillv2_mask_path)
coco_mask_list = load_mask_list(coco_mask_path)

def load_mask(mask_list, h, w):
    mask_index = random.randint(0, len(mask_list) - 1)
    mask = cv2.imread(mask_list[mask_index], cv2.IMREAD_GRAYSCALE)
    # mask = Image.open(mask_list[mask_index]).convert("L")
    # mask = np.array(mask)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = mask / 255

    return mask


# FIXME: hard code for mask loading. It loads predefined masks from Deepfillv2, COCOsegment, LaMa mask, and square mask.
def get_inpainting_mask(h, w, upper=0.85):
    rdv = random.random()
    # 100% containing: lama mask (global, thin=20%, medium=40%, tick=40%)
    # deepfillv2, coco, square分别为30%, 30%, 30%
    probas = np.array([0.2, 0.4, 0.4])
    probas /= probas.sum()
    kind = np.random.choice(len(probas), p=probas)
    mask = irr_gens[kind](np.zeros((3, h, w)))[0]

    if np.mean(mask) < upper:
        if rdv < 0.3:
            add_mask = load_mask(deepfillv2_mask_list, h, w)
        elif 0.3 <= rdv < 0.6:
            add_mask = load_mask(coco_mask_list, h, w)
        elif 0.6 <= rdv < 0.9:
            add_mask = rec_mask_gen(mask[None])[0]
        else:
            add_mask = np.zeros_like(mask)
    else:
        add_mask = np.zeros_like(mask)

    mask = np.clip(mask + add_mask, 0, 1).astype(np.float32)
    mask[mask > 0] = 1
    mask = torch.tensor(mask, dtype=torch.float32)[None]

    return mask


def random_irregular_object_mask(obj_mask, mask_enlarge=[0.0, 0.1], pts_size=[20, 30], width_range=[40, 60]):
    mask_bbox = np.where(obj_mask > 0)
    h_min, h_max = mask_bbox[0].min(), mask_bbox[0].max()
    w_min, w_max = mask_bbox[1].min(), mask_bbox[1].max()
    h, w = obj_mask.shape[0], obj_mask.shape[1]

    mask_enlarge_ = np.random.random() * (mask_enlarge[1] - mask_enlarge[0]) + mask_enlarge[0]
    if mask_enlarge_ > 0:
        # print("enlarge_rate", enlarge_rate)
        max_diff = max(h_max - h_min, w_max - w_min) * mask_enlarge_
        h_min = np.clip(h_min - max_diff, 0, h - 1)
        h_max = np.clip(h_max + max_diff, 0, h - 1)
        w_min = np.clip(w_min - max_diff, 0, w - 1)
        w_max = np.clip(w_max + max_diff, 0, w - 1)

    pts_size_ = np.random.randint(pts_size[0], pts_size[1] + 1)
    # print("pts_size", pts_size_)
    random_x = np.random.randint(w_min, w_max, size=pts_size_)
    random_y = np.random.randint(h_min, h_max, size=pts_size_)
    random_pts = np.stack([random_x, random_y], axis=1)
    irr_mask = Image.new('L', (w, h), 0)

    min_width_ = width_range[0] * (w / 512)
    max_width_ = width_range[1] * (w / 512)
    width = np.random.randint(min_width_, max_width_)
    # print("width", width)
    draw = ImageDraw.Draw(irr_mask)
    pts = np.append(random_pts, random_pts[:1], axis=0)
    pts = pts.astype(np.float32)
    draw.line(pts, fill=1, width=width)
    for v in pts:
        draw.ellipse((v[0] - width // 2, v[1] - width // 2, v[0] + width // 2, v[1] + width // 2), fill=1)
    irr_mask = np.asarray(irr_mask, np.float32).copy()
    mask = np.clip(obj_mask + irr_mask, 0, 1)

    return mask


def get_object_inpainting_mask(
        h, w,
        fg_probability,  # [1,h,w]
        upper=0.5,
        lama_sample_probs=[0.2, 0.3, 0.3, 0.2],
        lama_mask_rate=0.5,
        object_mask_rate=0.85,
        object_bbox_mask_rate=0.3,
        dilate_range=[10, 25],
        object_irr_mask_rate=0.99,
        mask_enlarge=[0.01, 0.1],
        pts_size=[20, 30],
        width_range=[30, 60],
):
    no_obj_mask = False
    if random.random() < object_mask_rate:
        dilate = random.randint(dilate_range[0], dilate_range[1])
        kernel = np.ones((dilate, dilate), np.uint8)
        fg_probability[fg_probability > 0] = 1
        fg_probability = fg_probability[0].numpy()
        obj_mask = cv2.dilate(fg_probability, kernel, iterations=1)
        obj_mask = cv2.resize(obj_mask, [w, h], interpolation=cv2.INTER_NEAREST)
        if np.sum(obj_mask) == 0:
            no_obj_mask = True
        if random.random() < object_bbox_mask_rate and not no_obj_mask:
            y_coords, x_coords = np.nonzero(obj_mask)
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            obj_mask[y_min:y_max, x_min:x_max] = 1
            dilate = random.randint(5, 15)
            kernel = np.ones((dilate, dilate), np.uint8)
            obj_mask = cv2.dilate(obj_mask, kernel, iterations=1)
        else:
            if random.random() < object_irr_mask_rate and not no_obj_mask:
                obj_mask = random_irregular_object_mask(obj_mask,
                                                        mask_enlarge=mask_enlarge,
                                                        pts_size=pts_size,
                                                        width_range=width_range)
        mask = obj_mask
    else:
        mask = np.zeros((h, w), dtype=np.float32)
        no_obj_mask = True

    if no_obj_mask or random.random() < lama_mask_rate:
        probas = np.array(lama_sample_probs)
        probas /= probas.sum()
        kind = np.random.choice(len(probas), p=probas)
        lama_mask = irr_gens2[kind](np.zeros((3, h, w)))[0]
        mask = np.clip(mask + lama_mask, 0, 1).astype(np.float32)

    rdv = random.random()
    if np.mean(mask) < upper:  # 补充mask
        if rdv < 0.45:
            add_mask = load_mask(deepfillv2_mask_list, h, w)
        elif 0.45 <= rdv < 0.9:
            add_mask = load_mask(coco_mask_list, h, w)
        else:
            add_mask = np.zeros_like(mask)
    else:
        add_mask = np.zeros_like(mask)

    mask = np.clip(mask + add_mask, 0, 1).astype(np.float32)
    mask[mask > 0] = 1
    mask = torch.tensor(mask, dtype=torch.float32)[None]

    return mask


def get_object_enlarged_mask(obj_mask, obj_dilate=15):
    size = obj_mask.shape[-1]
    obj_mask[obj_mask > 0] = 1
    obj_mask_ = obj_mask[0].numpy().copy()
    kernel = np.ones((math.ceil(obj_dilate * size / 512), math.ceil(obj_dilate * size / 512)), np.float32)
    obj_mask_ = cv2.dilate(obj_mask_, kernel, iterations=1)
    return obj_mask_


def get_object_bbox_mask(obj_mask, obj_dilate=15):
    obj_mask_ = get_object_enlarged_mask(obj_mask, obj_dilate)
    y_coords, x_coords = np.nonzero(obj_mask_)
    x_min = x_coords.min()
    x_max = x_coords.max()
    y_min = y_coords.min()
    y_max = y_coords.max()
    obj_mask_[y_min:y_max, x_min:x_max] = 1
    return obj_mask_


if __name__ == '__main__':
    pass

    # mask_rates = []
    # for i in tqdm(range(1000)):
    #     mask = get_object_inpainting_mask(512, 512, upper=0.5)
    #     mask_rates.append(mask.mean().item())
    #     # mask = get_inpainting_mask(512, 512, upper=0.85)
    #     # mask = (mask[0] * 255).numpy().astype(np.uint8)
    #     # cv2.imwrite(f"./data/masks/fixed_mask_1000/{str(i).zfill(4)}.png", mask)
    #
    # with open("./data/co3d_subset/masking.txt", 'w') as w:
    #     for m in mask_rates:
    #         w.write(str(m) + "\n")
