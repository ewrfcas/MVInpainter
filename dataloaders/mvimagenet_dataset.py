import collections
import os
import pickle
from glob import glob

import torch.nn.functional as F
from easydict import EasyDict
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from dataloaders.co3d_dataset import _seq_name_to_seed
from utils.masking import *


def process_prompt(prompt):
    prompt = prompt.replace("The video features", "").strip()
    prompt = prompt.replace("The video showcases", "").strip()

    return prompt

mvi_map = "./data/mvimagenet/mvimgnet_category.txt"
class_name_map = {}
with open(mvi_map, 'r') as f:
    for line in f:
        [code, class_name] = line.strip().split(",")
        class_name_map[code] = class_name


def inverse_crop(image, mask):
    h = image.shape[0]
    w = image.shape[1]
    if h == w:
        return image, mask

    y_pos, x_pos = np.where(mask == 1)
    x_min, x_max, y_min, y_max = x_pos.min(), x_pos.max(), y_pos.min(), y_pos.max()

    if h > w:
        top_margin = y_min
        bottom_margin = h - y_max

        if top_margin > bottom_margin:
            clamp_y = 0
        else:
            clamp_y = h - w
        clamp_x = 0
    else:
        left_margin = x_min
        right_margin = w - x_max

        if left_margin > right_margin:
            clamp_x = 0
        else:
            clamp_x = w - h
        clamp_y = 0

    size = min(h, w)

    image = image[clamp_y:clamp_y + size, clamp_x:clamp_x + size]
    mask = mask[clamp_y:clamp_y + size, clamp_x:clamp_x + size]

    return image, mask


def dynamic_crop(image, class_seq_name):
    h = image.shape[0]
    w = image.shape[1]

    if h == w:
        return image

    # consistent random crop across all seq
    seed = _seq_name_to_seed(class_seq_name)

    sub = max(h, w) - min(h, w)
    size = min(h, w)
    if random.Random(seed).random() < 0.1:
        # print(class_seq_name, "resize")
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    crop_st = random.Random(seed).randint(0, sub)
    # print(class_seq_name, crop_st)
    if h > w:
        image_crop = image[crop_st:crop_st + w, :]
    else:
        image_crop = image[:, crop_st:crop_st + h]

    return image_crop


def square_crop(image, mask):
    h = image.shape[0]
    w = image.shape[1]
    y_pos, x_pos = np.where(mask == 1)
    bbox_xyxy = [x_pos.min(), y_pos.min(), x_pos.max(), y_pos.max()]

    if h > w:
        bbox_center = ((bbox_xyxy[1] + bbox_xyxy[3]) // 2)
        # bbox_center += random.randint(-int(w * 0.1), int(w * 0.1))
        y0 = bbox_center - w // 2
        y1 = bbox_center + w // 2
        if y0 < 0:
            y1 -= y0
            y0 = 0
        elif y1 > h:
            y0 -= (y1 - h)
            y1 = h
        clamp_bbox_xyxy = [0, y0, w, y1]
    else:
        bbox_center = ((bbox_xyxy[0] + bbox_xyxy[2]) // 2).item()
        # bbox_center += random.randint(-int(h * 0.1), int(h * 0.1))
        x0 = bbox_center - h // 2
        x1 = bbox_center + h // 2
        if x0 < 0:
            x1 -= x0
            x0 = 0
        elif x1 > w:
            x0 -= (x1 - w)
            x1 = w
        clamp_bbox_xyxy = [x0, 0, x1, h]

    image = image[clamp_bbox_xyxy[1]:clamp_bbox_xyxy[3], clamp_bbox_xyxy[0]:clamp_bbox_xyxy[2]]
    mask = mask[clamp_bbox_xyxy[1]:clamp_bbox_xyxy[3], clamp_bbox_xyxy[0]:clamp_bbox_xyxy[2]]

    return image, mask


def longest_resize(image, size, interpolation=cv2.INTER_AREA):
    h, w = image.shape[:2]

    if h > w:
        image = cv2.resize(image, (int(w * (size / h)), size), interpolation=interpolation)
    else:
        image = cv2.resize(image, (size, int(h * (size / w))), interpolation=interpolation)

    return image


class MVImageNetDataset(Dataset):
    def __init__(self, dataset_root, data_list, image_height, image_width, n_frames_per_sequence,
                 rank=0, mode="train", sampling_interval=1.0, masking_type="random", masking_params=None,
                 random_crop=False, inverse_crop=False, load_caption=False, caption_path=None,
                 **kwargs):
        super(MVImageNetDataset, self).__init__()
        self.dataset_root = dataset_root
        self.image_height = image_height
        self.image_width = image_width
        self.global_data_list = collections.defaultdict(list)  # 全局index，每个epoch不会改变
        self.n_frames_per_sequence = n_frames_per_sequence
        self.mode = mode
        self.sampling_interval = sampling_interval
        self.square_crop = kwargs.get("square_crop", True)
        self.mask_path = kwargs.get("mask_path", "./data/mvimagenet/masks")
        self.masking_type = masking_type
        self.masking_params = masking_params
        self.n_samples_per_subset = kwargs.get("n_samples_per_subset", -1)
        self.random_frame_sample = kwargs.get("random_frame_sample", True)
        self.fixed_mask_files = glob("./data/masks/fixed_mask_1000/*.png")
        self.sort_frames = kwargs.get("sort_frames", True)
        self.shuffle_val = kwargs.get("shuffle_val", False)
        self.random_crop = random_crop
        self.inverse_crop = inverse_crop
        self.longest_resize = kwargs.get("longest_resize", False)
        self.load_caption = load_caption
        self.caption_path = caption_path
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.global_seq_dict = pickle.load(open("./data/mvimagenet/global_seq_dict.pkl", "rb"))
        self.total_num = 0
        with open(data_list, "r") as f:
            for line in tqdm(f, desc=f"loading global mvimagenet {mode} sequences...", disable=rank != 0):
                line = line.strip()
                class_name, seq_name = line.split("/")
                self.global_data_list[class_name].append(seq_name)
                self.total_num += len(self.global_seq_dict[f"{class_name}_{seq_name}"])

        self.seq_list_per_epoch = []  # used for consistent random cropping

        if mode == "train":
            self.data_list_per_epoch = []  # for training, it should be set in the sampler for each epoch
        else:  # for testing, directly loading from global_data_list
            self.data_list_per_epoch = self.mvimagenet_reset_dataset(mode=mode, rank=rank)

    def __len__(self):
        return len(self.data_list_per_epoch)

    def mvimagenet_reset_dataset(self, mode="train", rank=0, **kwargs):
        file_list = []
        sequence_index = []
        random_seed = kwargs.get("random_seed", 123)

        class_names = list(self.global_data_list.keys())
        if 0 < self.n_samples_per_subset < 1:
            class_names = class_names[::int(1 / self.n_samples_per_subset)]
            n_samples_per_subset = 1
        else:
            n_samples_per_subset = self.n_samples_per_subset

        for class_name in class_names:  # tqdm(class_names, desc=f"loading mvimagenet {mode} per epoch...", disable=rank != 0):

            seq_names = self.global_data_list[class_name]
            seq_num = 0
            if mode == "train" or self.shuffle_val:
                random.Random(random_seed).shuffle(seq_names)

            for seq_name in seq_names:

                if n_samples_per_subset > 0 and seq_num >= n_samples_per_subset:
                    break

                seq_indices = self.global_seq_dict[f"{class_name}_{seq_name}"].copy()
                seq_indices = [f"{class_name}/{seq_name}/images/{l}" for l in seq_indices]

                if len(seq_indices) < self.n_frames_per_sequence:
                    continue

                if type(self.sampling_interval) == str and self.sampling_interval.startswith("random"):
                    if mode == "val":  # 验证的时候，不随机，固定1.0
                        sampling_interval_ = 1.0
                    else:
                        lower = float(self.sampling_interval.split("_")[1])
                        sampling_interval_ = lower + random.random() * (1.0 - lower)
                else:
                    sampling_interval_ = self.sampling_interval

                if sampling_interval_ < 1.0:
                    split_num = max(int(len(seq_indices) * sampling_interval_), self.n_frames_per_sequence)
                    seq_indices = seq_indices[:split_num]
                    if mode == "train":
                        rst = min(random.randint(0, int((1.0 - sampling_interval_) * len(seq_indices))), len(seq_indices) - split_num)
                        seq_indices = seq_indices[rst:rst + split_num]
                    else:
                        seq_indices = seq_indices[:split_num]

                if self.n_frames_per_sequence > 0:
                    if self.random_frame_sample:
                        # infer the seed from the sequence name, this is reproducible
                        # and makes the selection differ for different sequences
                        seed = _seq_name_to_seed(f"{class_name}/{seq_name}") + random_seed
                        seq_idx_shuffled = random.Random(seed).sample(sorted(seq_indices), len(seq_indices))
                        new_idx = seq_idx_shuffled[:self.n_frames_per_sequence]
                    else:  # 等间隔采样
                        new_idx = seq_indices[::len(seq_indices) // self.n_frames_per_sequence][:self.n_frames_per_sequence]
                else:
                    new_idx = seq_indices

                if self.sort_frames:
                    new_idx.sort(key=lambda x: x.split("/")[-1])
                file_list.extend(new_idx)
                sequence_index.extend([seq_num] * len(new_idx))

                seq_num += 1

        self.seq_list_per_epoch = sequence_index

        return file_list

    def get_frame_mask(self, index, fg_probability=None, class_name=None, image_shape=None):
        if self.masking_type == "random":  # get masks as LDM-inpainting
            mask = get_inpainting_mask(512, 512)  # 训练固定512的mask是为了好决定mask超参数
            mask = F.interpolate(mask[None], (image_shape[0], image_shape[1]), mode="nearest")[0]

        elif self.masking_type == "random_object_mask":
            mask = get_object_inpainting_mask(512, 512, fg_probability, **self.masking_params)
            mask = F.interpolate(mask[None], (image_shape[0], image_shape[1]), mode="nearest")[0]

        elif self.masking_type == "none":
            mask = torch.zeros((1, image_shape[0], image_shape[1]), dtype=torch.float32)

        elif self.masking_type == "object_mask":
            mask = get_object_enlarged_mask(fg_probability, self.masking_params.get("obj_dilate", 15))
            mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = torch.tensor(mask, dtype=torch.float32)[None]

        elif self.masking_type == "bbox":
            mask = get_object_bbox_mask(fg_probability, self.masking_params.get("obj_dilate", 15))
            mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = torch.tensor(mask, dtype=torch.float32)[None]

        elif self.masking_type == "wide_bbox":
            mask = get_object_bbox_mask(fg_probability)
            mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
            y_coords, x_coords = np.nonzero(mask)
            left_x = min(15, x_coords.min())
            right_x = max(self.image_width - 15, x_coords.max())
            y_min = y_coords.min()
            y_max = y_coords.max()
            mask[y_min:y_max, left_x:right_x] = 1
            mask = torch.tensor(mask, dtype=torch.float32)[None]

        elif self.masking_type == "object_mask+bbox_mask":
            if index % 2 == 0:  # 偶数object mask
                mask = get_object_enlarged_mask(fg_probability, self.masking_params.get("obj_dilate", 15))
            else:  # 奇数bbox mask
                mask = get_object_bbox_mask(fg_probability, self.masking_params.get("obj_dilate", 15))
            mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = torch.tensor(mask, dtype=torch.float32)[None]
        elif self.masking_type == "fixed":
            rdx = random.Random(_seq_name_to_seed(class_name)).randint(0, len(self.fixed_mask_files) - 1)
            mask_path = self.fixed_mask_files[(index + rdx) % len(self.fixed_mask_files)]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = torch.tensor(mask / 255, dtype=torch.float32)[None]
        else:
            raise NotImplementedError

        return mask

    def __getitem__(self, index):
        image = cv2.imread(f"{self.dataset_root}/{self.data_list_per_epoch[index]}")
        class_seq_name = "/".join(self.data_list_per_epoch[index].split("/")[-4:-2])
        filename = self.data_list_per_epoch[index].split("/")[-1]
        # some mvi_images are broken?
        if image is None:
            print(f"Error image of {self.dataset_root}/{self.data_list_per_epoch[index]}...")
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            mask = np.zeros((512, 512), dtype=np.float32)
            mask[128:-128, 128:-128] = 1
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(f"{self.mask_path}/{class_seq_name}/{filename.replace('.jpg', '.png')}")
            mask = mask[:, :, 0] / 255  # mask:[0~1]

        # hard code: mask为最短边512，先把image下采样到和mask一样大
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)  # large to small, using "area"

        if self.inverse_crop:
            image, mask = inverse_crop(image, mask)
        elif self.random_crop:
            image = dynamic_crop(image, f"{class_seq_name}/{self.seq_list_per_epoch[index]}")
            mask = dynamic_crop(mask, f"{class_seq_name}/{self.seq_list_per_epoch[index]}")
        elif self.square_crop:
            image, mask = square_crop(image, mask)
        image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)

        image = self.transform(image)  # value:[-1~1] shape:[3,h,w]
        mask = torch.tensor(mask, dtype=torch.float32)[None]  # [1,h,w]
        mask = self.get_frame_mask(index, mask, class_name=class_seq_name,
                                   image_shape=(image.shape[1], image.shape[2]))

        caption = ""
        try:
            if self.load_caption and self.caption_path is not None:
                # caption_path = f"{self.caption_path}/mvimagenet/{class_name_map[class_seq_name.split('/')[0]]}/{class_seq_name.split('/')[1]}.txt"
                caption_path = f"{self.caption_path}/mvimagenet/{class_seq_name.split('/')[0]}/{class_seq_name.split('/')[1]}.txt"
                if os.path.exists(caption_path):
                    with open(caption_path, "r") as f:
                        caption = process_prompt(f.readline().strip())
                    if random.random() < 0.8 or self.mode != "train":
                        caption = caption.split(".")[0] + "."
                    elif random.random() < 0.5:
                        caption = [c.strip() for c in caption.split(".")[:2]]
                        caption = ". ".join(caption) + "."
        except:
            caption = ""

        meta = {
            "image_rgb": image,
            "mask": mask,
            "sequence_category": str(class_seq_name.split("/")[0]),
            "sequence_name": str(class_seq_name.split("/")[1]),
            "tag": "mvimagenet",
            "caption": caption.strip()
        }

        return EasyDict(meta)
