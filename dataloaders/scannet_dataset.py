from torch.utils.data import Dataset
from glob import glob
from dataloaders.co3d_dataset import _seq_name_to_seed
import torch.nn.functional as F
from utils.masking import *
from torchvision import transforms
import pickle
from easydict import EasyDict
from tqdm import tqdm


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


class ScanNetDataset(Dataset):
    def __init__(self, dataset_root, data_list, image_height, image_width, n_frames_per_sequence,
                 rank=0, mode="train", min_interval=1, max_interval=8, masking_type="random", masking_params=None,
                 **kwargs):
        super(ScanNetDataset, self).__init__()
        self.dataset_root = dataset_root
        self.image_height = image_height
        self.image_width = image_width
        self.global_data_list = []  # 全局index，每个epoch不会改变
        self.n_frames_per_sequence = n_frames_per_sequence
        self.mode = mode
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.masking_type = masking_type
        self.masking_params = masking_params
        self.n_samples_per_subset = kwargs.get("n_samples_per_subset", -1)
        self.random_frame_sample = kwargs.get("random_frame_sample", True)
        self.sort_frames = kwargs.get("sort_frames", True)
        self.shuffle_val = kwargs.get("shuffle_val", False)
        self.fixed_mask_files = glob("./data/masks/fixed_mask_1000/*.png")
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.global_seq_dict = pickle.load(open("./data/scannet++/global_seq_dict.pkl", "rb"))
        self.total_num = 0
        with open(data_list, "r") as f:
            for line in tqdm(f, desc=f"loading global scannet++ {mode} sequences...", disable=rank != 0):
                class_name = line.strip()
                self.global_data_list.append(class_name)
                self.total_num += len(self.global_seq_dict[class_name])

        self.data_list_per_epoch, self.seq_list_per_epoch = [], []

        # for training, it should be set in the sampler for each epoch
        # for testing, directly loading from global_data_list
        if mode == 'val':
            self.scannet_reset_dataset(mode=mode, rank=rank)

    def __len__(self):
        return len(self.data_list_per_epoch)

    def scannet_reset_dataset(self, mode="train", rank=0, **kwargs):
        file_list = []
        sequence_index = []
        # random_seed = kwargs.get("random_seed", 123)

        class_names = self.global_data_list.copy()
        if self.n_samples_per_subset < 1:
            class_names = class_names[::int(1 / self.n_samples_per_subset)]
            n_samples_per_subset = 1
        else:
            n_samples_per_subset = self.n_samples_per_subset

        # n_samples_per_subset用于限制在scannet中总共取多少组frames
        # 每组由相同的interval组成，不同组之间interval不一样
        min_range = self.n_frames_per_sequence * self.min_interval
        max_range = self.n_frames_per_sequence * self.max_interval
        for class_name in class_names: # tqdm(class_names, desc=f"loading scannet++ {mode} per epoch...", disable=rank != 0):
            seq_indices = self.global_seq_dict[class_name].copy()  # 首先获取某个场景图片数量
            seq_indices = [f"{class_name}/dslr/undistorted_images/{s}" for s in seq_indices]
            splits = []
            st = 0
            while len(splits) == 0 or splits[-1][1] < len(seq_indices) - min_range:
                if mode == "train":
                    splits.append([st, st + random.randint(min_range, max_range)])
                else:
                    splits.append([st, st + int((min_range + max_range) / 2)])
                st = splits[-1][1] + 1

            splits[-1][1] = min(splits[-1][1], len(seq_indices))

            if mode == "train":
                random.shuffle(splits)
            splits = splits[:n_samples_per_subset]

            seq_num = 0
            for split in splits:
                seq_indices_ = seq_indices[split[0]:split[1]]
                if len(seq_indices_) < self.n_frames_per_sequence:
                    continue
                if self.n_frames_per_sequence > 0:
                    if self.random_frame_sample:
                        seq_idx_shuffled = random.sample(sorted(seq_indices_), len(seq_indices_))
                        new_idx = seq_idx_shuffled[:self.n_frames_per_sequence]
                    else:  # 等间隔采样
                        new_idx = seq_indices_[::len(seq_indices_) // self.n_frames_per_sequence][:self.n_frames_per_sequence]
                else:
                    new_idx = seq_indices_

                if kwargs.get("sort_frames", True):
                    new_idx.sort(key=lambda x: x.split("/")[-1])
                file_list.extend(new_idx)
                sequence_index.extend([seq_num] * len(new_idx))
                seq_num += 1

        self.data_list_per_epoch = file_list
        self.seq_list_per_epoch = sequence_index

    def get_frame_mask(self, index, fg_probability=None, class_name=None):
        if self.masking_type == "random":  # get masks as LDM-inpainting
            mask = get_inpainting_mask(512, 512)  # 训练固定512的mask是为了好决定mask超参数
            mask = F.interpolate(mask[None], (self.image_height, self.image_width), mode="nearest")[0]

        elif self.masking_type == "random_object_mask":
            mask = get_object_inpainting_mask(512, 512, fg_probability, **self.masking_params)
            mask = F.interpolate(mask[None], (self.image_height, self.image_width), mode="nearest")[0]

        elif self.masking_type == "none":
            mask = torch.zeros((1, self.image_height, self.image_width), dtype=torch.float32)

        elif self.masking_type == "object_mask":
            mask = get_object_enlarged_mask(fg_probability, self.masking_params.get("obj_dilate", 15))
            mask = cv2.resize(mask, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
            mask = torch.tensor(mask, dtype=torch.float32)[None]

        elif self.masking_type == "bbox":
            mask = get_object_bbox_mask(fg_probability, self.masking_params.get("obj_dilate", 15))
            mask = cv2.resize(mask, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
            mask = torch.tensor(mask, dtype=torch.float32)[None]

        elif self.masking_type == "wide_bbox":
            mask = get_object_bbox_mask(fg_probability)
            mask = cv2.resize(mask, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
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
            mask = cv2.resize(mask, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
            mask = torch.tensor(mask, dtype=torch.float32)[None]

        elif self.masking_type == "fixed":
            rdx = random.Random(_seq_name_to_seed(class_name)).randint(0, len(self.fixed_mask_files) - 1)
            mask_path = self.fixed_mask_files[(index + rdx) % len(self.fixed_mask_files)]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.image_width, self.image_height), interpolation=cv2.INTER_NEAREST)
            mask = torch.tensor(mask / 255, dtype=torch.float32)[None]
        else:
            raise NotImplementedError

        return mask

    def __getitem__(self, index):
        image = cv2.imread(f"{self.dataset_root}/{self.data_list_per_epoch[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_name = self.data_list_per_epoch[index].split("/")[-4]
        seq_name = self.seq_list_per_epoch[index]
        # filename = self.data_list_per_epoch[index].split("/")[-1]

        # hard code: mask为最短边512，先把image下采样到和mask一样大
        image = dynamic_crop(image, f"{class_name}/{seq_name}")

        if image.shape[0] > self.image_height:
            image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        else:
            image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        image = self.transform(image)  # value:[-1~1] shape:[3,h,w]
        mask = self.get_frame_mask(index, fg_probability=None, class_name=class_name)

        meta = {
            "image_rgb": image,
            "mask": mask,
            "sequence_category": str(class_name),
            "sequence_name": str(seq_name),
            "tag": "scannet++",
            "caption": ""
        }

        return EasyDict(meta)
