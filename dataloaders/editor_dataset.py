import collections
import os.path

from torch.utils.data import Dataset
from glob import glob
from utils.masking import *
from torchvision import transforms
from easydict import EasyDict
from tqdm import tqdm


def square_crop(image, mask):
    h = image.shape[0]
    w = image.shape[1]
    y_pos, x_pos = np.where(mask == 1)
    bbox_xyxy = [x_pos.min(), y_pos.min(), x_pos.max(), y_pos.max()]

    if h > w:
        bbox_center = ((bbox_xyxy[1] + bbox_xyxy[3]) // 2)
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


class EditorDataset(Dataset):
    def __init__(self, dataset_root, edited_index, image_height, image_width, n_frames_per_sequence,
                 sampling_interval=1.0, rank=0, square_crop=True, mask_dilate=None,
                 reference_path="inpainted", reference_num=-1, reference_split=-1,
                 caption_type=None,
                 **kwargs):
        super(EditorDataset, self).__init__()
        self.dataset_root = dataset_root
        self.image_height = image_height
        self.image_width = image_width
        self.global_data_list = collections.defaultdict(list)
        self.n_frames_per_sequence = n_frames_per_sequence
        self.square_crop = square_crop
        self.sampling_interval = sampling_interval
        self.reference_split = reference_split
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.total_num = 0
        self.global_data_list = collections.defaultdict(dict)
        self.scenes = glob(f"{dataset_root}/*")
        self.scenes = [s.split('/')[-1] for s in self.scenes]

        for scene in self.scenes:
            self.global_data_list[scene]["masks"] = sorted(glob(f"{dataset_root}/{scene}/warp_masks/*.png"))
            self.global_data_list[scene]["images"] = sorted(glob(f"{dataset_root}/{scene}/removal/*.png"))
            self.global_data_list[scene]["images"] = self.global_data_list[scene]["images"][:len(self.global_data_list[scene]["masks"])]
            self.global_data_list[scene]["inpainted"] = [f"{dataset_root}/{scene}/obj_bbox/{str(edited_index).zfill(4)}.png"]

        self.image_list, self.mask_list, self.inpainted_list, self.fname_list = self.realworld_reset_dataset(rank=rank)
        self.mask_dilate = mask_dilate
        self.enable_bbox_mask = kwargs.get("enable_bbox_mask", False)
        self.caption_type = caption_type

    def __len__(self):
        return len(self.image_list)

    def realworld_reset_dataset(self, rank=0, **kwargs):
        image_list = []
        mask_list = []
        inpainted_list = []
        fname_list = []

        class_names = list(self.global_data_list.keys())

        for class_name in tqdm(class_names, desc=f"loading realset...", disable=rank != 0):

            image_indices = self.global_data_list[f"{class_name}"]["images"].copy()
            mask_indices = self.global_data_list[f"{class_name}"]["masks"].copy()

            if len(mask_indices) == len(image_indices) + 1:
                mask_indices = mask_indices[1:]

            if len(image_indices) + 1 < self.n_frames_per_sequence:
                continue

            # assert len(image_indices) + 1 >= self.n_frames_per_sequence
            assert len(mask_indices) == len(image_indices)

            if self.sampling_interval < 1.0:
                split_num = max(int(len(image_indices) * self.sampling_interval), self.n_frames_per_sequence)
                image_indices = image_indices[:split_num]
                mask_indices = mask_indices[:split_num]
            new_image_idx = image_indices[::len(image_indices) // self.n_frames_per_sequence][:self.n_frames_per_sequence]
            new_mask_idx = mask_indices[::len(mask_indices) // self.n_frames_per_sequence][:self.n_frames_per_sequence]

            if kwargs.get("sort_frames", True):
                new_image_idx.sort(key=lambda x: x.split("/")[-1])
                new_mask_idx.sort(key=lambda x: x.split("/")[-1])

            for inpainted in self.global_data_list[f"{class_name}"]["inpainted"]:
                image_list.extend(new_image_idx)
                mask_list.extend(new_mask_idx)
                inpainted_list.extend([inpainted] * len(new_image_idx))

            if len(self.global_data_list[f"{class_name}"]["inpainted"]) == 0:
                image_list.extend(new_image_idx)
                mask_list.extend(new_mask_idx)
                inpainted_list.extend([None] * len(new_image_idx))  # 相当于测原图为reference

        return image_list, mask_list, inpainted_list, fname_list

    def __getitem__(self, index):
        image = cv2.imread(self.image_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # class_seq_name = "/".join(self.image_list[index].split("/")[-4:-2])
        class_seq_name = self.image_list[index].split("/")[-3]
        # mask:[0~1]
        mask = cv2.imread(self.mask_list[index])[:, :, -1] / 255

        image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        image = self.transform(image)  # value:[-1~1] shape:[3,h,w]
        mask = cv2.resize(mask, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
        mask[mask > 0] = 1
        if self.enable_bbox_mask:
            y_pos, x_pos = np.where(mask == 1)
            mask[y_pos.min():y_pos.max(), x_pos.min():x_pos.max()] = 1
        if self.mask_dilate is not None:
            kernel = np.ones((self.mask_dilate, self.mask_dilate), np.float32)
            mask = cv2.dilate(mask, kernel, iterations=1)
        mask = torch.tensor(mask, dtype=torch.float32)[None]  # [1,h,w]

        # load inpainted
        if self.inpainted_list[index] is not None:
            inpainted = cv2.imread(self.inpainted_list[index])
            inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
            inpainted = cv2.resize(inpainted, (self.image_width, self.image_height), interpolation=cv2.INTER_AREA)
            class_seq_name += f"*{self.inpainted_list[index].split('/')[-1].replace('.png', '')}"
            inpainted = self.transform(inpainted)
        else:
            inpainted = torch.zeros_like(image)

        if len(self.fname_list) > 0:
            class_seq_name += f"*{self.fname_list[index]}"

        sequence_category = class_seq_name.split("/")[0]
        if len(class_seq_name.split("/")) > 1:
            sequence_name = class_seq_name.split("/")[1]
        else:
            sequence_name = "seq"

        caption_path = "/".join(self.mask_list[index].split("/")[:-2])
        caption = ""

        if os.path.exists(f"{caption_path}/caption.txt"):
           captions = []
           with open(f"{caption_path}/caption.txt", "r") as f:
                for line in f:
                    captions.append(line.strip())
           if self.caption_type is None or self.caption_type == "first":
               caption = captions[0]
           else:
               caption = captions[-1]


        meta = {
            "image_rgb": image,
            "mask": mask,
            "sequence_category": sequence_category,
            "sequence_name": sequence_name,
            "tag": "realworld",
            "caption": caption.strip(),
            "inpainted": inpainted,
            "filename": self.image_list[index].split("/")[-1]
        }

        return EasyDict(meta)
