import bisect
import os.path
import pickle
from typing import (
    Iterable,
    List,
    TypeVar
)

from easydict import EasyDict
from omegaconf import OmegaConf
from torch.utils.data import Dataset, IterableDataset

from dataloaders.co3d_dataset import CO3Dv2Dataset
from dataloaders.dl3dv_dataset import DL3DVDataset
from dataloaders.mvimagenet_dataset import MVImageNetDataset
from dataloaders.real10k_dataset import Real10kDataset
from dataloaders.scannet_dataset import ScanNetDataset

T_co = TypeVar('T_co', covariant=True)


class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        return self.cumulative_sizes


def load_global_dataset(config, dataset_names, rank=0, load_eval_pickle=False, cfg=None, img_size=512, dynamic_nframe=False, **kwargs):
    train_datasets, val_datasets = [], []
    if cfg is None:  # using pre-defined cfg_dict
        cfg = dict()
    else:
        for dataset_name in cfg:
            cfg[dataset_name].n_frames_per_sequence = config.n_frames_per_sequence
            cfg[dataset_name].image_height = img_size
            cfg[dataset_name].image_width = img_size

    co3d_val_datasets = []
    for dataset_name in dataset_names:
        # for co3dv2, we use the official dataset codes for dataloading
        if dataset_name == "co3dv2" or dataset_name == "co3dv2_bg":
            origin_dataname = dataset_name
            dataset_name = "co3dv2"
            if dataset_name not in cfg:
                cfg[dataset_name] = EasyDict(OmegaConf.load(f"configs/datasets/co3dv2.yaml"))
            cfg[dataset_name].n_frames_per_sequence = config.n_frames_per_sequence
            cfg[dataset_name].image_height = img_size
            cfg[dataset_name].image_width = img_size
            if not load_eval_pickle:
                dataset_ = CO3Dv2Dataset(data_list=cfg[dataset_name].train_list, mode="train",
                                         masking_type=cfg[dataset_name].train_masking_type,
                                         masking_params=cfg[dataset_name].train_masking_params,
                                         n_samples_per_subset=cfg[dataset_name].train_n_samples_per_subset,
                                         random_frame_sample=cfg[dataset_name].train_random_frame_sample,
                                         rank=rank,
                                         **cfg[dataset_name])
                dataset_.global_tag = "co3dv2"
                train_datasets.append(dataset_)

            if hasattr(cfg[dataset_name], "random_frame_sample"):
                cfg[dataset_name].pop("random_frame_sample")
            dataset_ = CO3Dv2Dataset(data_list=cfg[dataset_name].val_list, mode="val",
                                     masking_type=cfg[dataset_name].val_masking_type,
                                     masking_params=cfg[dataset_name].val_masking_params,
                                     n_samples_per_subset=cfg[dataset_name].val_n_samples_per_subset,
                                     random_frame_sample=False if cfg[dataset_name].sort_frames else True,
                                     rank=rank,
                                     **cfg[dataset_name])
            dataset_.global_tag = "co3dv2"
            val_datasets.append(dataset_)

        elif dataset_name == "mvimagenet" or dataset_name == "mvimagenet_bg":
            origin_dataname = dataset_name
            dataset_name = "mvimagenet"
            if dataset_name not in cfg:
                cfg[dataset_name] = EasyDict(OmegaConf.load(f"configs/datasets/mvimagenet.yaml"))
            cfg[dataset_name].n_frames_per_sequence = config.n_frames_per_sequence
            cfg[dataset_name].image_height = img_size
            cfg[dataset_name].image_width = img_size
            if not load_eval_pickle:
                dataset_ = MVImageNetDataset(data_list=cfg[dataset_name].train_list, mode="train",
                                             masking_type=cfg[dataset_name].train_masking_type,
                                             masking_params=cfg[dataset_name].train_masking_params,
                                             n_samples_per_subset=cfg[dataset_name].train_n_samples_per_subset,
                                             random_frame_sample=cfg[dataset_name].train_random_frame_sample,
                                             rank=rank,
                                             **cfg[dataset_name])
                dataset_.global_tag = "mvimagenet"
                train_datasets.append(dataset_)

            if hasattr(cfg[dataset_name], "random_frame_sample"):
                cfg[dataset_name].pop("random_frame_sample")
            dataset_ = MVImageNetDataset(data_list=cfg[dataset_name].val_list, mode="val",
                                         masking_type=cfg[dataset_name].val_masking_type,
                                         masking_params=cfg[dataset_name].val_masking_params,
                                         n_samples_per_subset=cfg[dataset_name].val_n_samples_per_subset,
                                         random_frame_sample=False if cfg[dataset_name].sort_frames else True,
                                         rank=rank,
                                         **cfg[dataset_name])
            dataset_.global_tag = "mvimagenet"
            val_datasets.append(dataset_)

        elif dataset_name == "scannet++":
            if dataset_name not in cfg:
                cfg[dataset_name] = EasyDict(OmegaConf.load("configs/datasets/scannet++.yaml"))
            cfg[dataset_name].n_frames_per_sequence = config.n_frames_per_sequence
            cfg[dataset_name].image_height = img_size
            cfg[dataset_name].image_width = img_size
            if dynamic_nframe:  # dynamic would increase nframe from 12 to 24，so the sampling range would be too large, need some revise
                cfg[dataset_name].min_interval = max(
                    1, int(cfg[dataset_name].min_interval * (config.old_nframe / config.n_frames_per_sequence))
                )
                cfg[dataset_name].max_interval = max(
                    1, int(cfg[dataset_name].max_interval * (config.old_nframe / config.n_frames_per_sequence))
                )

            if not load_eval_pickle:
                dataset_ = ScanNetDataset(data_list=cfg[dataset_name].train_list, mode="train",
                                          masking_type=cfg[dataset_name].train_masking_type,
                                          masking_params=cfg[dataset_name].train_masking_params,
                                          n_samples_per_subset=cfg[dataset_name].train_n_samples_per_subset,
                                          random_frame_sample=cfg[dataset_name].train_random_frame_sample,
                                          rank=rank,
                                          **cfg[dataset_name])
                dataset_.global_tag = "scannet++"
                train_datasets.append(dataset_)

            if hasattr(cfg[dataset_name], "random_frame_sample"):
                cfg[dataset_name].pop("random_frame_sample")
            dataset_ = ScanNetDataset(data_list=cfg[dataset_name].val_list, mode="val",
                                      masking_type=cfg[dataset_name].val_masking_type,
                                      masking_params=cfg[dataset_name].val_masking_params,
                                      n_samples_per_subset=cfg[dataset_name].val_n_samples_per_subset,
                                      random_frame_sample=False,
                                      rank=rank,
                                      **cfg[dataset_name])
            dataset_.global_tag = "scannet++"
            val_datasets.append(dataset_)
        elif dataset_name == "real10k":
            if dataset_name not in cfg:
                cfg[dataset_name] = EasyDict(OmegaConf.load("configs/datasets/real10k.yaml"))
            cfg[dataset_name].n_frames_per_sequence = config.n_frames_per_sequence
            cfg[dataset_name].image_height = img_size
            cfg[dataset_name].image_width = img_size
            if dynamic_nframe:  # dynamic would increase nframe from 12 to 24，so the sampling range would be too large, need some revise
                cfg[dataset_name].min_interval = max(
                    1, int(cfg[dataset_name].min_interval * (config.old_nframe / config.n_frames_per_sequence))
                )
                cfg[dataset_name].max_interval = max(
                    1, int(cfg[dataset_name].max_interval * (config.old_nframe / config.n_frames_per_sequence))
                )

            if not load_eval_pickle:
                dataset_ = Real10kDataset(data_list=cfg[dataset_name].train_list, mode="train",
                                          masking_type=cfg[dataset_name].train_masking_type,
                                          masking_params=cfg[dataset_name].train_masking_params,
                                          n_samples_per_subset=cfg[dataset_name].train_n_samples_per_subset,
                                          random_frame_sample=cfg[dataset_name].train_random_frame_sample,
                                          rank=rank,
                                          **cfg[dataset_name])
                dataset_.global_tag = "real10k"
                train_datasets.append(dataset_)

            if hasattr(cfg[dataset_name], "random_frame_sample"):
                cfg[dataset_name].pop("random_frame_sample")
            dataset_ = Real10kDataset(data_list=cfg[dataset_name].val_list, mode="val",
                                      masking_type=cfg[dataset_name].val_masking_type,
                                      masking_params=cfg[dataset_name].val_masking_params,
                                      n_samples_per_subset=cfg[dataset_name].val_n_samples_per_subset,
                                      random_frame_sample=False,
                                      rank=rank,
                                      **cfg[dataset_name])
            dataset_.global_tag = "real10k"
            val_datasets.append(dataset_)
        elif dataset_name == "dl3dv":
            if dataset_name not in cfg:
                cfg[dataset_name] = EasyDict(OmegaConf.load("configs/datasets/dl3dv.yaml"))
            cfg[dataset_name].n_frames_per_sequence = config.n_frames_per_sequence
            cfg[dataset_name].image_height = img_size
            cfg[dataset_name].image_width = img_size
            if dynamic_nframe:  # dynamic would increase nframe from 12 to 24，so the sampling range would be too large, need some revise
                cfg[dataset_name].min_interval = max(
                    1, int(cfg[dataset_name].min_interval * (config.old_nframe / config.n_frames_per_sequence))
                )
                cfg[dataset_name].max_interval = max(
                    1, int(cfg[dataset_name].max_interval * (config.old_nframe / config.n_frames_per_sequence))
                )

            if not load_eval_pickle:
                dataset_ = DL3DVDataset(data_list=cfg[dataset_name].train_list, mode="train",
                                        masking_type=cfg[dataset_name].train_masking_type,
                                        masking_params=cfg[dataset_name].train_masking_params,
                                        n_samples_per_subset=cfg[dataset_name].train_n_samples_per_subset,
                                        random_frame_sample=cfg[dataset_name].train_random_frame_sample,
                                        rank=rank,
                                        **cfg[dataset_name])
                dataset_.global_tag = "dl3dv"
                train_datasets.append(dataset_)

            if hasattr(cfg[dataset_name], "random_frame_sample"):
                cfg[dataset_name].pop("random_frame_sample")
            dataset_ = DL3DVDataset(data_list=cfg[dataset_name].val_list, mode="val",
                                    masking_type=cfg[dataset_name].val_masking_type,
                                    masking_params=cfg[dataset_name].val_masking_params,
                                    n_samples_per_subset=cfg[dataset_name].val_n_samples_per_subset,
                                    random_frame_sample=False,
                                    rank=rank,
                                    **cfg[dataset_name])
            dataset_.global_tag = "dl3dv"
            val_datasets.append(dataset_)
        else:
            raise NotImplementedError("Not implemented dataset for {}".format(dataset_name))

    if load_eval_pickle and not os.path.exists("data/co3d_global_eval.pkl") and rank == 0:
        pickle.dump(co3d_val_datasets, open("data/co3d_global_eval.pkl", "wb"))

    if len(train_datasets) > 0:
        train_dataset = ConcatDataset(train_datasets)
    else:
        train_dataset = None
    val_datasets = ConcatDataset(val_datasets)

    return train_dataset, val_datasets, cfg
