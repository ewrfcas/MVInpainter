# MVInpainter
[NeurIPS 2024] MVInpainter: Learning Multi-View Consistent Inpainting to Bridge 2D and 3D Editing

[[arXiv]](https://arxiv.org/pdf/2408.08000) [[Project Page]](https://ewrfcas.github.io/MVInpainter/)

### TODO List
- [x] Environment and dataset setup
- [x] Training codes
- [x] Inference models and pipeline
- [ ] Release better models for 512x512 resolution

## Preparation

### Setup repository and environment
```
git clone https://github.com/ewrfcas/MVInpainter.git
cd MVInpainter

conda create -n mvinpainter python=3.8
conda activate mvinpainter

pip install -r requirements.txt
mim install mmcv-full
pip install mmflow

# We need to replace the new decoder py of mmflow for faster flow estimation
cp ./check_points/mmflow/raft_decoder.py /usr/local/conda/envs/mvinpainter/lib/python3.8/site-packages/mmflow/models/decoders/
```

### Dataset preparation (training)
1. Downloading [Co3dv2](https://github.com/facebookresearch/co3d), [MVImgNet](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet) for MVInpainter-O.
Downloading [Real10k](https://google.github.io/realestate10k/download.html), [DL3DV](https://github.com/DL3DV-10K/Dataset), [Scannet++](https://kaldir.vc.in.tum.de/scannetpp) for MVInpainter-F.
2. Downloading information of indices, masking formats, and captions from [Link](https://huggingface.co/ewrfcas/MVInpainter/resolve/main/data.zip?download=true). Put them to `./data`. 
Note that we remove some dirty samples from aforementioned datasets. Since Co3dv2 data contains object masks but MVImgNet does not, we additionally provide complete [foreground masks](https://huggingface.co/ewrfcas/MVInpainter/resolve/main/mvimagenet_masks.zip?download=true) for MVImgNet through `CarveKit`. Please put the MVImgNet masks to `./data/mvimagenet/masks`.

### Pretrained weights
1. [RAFT weights](https://huggingface.co/ewrfcas/MVInpainter/resolve/main/raft_8x2_100k_mixed_368x768.pth?download=true) (put it to `./check_points/mmflow/`).
2. [SD1.5-inpainting](https://huggingface.co/ewrfcas/MVInpainter/resolve/main/models--runwayml--stable-diffusion-inpainting.zip?download=true)  (put it to `./check_points/`).
3. [AnimateDiff weights](https://huggingface.co/ewrfcas/MVInpainter/resolve/main/AnimateDiff.zip?download=true). We revise the key name for easier `peft` usages (put it to `./check_points/`).

## Training

Training with fixed nframe=12:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --mixed_precision="fp16" --num_processes=8 --num_machines 1 --main_process_port 29502 \
  --config_file configs/deepspeed/acc_zero2.yaml train.py \
  --config_file="configs/mvinpainter_{o,f}.yaml" \
  --output_dir="check_points/mvinpainter_{o,f}_256" \
  --train_log_interval=250 \
  --val_interval=2000 \
  --val_cfg=7.5 \
  --img_size=256
```

Finetuning with dynamic frames (8~24):
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --mixed_precision="fp16" --num_processes=8 --num_machines 1 --main_process_port 29502 \
  --config_file configs/deepspeed/acc_zero2.yaml train.py \
  --config_file="configs/mvinpainter_{o,f}.yaml" \
  --output_dir="check_points/mvinpainter_{o,f}_256" \
  --train_log_interval=250 \
  --val_interval=2000 \
  --val_cfg=7.5 \
  --img_size=256 \
  --resume_from_checkpoint="latest" \
  --dynamic_nframe \
  --low_nframe 8 \
  --high_nframe 24
```
Please use `mvinpainter_{o,f}_512.yaml` to train 512x512 models.

## Inference

### Model weights
1. [MVSInpainter-O](https://huggingface.co/ewrfcas/MVInpainter/resolve/main/mvinpainter_o_256.zip?download=true) (Novel view synthesis, put it to `./check_points/`).
2. [MVSInpainter-F](https://huggingface.co/ewrfcas/MVInpainter/resolve/main/mvinpainter_f_256.zip?download=true) (Removal, put it to `./check_points/`).

### Pipeline

1. Removing or synthesis foreground of the first view through 2D-inpainting. We recommend using [Fooocus-inpainting](https://github.com/lllyasviel/Fooocus) to accomplish this. Getting tracking masks through [Track-Anything](https://github.com/gaomingqi/Track-Anything). 
Some examples are provided in `./demo`.
```
- <folder>
    - images      # input images with foregrounds
    - inpainted   # inpainted result of the first view
    - masks       # masks for images
```
2. (Optional) removing foregrounds from all other views through `MVInpainter-F`:
```
CUDA_VISIBLE_DEVICES=0 python test_removal.py \
  --load_path="check_points/mvinpainter_f_256" \
  --dataset_root="./demo/removal" \
  --output_path="demo_removal" \
  --resume_from_checkpoint="best" \
  --val_cfg=1.0 \
  --img_size=256 \
  --sampling_interval=1.0 \
  --dataset_names realworld \
  --reference_path="inpainted" \
  --nframe=24 \
  --save_images # (whether to save samples respectively)
```
![removal](assets/kitchen_DSCF0676_removal_seq_0.jpg)

3. Achieving 3d bbox of the object generated from 2D-inpainting through `python draw_bbox.py`. Put the image `000x.png` and `000x.json` from `./bbox` to `obj_bbox` of the target folder.

![draw_bbox_demo](assets/draw_bbox.gif)

4. Mask adaption to achieve `warp_masks`. If the basic plane where the foreground placed on enjoys a small percentage of the whole image, please use methods like [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to get `plane_masks`.
```
CUDA_VISIBLE_DEVICES=0 python mask_adaption.py --input_path="demo/nvs/kitchen" --edited_index=0
``` 
You can also use `--no_irregular_mask` to disable irregular mask for more precise warped masks.

![warp_bbox](assets/0000_bbox.jpg)

Make sure the final folder looks like:
```
- <folder>
    - obj_bbox    # inpainted 2d images with new foreground and bbox json
    - removal     # images without foregrounds
    - warp_masks  # masks from adaption for the removal folder
    - plane_masks # (optional, only for mask_adaption) masks of basic plane where the foreground is placed on
```

5. Run `MVInpainter-O` for novel view synthesis:
```
CUDA_VISIBLE_DEVICES=0 python test_nvs.py \
  --load_path="check_points/mvinpainter_o_256" \
  --dataset_root="./demo/nvs" \
  --output_path="demo_nvs" \
  --edited_index=0 \
  --resume_from_checkpoint="best" \
  --val_cfg=7.5 \
  --img_height=256 \
  --img_width=256 \
  --sampling_interval=1.0 \
  --nframe=24 \
  --prompt="a red apple with circle and round shape on the table." \
  --limit_frame=24
```
![nvs_result](assets/kitchen_0000_seq_0.jpg)

6. 3D reconstruction: See [Dust3R](https://github.com/naver/dust3r), [MVSFormer++](https://github.com/maybeLx/MVSFormerPlusPlus), and [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) for more details.


## Cite
If you found our project helpful, please consider citing:

```
@article{cao2024mvinpainter,
  title={MVInpainter: Learning Multi-View Consistent Inpainting to Bridge 2D and 3D Editing},
  author={Cao, Chenjie and Yu, Chaohui and Fu, Yanwei and Wang, Fan and Xue, Xiangyang},
  journal={arXiv preprint arXiv:2408.08000},
  year={2024}
}
```