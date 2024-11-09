import argparse
import os
import random

import accelerate
import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from diffusers import AutoencoderKL
from easydict import EasyDict
from mmflow.apis import init_model
from mmflow.datasets import visualize_flow
from omegaconf import OmegaConf
from peft import LoraConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

from dataloaders.editor_dataset import EditorDataset
from dataloaders.global_datasets import ConcatDataset
from dataloaders.global_sampler import GlobalConcatSampler
from diffusion_pipelines.pipeline_stable_diffusion_3d_inpaint import StableDiffusionInpaint3DPipeline
from models.animatediff.animatediff_unet_models import AnimateDiffModel
from models.prompt_clip import PromptCLIP
from models.unet_models import UNet3DConditionModel
from train import get_flow
from diffusers import DDIMScheduler


def mask_crop(image, mask, flow=None):
    b, c, h, w = image.shape
    bboxes = []
    min_hw = min(h, w)
    image_crop = torch.zeros((b, c, min_hw, min_hw), dtype=image.dtype, device=image.device)
    mask_crop = torch.zeros((b, 1, min_hw, min_hw), dtype=mask.dtype, device=mask.device)
    flow_crop = torch.zeros((1, 3, b - 1, min_hw, min_hw), dtype=flow.dtype, device=flow.device)

    for i in range(b):
        if i == 0:
            y_pos, x_pos = torch.where(mask[i + 1, 0] == 1)
        else:
            y_pos, x_pos = torch.where(mask[i, 0] == 1)
        bbox_xyxy = [x_pos.min().item(), y_pos.min().item(), x_pos.max().item(), y_pos.max().item()]

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
            bbox_center = ((bbox_xyxy[0] + bbox_xyxy[2]) // 2)
            x0 = bbox_center - h // 2
            x1 = bbox_center + h // 2
            if x0 < 0:
                x1 -= x0
                x0 = 0
            elif x1 > w:
                x0 -= (x1 - w)
                x1 = w
            clamp_bbox_xyxy = [x0, 0, x1, h]

        image_crop[i] = image[i, :, clamp_bbox_xyxy[1]:clamp_bbox_xyxy[3], clamp_bbox_xyxy[0]:clamp_bbox_xyxy[2]]
        mask_crop[i] = mask[i, :, clamp_bbox_xyxy[1]:clamp_bbox_xyxy[3], clamp_bbox_xyxy[0]:clamp_bbox_xyxy[2]]
        if i > 0:
            flow_crop[:, :, i - 1] = flow[:, :, i - 1, clamp_bbox_xyxy[1]:clamp_bbox_xyxy[3], clamp_bbox_xyxy[0]:clamp_bbox_xyxy[2]]

        bboxes.append(clamp_bbox_xyxy)

    return image_crop, mask_crop, flow_crop, bboxes


def log_validation(accelerator, config, args, vae, text_encoder, tokenizer, unet, weight_dtype, val_dataloader,
                   prompt_text, device, **kwargs):
    if accelerator.is_main_process:
        os.makedirs(f"outputs/{args.output_path}", exist_ok=True)
        accelerator.print(f"Validation start")

    flow_net = kwargs.get("flow_net", None)

    scheduler = DDIMScheduler.from_pretrained(config.pretrained_model_name_or_path,
                                              subfolder="scheduler", local_files_only=True,
                                              rescale_betas_zero_snr=config.zerosnr,
                                              prediction_type=config.prediction_type,
                                              beta_schedule=config.beta_schedule)

    pipeline = StableDiffusionInpaint3DPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        scheduler=scheduler,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    # pipeline.set_progress_bar_config(disable=True)

    for l in range(args.repeat):
        random.seed(args.seed + l)
        np.random.seed(args.seed + l)
        torch.manual_seed(args.seed + l)
        torch.cuda.manual_seed_all(args.seed + l)
        with torch.no_grad(), torch.autocast("cuda"):
            for batch in tqdm(val_dataloader, desc=f"Validation rank{accelerator.process_index}..."):
                if "is_pad" in batch:
                    valid_num = 0
                    for pad in batch.is_pad:
                        if pad.item() is False:
                            valid_num += 1
                    batch.image_rgb = batch.image_rgb[:valid_num]
                    batch.mask = batch.mask[:valid_num]
                    batch.inpainted = batch.inpainted[:valid_num]
                    args.limit_frame = valid_num
                    print("valid_num", valid_num)

                init_image = torch.clamp((batch.image_rgb.to(device) + 1) / 2, 0, 1)
                if not hasattr(batch, "inpainted"):
                    inpainted = torch.zeros_like(batch.image_rgb)
                elif batch.inpainted.abs().sum() != 0:
                    inpainted = torch.clamp((batch.inpainted.to(device) + 1) / 2, 0, 1)
                else:
                    inpainted = batch.inpainted

                final_text = prompt_text
                if hasattr(config, "caption_suffix") and config.caption_suffix is not None:
                    if not final_text.endswith("."):
                        final_text += ". "
                    else:
                        final_text += " "
                    final_text += config.caption_suffix

                if args.save_text:
                    os.makedirs(f"outputs/{args.output_path}/caption", exist_ok=True)
                    with open(f"outputs/{args.output_path}/caption/{batch.sequence_category[0].split('*')[0]}.txt", 'w') as w:
                        w.write(final_text[0])

                if args.limit_frame is not None:
                    init_image = init_image[:args.limit_frame]
                    batch.image_rgb = batch.image_rgb[:args.limit_frame]
                    batch.mask = batch.mask[:args.limit_frame]
                    config.n_frames_per_sequence = args.limit_frame

                print(batch.sequence_name[0], final_text)
                mask = batch.mask.to(device)  # [B,1,H,W]
                mask[::config.n_frames_per_sequence] = 0  # the first view of every group should not be masked

                if inpainted.abs().sum() != 0:  # 如果=0，则是用原图来inpainting
                    init_image[::config.n_frames_per_sequence] = inpainted[::config.n_frames_per_sequence]

                if flow_net is not None:
                    if args.flow_mask_dilate is None:
                        flow_mask = mask
                    else:
                        import torch.nn.functional as F
                        flow_mask = mask.clone()
                        flow_mask = F.interpolate(flow_mask, scale_factor=args.flow_mask_dilate, mode="area")
                        flow_mask[flow_mask > 0] = 1
                        flow_mask = F.interpolate(flow_mask, scale_factor=1 / args.flow_mask_dilate, mode="nearest")

                    if args.flow_after_inpaint:  # 因为有些情况我们并没有原图view0对应的GT
                        flows = get_flow(flow_net, init_image, flow_mask, n_frame=config.n_frames_per_sequence)
                    else:
                        flows = get_flow(flow_net, batch.image_rgb.to(device), flow_mask, n_frame=config.n_frames_per_sequence)
                else:
                    flows = None

                if init_image.shape[2] != init_image.shape[3]:  # mask crop
                    init_image_, mask_, flows_, bboxes = mask_crop(init_image.clone(), mask.clone(), flows.clone())
                else:
                    init_image_, mask_, flows_, bboxes = init_image, mask, flows, []

                preds = pipeline(final_text, init_image_, mask_, flows=flows_,
                                 n_frames_per_sequence=config.n_frames_per_sequence,
                                 height=init_image_.shape[2], width=init_image_.shape[3],
                                 num_inference_steps=args.inference_step, cov_alpha=args.cov_alpha,
                                 guidance_scale=args.val_cfg, output_type="np").images

                init_image = einops.rearrange(init_image, "b c h w -> b h w c").cpu().numpy()
                if len(bboxes) > 0:
                    preds_copy = preds.copy()
                    preds = init_image.copy()
                    for i_ in range(1, len(bboxes)):
                        preds[i_, bboxes[i_][1]:bboxes[i_][3], bboxes[i_][0]:bboxes[i_][2]] = preds_copy[i_]
                mask = einops.rearrange(mask, "f c h w -> f h w c").cpu().numpy()
                masked_image = init_image * (1 - mask)
                preds = preds * mask + (1 - mask) * init_image

                if args.save_images:
                    save_results = preds[0:]
                    os.makedirs(f"outputs/{args.output_path}/{batch.sequence_category[0].split('*')[0]}", exist_ok=True)
                    for i in range(save_results.shape[0]):
                        save_result = (save_results[i, :, :, ::-1] * 255).astype(np.uint8)
                        cv2.imwrite(f"outputs/{args.output_path}/{batch.sequence_category[i].split('*')[0]}/{batch.filename[i].replace('.jpg', '.png')}", save_result)
                else:
                    if flows is not None:  # [1,3,f-1,h,w]
                        flows = einops.rearrange(flows[0, :2], "c f h w -> f h w c").cpu().numpy()
                        masked_flows = []
                        for fi in range(flows.shape[0]):
                            flow = visualize_flow(flows[fi]) / 255
                            masked_flows.append(flow)
                        masked_flows = [np.zeros_like(masked_flows[0])] + masked_flows
                        masked_flows = [cv2.resize(mf, (init_image.shape[2], init_image.shape[1])) for mf in masked_flows]
                        masked_flows = np.stack(masked_flows, axis=0)
                        show_image = np.concatenate([init_image, masked_image, masked_flows, preds], axis=1)
                    else:
                        show_image = np.concatenate([init_image, masked_image, preds], axis=1)
                    show_image = np.clip(np.concatenate([img for img in show_image], axis=1) * 255, 0, 255).astype(np.uint8)
                    cv2.imwrite(f"outputs/{args.output_path}/{batch.sequence_category[0]}_{batch.sequence_name[0]}_{l}.png", show_image[:, :, ::-1])


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_co3dv2"
    )
    parser.add_argument("--seed", type=int, default=1235, help="A seed for reproducible training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
    )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--val_cfg", type=float, default=1.0)
    parser.add_argument("--sampling_interval", type=float, default=1.0, help="sample from frame%, (0,1]")
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--img_width", type=int, default=512)
    parser.add_argument("--inference_step", type=int, default=50)
    parser.add_argument("--cov_alpha", type=float, default=0.0)
    parser.add_argument("--dataset_root", default=None, type=str)
    parser.add_argument("--edited_index", type=int, default=0)
    parser.add_argument("--nframe", default=None, type=int)
    parser.add_argument("--mask_dilate", default=None, type=int)
    parser.add_argument("--removal_prompt", action="store_true")
    parser.add_argument("--reference_path", default="inpainted", type=str, help="path to save reference-guided images")
    parser.add_argument("--reference_num", default=-1, type=int, help="limit the reference num used for guidance")
    parser.add_argument("--reference_split", default=-1, type=int, help="split the origin images into N groups (should be <= nframe) to inpaint")
    parser.add_argument("--flow_after_inpaint", action="store_true")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_text", action="store_true")
    parser.add_argument("--enable_bbox_mask", action="store_true")
    parser.add_argument("--caption_type", default=None, type=str)
    parser.add_argument("--flow_mask_dilate", default=None, type=float)
    parser.add_argument("--limit_frame", default=None, type=int)
    parser.add_argument("--repeat", default=1, type=int)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    args.config_file = os.path.join(args.load_path, "config.yaml")
    args.output_path = args.output_path + f"_interval{args.sampling_interval}"
    config = EasyDict(OmegaConf.load(args.config_file))

    if args.nframe is not None and config.n_frames_per_sequence != args.nframe:
        args.output_path = args.output_path + f"_nframe{args.nframe}"
        config.n_frames_per_sequence = args.nframe
    if args.mask_dilate is not None:
        args.output_path = args.output_path + f"_enlarge{args.mask_dilate}"
    args.output_path += f"_ckpt-{args.resume_from_checkpoint}"
    args.output_path += f"_cfg{args.val_cfg}"
    if args.caption_type is not None:
        args.output_path += f"_caption-{args.caption_type}"

    cfg = dict()

    return args, config, cfg


if __name__ == '__main__':
    args, config, cfg = parse_args()

    accelerator = Accelerator()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    val_dataset = EditorDataset(dataset_root=args.dataset_root,
                                edited_index=args.edited_index,
                                image_height=args.img_height, image_width=args.img_width,
                                n_frames_per_sequence=config.n_frames_per_sequence,
                                sampling_interval=args.sampling_interval,
                                rank=accelerator.process_index,
                                mask_dilate=args.mask_dilate,
                                reference_path=args.reference_path,
                                reference_num=args.reference_num,
                                reference_split=args.reference_split,
                                enable_bbox_mask=args.enable_bbox_mask,
                                caption_type=args.caption_type)

    accelerator.print(f"Total image num:{len(val_dataset)},"
                      f" groun num:{len(val_dataset) / config.n_frames_per_sequence}")
    val_dataset = ConcatDataset([val_dataset])
    val_sampler = GlobalConcatSampler(
        val_dataset,
        shuffle=False,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        n_frames_per_sample=config.n_frames_per_sequence,
        mode="val",
        seed=args.seed
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=val_sampler,
        batch_size=config.n_frames_per_sequence,
        num_workers=4,
    )


    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]


    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        print('load vae')
        vae = AutoencoderKL.from_pretrained(f"{config.pretrained_model_name_or_path}/vae", subfolder="vae")
        print('load tokenizer')
        tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_name_or_path, subfolder="tokenizer")

    if hasattr(config.model_cfg, "add_model_cfg") and config.model_cfg.add_model_cfg is not None:
        add_model_cfg = dict(config.model_cfg.add_model_cfg)
    else:
        add_model_cfg = None

    if hasattr(config, "use_animatediff") and config.use_animatediff is True:
        inference_config = OmegaConf.load("configs/animatediff/inference-v3.yaml")
        unet = AnimateDiffModel.from_pretrained_2d(config.pretrained_model_name_or_path,
                                                   subfolder="unet",
                                                   unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs),
                                                   add_model_cfg=add_model_cfg)
    else:
        unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_model_name_or_path,
                                                       subfolder="unet",
                                                       rank=accelerator.process_index,
                                                       add_model_cfg=add_model_cfg,
                                                       do_not_load_weights=True)

    if config.model_cfg.trainable_text_encoder:
        prompt_text = ""
        unet.text_encoder = PromptCLIP.from_pretrained(config.pretrained_model_name_or_path, subfolder="text_encoder")
        special_tokens_dict = {'additional_special_tokens': []}

        prompt_length = config.model_cfg.prompt_length
        for i in range(prompt_length):
            special_tokens_dict['additional_special_tokens'].append(f"<special-token{i}>")
            prompt_text += f"<special-token{i}> "
        prompt_text = prompt_text.strip()
        tokenizer.add_special_tokens(special_tokens_dict)
        # re-init special embeddings
        init_sp_embedding = unet.text_encoder.text_model.embeddings.token_embedding.weight[-1:].clone().repeat(prompt_length, 1)
        unet.text_encoder.text_model.embeddings.special_embedding = nn.Embedding(prompt_length,
                                                                                 embedding_dim=unet.text_encoder.text_model.embeddings.special_embedding.embedding_dim,
                                                                                 _weight=init_sp_embedding)
        unet.text_encoder.requires_grad_(False)
        unet.text_encoder.text_model.embeddings.special_embedding.requires_grad_(True)
        text_encoder = unet.text_encoder  # mapping
    else:
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="text_encoder")
            text_encoder.requires_grad_(False)
        prompt_text = args.prompt

    if config.model_cfg.lora_spatial:
        print("Set lora to the origin diffusion modules.")
        lora_target_modules = set()
        for k, _ in unet.named_parameters():
            if "attentions" in k and "temp_attentions" not in k:
                if any([dm in k for dm in ["to_k", "to_q", "to_v", "to_out.0"]]):
                    lora_target_modules.add(k.replace(".weight", "").replace(".bias", ""))
        unet_lora_config = LoraConfig(
            r=config.model_cfg.lora_rank,
            lora_alpha=config.model_cfg.lora_rank,
            init_lora_weights="gaussian",
            target_modules=list(lora_target_modules),
        )
        unet.add_adapter(unet_lora_config)

    if add_model_cfg is not None:
        enable_flow = add_model_cfg.get("enable_flow", False)

        if enable_flow:
            with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
                flow_net = init_model(config="./check_points/mmflow/configs/raft/raft_8x2_100k_mixed_368x768.py",
                                      checkpoint="./check_points/mmflow/raft_8x2_100k_mixed_368x768.pth",
                                      device=accelerator.device)
        else:
            flow_net = None
    else:
        enable_flow = False
        flow_net = None

    device = accelerator.process_index

    # Potentially load in the weights and states from a previous save
    # Get the most recent checkpoint
    if args.resume_from_checkpoint == "last":
        dirs = os.listdir(args.load_path)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
    else:
        path = args.resume_from_checkpoint

    weights = torch.load(f"{args.load_path}/{path}/pytorch_model/mp_rank_00_model_states.pt", map_location="cpu")
    accelerator.print(f"Load weights from {args.load_path}/{path}/pytorch_model/mp_rank_00_model_states.pt")
    unet.load_state_dict(weights['module'], strict=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    if not config.model_cfg.trainable_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    log_validation(
        accelerator=accelerator,
        config=config,
        args=args,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        weight_dtype=weight_dtype,
        val_dataloader=val_dataloader,
        prompt_text=prompt_text,
        device=device,
        flow_net=flow_net
    )
