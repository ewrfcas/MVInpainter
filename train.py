import argparse
import logging
import math
import os
import random
import shutil

import accelerate
import cv2
import diffusers
import einops
import lpips
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version
from easydict import EasyDict
from mmflow.apis import init_model
from mmflow.datasets import visualize_flow
from omegaconf import OmegaConf
from peft import LoraConfig
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

from dataloaders.global_datasets import load_global_dataset
from dataloaders.global_sampler import GlobalConcatSampler
from diffusion_pipelines.pipeline_stable_diffusion_3d_inpaint import StableDiffusionInpaint3DPipeline
from models.animatediff.animatediff_unet_models import AnimateDiffModel
from models.prompt_clip import PromptCLIP
from models.unet_models import UNet3DConditionModel
from utils.model_setting import get_caption_model
from utils.others import get_clip_score, get_lpips_score, get_captions

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def slice_vae_encode(vae, image, sub_size):  # vae fails to encode large tensor directly, we need to slice it
    if (image.shape[-1] > 256 and image.shape[0] > sub_size) or (image.shape[0] > 192):
        slice_num = image.shape[0] // sub_size
        if image.shape[0] % sub_size != 0:
            slice_num += 1
        latents = []
        for i in range(slice_num):
            latents_ = vae.encode(image[i * sub_size:(i + 1) * sub_size]).latent_dist.sample()
            latents.append(latents_)
        latents = torch.cat(latents, dim=0)
    else:
        latents = vae.encode(image).latent_dist.sample()

    return latents


def get_caption(config, batch, init_image, caption_model, caption_processor, device):
    caption = batch.caption[::config.n_frames_per_sequence]
    no_caption_index = [j for j in range(len(caption)) if caption[j] == ""]  # "" means no caption from source data
    if caption_model is not None:
        image_caption = (init_image[::config.n_frames_per_sequence] + 1) / 2 * 255
        image_caption = image_caption.to(dtype=torch.uint8)
        if len(no_caption_index) > 0:
            image_caption = image_caption[no_caption_index]
            caption_texts = get_captions(image_caption, caption_model, caption_processor, device)
            for j_index, j in enumerate(no_caption_index):
                caption[j] = caption_texts[j_index]
        caption_texts = caption.copy()
        if hasattr(config, "caption_suffix") and config.caption_suffix is not None:
            for j in range(len(caption_texts)):
                if not caption_texts[j].endswith("."):
                    caption_texts[j] += ". "
                else:
                    caption_texts[j] += " "
                caption_texts[j] += config.caption_suffix
    else:
        caption_texts = []

    return caption_texts


def get_flow(flow_net, image, mask, n_frame):
    # image:[bf,3,h,w], mask:[bf,1,h,w]
    # tune down the resolution to 256 for flow estimation
    if image.shape[2] > 256:
        image = F.interpolate(image, size=(256, 256), mode="bicubic")
    if mask.shape[2] > 256:
        mask = F.interpolate(mask, size=(256, 256), mode="area")
        mask[mask > 0] = 1
    image = einops.rearrange(image, "(b f) c h w -> b f c h w", f=n_frame)
    mask = einops.rearrange(mask, "(b f) c h w -> b f c h w", f=n_frame)

    # get the inverse flow
    image0 = torch.flip(image[:, 1:], dims=[2])  # RGB to BGR
    image1 = torch.flip(image[:, :-1], dims=[2])

    flow_inputs = torch.cat([image0, image1], dim=2)  # [b,f-1,2c,h,w]
    flow_inputs = einops.rearrange(flow_inputs, "b f c h w -> (b f) c h w")

    with torch.no_grad(), torch.autocast(device_type=flow_inputs.device.type, enabled=True):
        flows = flow_net(flow_inputs)  # [b(f-1),h,w,2]
        flows = einops.rearrange(flows, "(b f) h w c -> b f c h w", f=n_frame - 1)  # [b,f-1,2,h,w]

    flows = flows * (1 - mask[:, 1:])
    flows = torch.cat([flows, mask[:, 1:]], dim=2)  # [b,f-1,3,h,w]
    flows = einops.rearrange(flows, "b f c h w -> b c f h w")  # [b,3,f-1,h,w]

    return flows


def log_validation(accelerator, config, args, vae, text_encoder, tokenizer, unet, weight_dtype, val_dataloader,
                   prompt_text, step, device, caption_processor=None, caption_model=None, **kwargs):
    if accelerator.is_main_process:
        logger.info(f"Validation log in step {step}")

    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", device=torch.device('cpu'), pretrained="laion2b_s32b_b79k")
    clip_model = clip_model.to(device).eval()
    loss_fn_alex = lpips.LPIPS(net='alex').to(device).eval()
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
        local_files_only=True
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    psnr_scores = []
    lpips_scores = []
    clip_scores = []
    show_images = []
    rdx = 0
    with torch.no_grad(), torch.autocast("cuda"):
        for batch in tqdm(val_dataloader, desc=f"Validation rank{accelerator.process_index}..."):
            init_image = torch.clamp((batch.image_rgb.to(device) + 1) / 2, 0, 1)
            caption_texts = get_caption(config, batch, init_image, caption_model, caption_processor, device)
            if len(caption_texts) == 0:
                final_text = prompt_text
            else:
                final_text = caption_texts

            mask = batch.mask.to(device)  # [B,1,H,W]
            mask[::config.n_frames_per_sequence] = 0  # the first view of every group should not be masked

            # dynamic nframe
            if args.dynamic_nframe:
                random_nframe = random.Random(rdx + accelerator.process_index).randint(args.low_nframe, args.high_nframe)
                init_image = einops.rearrange(init_image, "(b f) c h w -> b f c h w", f=config.n_frames_per_sequence)
                init_image = init_image[:, ::config.n_frames_per_sequence // random_nframe][:, :random_nframe]
                init_image = einops.rearrange(init_image, "b f2 c h w -> (b f2) c h w")
                mask = einops.rearrange(mask, "(b f) c h w -> b f c h w", f=config.n_frames_per_sequence)
                mask = mask[:, ::config.n_frames_per_sequence // random_nframe][:, :random_nframe]
                mask = einops.rearrange(mask, "b f2 c h w -> (b f2) c h w")
                rdx += 1
            else:
                random_nframe = config.n_frames_per_sequence

            if flow_net is not None:  # make init_image from 0~1 to -1~1
                flow_mask = mask.clone()  # mask dilation
                flow_mask = F.interpolate(flow_mask, scale_factor=0.5, mode="area")
                flow_mask[flow_mask > 0] = 1
                flow_mask = F.interpolate(flow_mask, scale_factor=2, mode="nearest")
                flows = get_flow(flow_net, init_image * 2 - 1, flow_mask, n_frame=random_nframe)
            else:
                flows = None

            preds = pipeline(final_text, init_image, mask, flows=flows,
                             height=init_image.shape[2], width=init_image.shape[3],
                             n_frames_per_sequence=random_nframe,
                             num_inference_steps=50, guidance_scale=args.val_cfg, output_type="np").images

            init_image = einops.rearrange(init_image, "b c h w -> b h w c").cpu().numpy()
            mask = einops.rearrange(mask, "f c h w -> f h w c").cpu().numpy()
            masked_image = init_image * (1 - mask)
            preds = preds * mask + (1 - mask) * init_image

            if len(show_images) < 16:  # only show 16 group of the first process
                if flows is not None:  # [1,3,f-1,h,w]
                    flows = einops.rearrange(flows[0, :2], "c f h w -> f h w c").cpu().numpy()
                    masked_flows = []
                    for fi in range(flows.shape[0]):
                        flow = visualize_flow(flows[fi]) / 255
                        if flow.shape[0] != init_image.shape[1]:
                            flow = cv2.resize(flow, (init_image.shape[2], init_image.shape[1]))
                        masked_flows.append(flow)
                    masked_flows = [np.zeros_like(masked_flows[0])] + masked_flows
                    show_image = np.concatenate([init_image, masked_image, masked_flows, preds], axis=1)
                else:
                    show_image = np.concatenate([init_image, masked_image, preds], axis=1)
                show_image = np.clip(np.concatenate([img for img in show_image], axis=1) * 255, 0, 255).astype(np.uint8)
                show_images.append(show_image)

            ref_image, gt_images = init_image[0], init_image[1:]
            preds = preds[1:]

            for i in range(preds.shape[0]):
                # we need to put the value to gpu for the sharing of accelerate
                psnr_ = peak_signal_noise_ratio(gt_images[i], preds[i], data_range=1.0)
                psnr_scores.append(psnr_)
                lpips_ = get_lpips_score(loss_fn_alex, gt_images[i], preds[i], device)
                lpips_scores.append(lpips_)
                clip_ = get_clip_score(clip_model, ref_image, preds[i], device)
                clip_scores.append(clip_)

    # unify all results
    psnr_score = torch.tensor(np.mean(psnr_scores), device=device, dtype=torch.float32)
    lpips_score = torch.tensor(np.mean(lpips_scores), device=device, dtype=torch.float32)
    clip_score = torch.tensor(np.mean(clip_scores), device=device, dtype=torch.float32)

    psnr_score = accelerator.gather(psnr_score).mean().item()
    lpips_score = accelerator.gather(lpips_score).mean().item()
    clip_score = accelerator.gather(clip_score).mean().item()

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.add_scalar("val/psnr", psnr_score, global_step=step)
                tracker.writer.add_scalar("val/lpips", lpips_score, global_step=step)
                tracker.writer.add_scalar("val/clip_score", clip_score, global_step=step)
                for j in range(len(show_images)):
                    if show_image[j].shape[0] > 1024:
                        show_images[j] = cv2.resize(show_images[j], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                    tracker.writer.add_images(f"val/gt_masked_pred_images{j}", show_images[j], step, dataformats="HWC")
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    del clip_model
    del loss_fn_alex
    torch.cuda.empty_cache()

    return lpips_score


def log_train(accelerator, config, args, vae, text_encoder, tokenizer, unet, weight_dtype, prompt_text,
              init_image, mask, step, caption_texts=[], random_nframe=12, **kwargs):
    logger.info(f"Train log in step {step}")

    init_image = init_image[:random_nframe]  # only show one group
    if len(caption_texts) > 0:
        final_text = caption_texts[:1]
    else:
        final_text = prompt_text
    init_image = torch.clamp((init_image + 1) / 2, 0, 1)
    mask = mask[:random_nframe]
    flows = kwargs.get("flows", None)
    if flows is not None:
        flows = flows[0:1]

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
        local_files_only=True
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    with torch.no_grad(), torch.autocast("cuda"):
        preds = pipeline(final_text, init_image, mask,
                         flows=flows, height=init_image.shape[2], width=init_image.shape[3],
                         n_frames_per_sequence=random_nframe,
                         num_inference_steps=50, guidance_scale=args.val_cfg,
                         output_type="np").images  # [f,h,w,c]

    init_image = einops.rearrange(init_image, "f c h w -> f h w c").cpu().numpy()
    mask = einops.rearrange(mask, "f c h w -> f h w c").cpu().numpy()
    masked_image = init_image * (1 - mask)
    preds = preds * mask + (1 - mask) * init_image

    # for flow visualization
    if flows is not None:  # [1,3,f-1,h,w]
        flows = einops.rearrange(flows[0, :2], "c f h w -> f h w c").cpu().numpy()
        masked_flows = []
        for fi in range(flows.shape[0]):
            flow = visualize_flow(flows[fi]) / 255
            if flow.shape[0] != init_image.shape[1]:
                flow = cv2.resize(flow, (init_image.shape[2], init_image.shape[1]))
            masked_flows.append(flow)
        masked_flows = [np.zeros_like(masked_flows[0])] + masked_flows
        show_image = np.concatenate([init_image, masked_image, masked_flows, preds], axis=1)
    else:
        show_image = np.concatenate([init_image, masked_image, preds], axis=1)
    show_image = np.clip(np.concatenate([img for img in show_image], axis=1) * 255, 0, 255).astype(np.uint8)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            if show_image.shape[0] > 1024:
                show_image = cv2.resize(show_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            tracker.writer.add_images("train/gt_masked_pred_images", show_image, step, dataformats="HWC")
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="mv-inpainting",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--train_log_interval", type=int, default=500)
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--no_val", action="store_true")
    parser.add_argument("--val_cfg", type=float, default=1.0)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--finetune_newdata", action="store_true")
    parser.add_argument("--resume_weight_only", action="store_true")
    parser.add_argument("--show_log", action="store_true", help="used for training in submission mode...")
    parser.add_argument("--dynamic_nframe", action="store_true", help="dynamic nframe training")
    parser.add_argument("--low_nframe", default=8, type=int)
    parser.add_argument("--high_nframe", default=24, type=int)
    parser.add_argument("--lr_rescale", default=1.0, type=float)
    parser.add_argument("--restart_global_step", default=0, type=int)
    parser.add_argument("--eval_at_first", action="store_true")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.resume_from_checkpoint is not None and args.resume_path is None:
        config = EasyDict(OmegaConf.load(os.path.join(args.output_dir, "config.yaml")))
        cfg = dict()
        for data_name in config.dataset_names:
            cfg[data_name] = EasyDict(OmegaConf.load(os.path.join(args.output_dir, f"{data_name}.yaml")))
    else:
        config = EasyDict(OmegaConf.load(args.config_file))
        cfg = None

    if args.dynamic_nframe:  # dynamic frame: fixed with the longest frame, then randomly clip during training
        old_nframe = config.n_frames_per_sequence
        config.old_nframe = old_nframe
        config.n_frames_per_sequence = args.high_nframe
        config.dynamic_nframe = [args.low_nframe, args.high_nframe]
        config.train_batch_size = int(config.train_batch_size // old_nframe * args.high_nframe)

    # Sanity checks
    assert config.dataset_names is not None and len(config.dataset_names) > 0

    return args, config, cfg


def main():
    args, config, cfg = parse_args()
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler",
                                                    rescale_betas_zero_snr=config.zerosnr,
                                                    beta_schedule=config.beta_schedule,
                                                    prediction_type=config.prediction_type,
                                                    local_files_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_name_or_path,
                                              subfolder="tokenizer", local_files_only=True)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    accelerator.print("Loading model weights...")
    # take text_encoder and vae away from parameter sharding across multi-gpu in ZeRO
    caption_processor, caption_model = None, None
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        vae = AutoencoderKL.from_pretrained(f"{config.pretrained_model_name_or_path}/vae", subfolder="vae", local_files_only=True)
        vae.requires_grad_(False)
        if hasattr(config, "caption_model") and config.caption_model is not None:
            caption_processor, caption_model = get_caption_model(config.caption_model)

    if hasattr(config.model_cfg, "add_model_cfg") and config.model_cfg.add_model_cfg is not None:
        add_model_cfg = dict(config.model_cfg.add_model_cfg)
    else:
        add_model_cfg = None

    if hasattr(config, "use_animatediff") and config.use_animatediff is True:
        inference_config = OmegaConf.load("configs/animatediff/inference-v3.yaml")
        unet = AnimateDiffModel.from_pretrained_2d(config.pretrained_model_name_or_path, subfolder="unet",
                                                   unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs),
                                                   rank=accelerator.process_index,
                                                   add_model_cfg=add_model_cfg)
    else:
        unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_model_name_or_path,
                                                       subfolder="unet",
                                                       rank=accelerator.process_index,
                                                       add_model_cfg=add_model_cfg)

    if config.model_cfg.lora_spatial:
        accelerator.print("Set lora to the origin diffusion modules.")
        lora_target_modules = set()
        for k, _ in unet.named_parameters():
            if "attentions" in k and "temp_attentions" not in k and "motion_modules" not in k and "flow_convs" not in k:
                if any([dm in k for dm in ["to_k", "to_q", "to_v", "to_out.0"]]):
                    lora_target_modules.add(k.replace(".weight", "").replace(".bias", ""))
        unet_lora_config = LoraConfig(
            r=config.model_cfg.lora_rank,
            lora_alpha=config.model_cfg.lora_rank,
            init_lora_weights="gaussian",
            target_modules=list(lora_target_modules),
        )
        unet.add_adapter(unet_lora_config)
        # we need to reset trainable params
        for n, p in unet.named_parameters():
            if any([dm in n for dm in ["transformer_in", "temp_convs", "temp_attentions", "motion_modules"]]):
                p.requires_grad = True
                # accelerator.print(n, p.requires_grad) # debug for lora and temporal

    if add_model_cfg.get("cross_view_pe", False):
        for n, p in unet.named_parameters():
            if any([dm in n for dm in ["view_abs_pe_layer", "pe_scale"]]):
                p.requires_grad = True
                accelerator.print(n, p.requires_grad)  # debug for lora and temporal

    if add_model_cfg is not None:
        enable_flow = add_model_cfg.get("enable_flow", False)
        if enable_flow:
            for n, p in unet.named_parameters():  # zero_ada_linear
                flow_trainable_keywords = ["flow_convs"]
                flow_cfg = add_model_cfg.get("flow_cfg", {"name": "default"})
                flow_combine = flow_cfg["name"].split("+")[-1]
                if flow_combine == "norm":
                    flow_trainable_keywords.append("time_emb_proj")
                    flow_trainable_keywords.append("zero_ada_linear")
                if any([dm in n for dm in flow_trainable_keywords]):
                    p.requires_grad = True
                    # accelerator.print(n, p.requires_grad)  # debug for flow

            with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
                flow_net = init_model(config="./check_points/mmflow/configs/raft/raft_8x2_100k_mixed_368x768.py",
                                      checkpoint="./check_points/mmflow/raft_8x2_100k_mixed_368x768.pth",
                                      device=accelerator.device)
        else:
            flow_net = None
    else:
        enable_flow = False
        flow_net = None

    if hasattr(config, "use_animatediff") and config.use_animatediff is True:
        animatediff_lora = torch.load("./check_points/AnimateDiff/v3_sd15_adapter_converted.ckpt", map_location="cpu")
        res = unet.load_state_dict(animatediff_lora, strict=False)
        assert len(res.unexpected_keys) == 0
        animatediff_motion = torch.load("./check_points/AnimateDiff/v3_sd15_mm.ckpt", map_location="cpu")
        res = unet.load_state_dict(animatediff_motion, strict=False)
        assert len(res.unexpected_keys) == 0

    # we need to put text_encoder to unet, because deepspeed only supports one module
    # need to take it after the lora
    if config.model_cfg.trainable_text_encoder:
        prompt_text = ""
        unet.text_encoder = PromptCLIP.from_pretrained(config.pretrained_model_name_or_path, subfolder="text_encoder",
                                                       local_files_only=True)
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
            text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="text_encoder", local_files_only=True)
            text_encoder.requires_grad_(False)
        prompt_text = args.prompt

    unet.train()

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.opt_cfg.scale_lr:
        config.learning_rate = (
                config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
        )

    # set trainable parameters
    trainable_params = []
    # if config.model_cfg.lora_spatial:
    params = [p for n, p in unet.named_parameters() if p.requires_grad and "text_encoder" not in n]
    trainable_params.append({'params': params, 'lr': config.opt_cfg.learning_rate})
    if config.model_cfg.trainable_text_encoder:
        trainable_params.append({'params': unet.text_encoder.text_model.embeddings.special_embedding.parameters(), 'lr': config.opt_cfg.prompt_lr})
    if config.get("full_model_trainable", False):  # let the whole unet trainable
        params_backbone = []
        for n, p in unet.named_parameters():
            if not p.requires_grad:
                p.requires_grad = True
                params_backbone.append(p)
        trainable_params.append({"params": params_backbone, "lr": config.opt_cfg.sd_lr})

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.opt_cfg.learning_rate,
        betas=(config.opt_cfg.adam_beta1, config.opt_cfg.adam_beta2),
        weight_decay=config.opt_cfg.adam_weight_decay,
        eps=config.opt_cfg.adam_epsilon,
    )

    # Get the datasets
    dynamic_sampling = config.get("dynamic_sampling", False)
    train_dataset, val_dataset, cfg = load_global_dataset(config, config.dataset_names,
                                                          no_training_filter=True if dynamic_sampling else False,
                                                          rank=accelerator.process_index, cfg=cfg,
                                                          img_size=args.img_size, dynamic_nframe=args.dynamic_nframe)

    if accelerator.is_main_process:
        OmegaConf.save(dict(config), os.path.join(args.output_dir, 'config.yaml'))
        for data_name in cfg:
            OmegaConf.save(dict(cfg[data_name]), os.path.join(args.output_dir, f'{data_name}.yaml'))

    sampler = GlobalConcatSampler(train_dataset,
                                  n_frames_per_sample=config.n_frames_per_sequence,
                                  shuffle=True,
                                  dynamic_sampling=dynamic_sampling,
                                  rank=accelerator.process_index,
                                  num_replicas=accelerator.num_processes,
                                  data_config=cfg)
    val_sampler = GlobalConcatSampler(val_dataset,
                                      n_frames_per_sample=config.n_frames_per_sequence,
                                      shuffle=False,
                                      rank=accelerator.process_index,
                                      num_replicas=accelerator.num_processes,
                                      data_config=cfg)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        sampler=sampler,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=val_sampler,
        batch_size=config.n_frames_per_sequence,
        num_workers=4,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        config.opt_cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.opt_cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return [total_num, trainable_num]

    param_info_vae = get_parameter_number(vae)
    accelerator.print(f'########## VAE, Total:{param_info_vae[0] / 1e6}M, Trainable:{param_info_vae[1] / 1e6}M ##################')
    param_info_txt = get_parameter_number(text_encoder)
    accelerator.print(f'########## Text Encoder, Total:{param_info_txt[0] / 1e6}M, Trainable:{param_info_txt[1] / 1e6}M ##################')
    param_info_unet = get_parameter_number(unet)
    if config.model_cfg.trainable_text_encoder:
        accelerator.print(f'########## Unet, Total:{(param_info_unet[0] - param_info_txt[0]) / 1e6}M, '
                          f'Trainable:{(param_info_unet[1] - param_info_txt[1]) / 1e6}M ##################')
    else:
        accelerator.print(f'########## Unet, Total:{(param_info_unet[0]) / 1e6}M, '
                          f'Trainable:{(param_info_unet[1]) / 1e6}M ##################')

    if args.resume_weight_only:
        resume_path = args.resume_path if args.resume_path is not None else args.output_dir
        dirs = os.listdir(resume_path)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        weights = torch.load(f"{resume_path}/{path}/pytorch_model/mp_rank_00_model_states.pt", map_location="cpu")
        accelerator.print(f"Load weights from {resume_path}/{path}/pytorch_model/mp_rank_00_model_states.pt")
        unet.load_state_dict(weights["module"], strict=True)

    # Prepare everything with our `accelerator`.
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = config.train_batch_size
    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)  # train_dataloader

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
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
    if caption_model is not None:
        caption_model.to(accelerator.device, dtype=weight_dtype)
    if enable_flow:
        flow_net.to(dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name)

    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    logger.info(f"  Training resolution = {args.img_size}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
            # resume_path = "/".join(path.split("/")[:-1])
            resume_path = args.resume_path
        else:
            # Get the most recent checkpoint
            resume_path = args.resume_path if args.resume_path is not None else args.output_dir
            dirs = os.listdir(resume_path)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")

            if not args.resume_weight_only:
                accelerator.load_state(os.path.join(resume_path, path))
                if args.restart_global_step != 0:
                    global_step = args.restart_global_step
                else:
                    global_step = int(path.split("-")[1])
                # reset learning_rate if needed
                if args.lr_rescale != 1.0:
                    for pg in optimizer.param_groups:
                        pg['initial_lr'] *= args.lr_rescale
                        pg['lr'] *= args.lr_rescale

                    for opt in lr_scheduler.optimizers:
                        for pg in opt.optimizer.param_groups:
                            pg['initial_lr'] *= args.lr_rescale
                            pg['lr'] *= args.lr_rescale

            if not args.finetune_newdata:
                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
            else:
                initial_global_step = 0
                first_epoch = 0
                global_step = 0

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    device = unet.device
    best_metric = 1000
    for epoch in range(first_epoch, config.num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                if type(batch) == dict:
                    batch = EasyDict(batch)
                init_image = batch.image_rgb.to(device)  # -1~1 tensor [B*F,3,H,W]
                caption_texts = get_caption(config, batch, init_image, caption_model, caption_processor, device)

                mask = batch.mask.to(device)  # [B,1,H,W]
                mask[::config.n_frames_per_sequence] = 0  # the first view of every group should not be masked

                # dynamic nframe
                if args.dynamic_nframe:
                    if step == 0 and epoch == first_epoch:  # the first frame is the longest
                        random_nframe = args.high_nframe
                    else:  # ensure all processes share the same nframe, or you need to wait for the slowest one
                        random_nframe = random.Random(global_step).randint(args.low_nframe, args.high_nframe)
                    init_image = einops.rearrange(init_image, "(b f) c h w -> b f c h w", f=config.n_frames_per_sequence)
                    init_image = init_image[:, ::config.n_frames_per_sequence // random_nframe][:, :random_nframe]
                    init_image = einops.rearrange(init_image, "b f2 c h w -> (b f2) c h w")
                    mask = einops.rearrange(mask, "(b f) c h w -> b f c h w", f=config.n_frames_per_sequence)
                    mask = mask[:, ::config.n_frames_per_sequence // random_nframe][:, :random_nframe]
                    mask = einops.rearrange(mask, "b f2 c h w -> (b f2) c h w")
                else:
                    random_nframe = config.n_frames_per_sequence

                origin_mask = mask.clone()
                masked_image = init_image * (mask < 0.5)

                if enable_flow:
                    flow_mask = mask.clone()  # mask dilation
                    flow_mask = F.interpolate(flow_mask, scale_factor=0.5, mode="area")
                    flow_mask[flow_mask > 0] = 1
                    flow_mask = F.interpolate(flow_mask, scale_factor=2, mode="nearest")
                    flows = get_flow(flow_net, init_image, flow_mask, n_frame=random_nframe)
                else:
                    flows = None

                latents = slice_vae_encode(vae, init_image.to(weight_dtype), sub_size=24 if init_image.shape[-1] > 256 else 192)
                latents = latents * vae.config.scaling_factor
                masked_image_latents = slice_vae_encode(vae, masked_image.to(weight_dtype), sub_size=24 if init_image.shape[-1] > 256 else 192)
                masked_image_latents = masked_image_latents * vae.config.scaling_factor

                mask = F.interpolate(mask, size=latents.shape[-2:], mode='nearest')

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if config.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
                if config.input_perturbation:
                    new_noise = noise + config.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                gsz = bsz // random_nframe
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (gsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if config.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(
                        einops.rearrange(latents, "(b f) c h w -> b f c h w", f=random_nframe),
                        einops.rearrange(noise, "(b f) c h w -> b f c h w", f=random_nframe),
                        timesteps
                    )
                    noisy_latents = einops.rearrange(noisy_latents, "b f c h w -> (b f) c h w")

                # Get the text embedding for conditioning; set prompt to "" in some probability.
                if random.random() < config.model_cfg.cfg_training_rate:
                    inputs_ids = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length",
                                           truncation=True, return_tensors="pt").input_ids.repeat(gsz, 1)
                else:
                    if len(caption_texts) == 0:
                        inputs_ids = tokenizer(prompt_text, max_length=tokenizer.model_max_length, padding="max_length",
                                               truncation=True, return_tensors="pt").input_ids.repeat(gsz, 1)
                    else:
                        inputs_ids = tokenizer(caption_texts, max_length=tokenizer.model_max_length, padding="max_length",
                                               truncation=True, return_tensors="pt").input_ids
                encoder_hidden_states = text_encoder(inputs_ids.to(device), return_dict=False)[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps.repeat_interleave(random_nframe, dim=0))
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # reshape to temporal inputs
                inputs = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)
                inputs = einops.rearrange(inputs, "(b f) c h w -> b c f h w", b=gsz, f=random_nframe)

                # Predict the noise residual and compute loss
                with torch.autocast(device_type=inputs.device.type, enabled=True, dtype=weight_dtype):
                    model_pred = unet(inputs, timesteps, encoder_hidden_states,
                                      return_dict=False, flows=flows)[0]  # [B,C,F,H,W]
                model_pred = einops.rearrange(model_pred, "b c f h w -> (b f) c h w")

                if config.opt_cfg.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.opt_cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % 20 == 0:
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                    train_loss = avg_loss.item() / config.gradient_accumulation_steps
                    accelerator.log({"train/loss": train_loss}, step=global_step)
                    accelerator.log({"train/lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                    if args.show_log:
                        logger.info(f"Loss: {train_loss}")

                if (global_step == 1 or global_step % args.train_log_interval == 0) and accelerator.is_main_process:
                    log_train(
                        accelerator=accelerator,
                        config=config,
                        args=args,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        unet=unet,
                        weight_dtype=weight_dtype,
                        prompt_text=prompt_text,
                        init_image=init_image,
                        mask=origin_mask,
                        step=global_step,
                        caption_texts=caption_texts,
                        flows=flows,
                        random_nframe=random_nframe
                    )

                if global_step % config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if args.eval_at_first or ((global_step == 1 or global_step % args.val_interval == 0) and not args.no_val):
                    res = log_validation(
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
                        step=global_step,
                        device=device,
                        caption_processor=caption_processor,
                        caption_model=caption_model,
                        flow_net=flow_net,
                    )
                    args.eval_at_first = False

                    if res <= best_metric:
                        best_metric = res
                        save_path = os.path.join(args.output_dir, "best")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved best state to {save_path}")

            logs = {"epoch": epoch + 1, "step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= config.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    main()
