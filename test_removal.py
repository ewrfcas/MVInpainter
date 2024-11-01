import argparse
import os
import random

import accelerate
import cv2
import einops
import lpips
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from diffusers import AutoencoderKL
from easydict import EasyDict
from mmflow.apis import init_model
from mmflow.datasets import visualize_flow
from omegaconf import OmegaConf
from peft import LoraConfig
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

from dataloaders.global_datasets import ConcatDataset
from dataloaders.global_datasets import load_global_dataset
from dataloaders.global_sampler import GlobalConcatSampler
from dataloaders.realworld_dataset import RealWorldDataset
from diffusion_pipelines.pipeline_stable_diffusion_3d_inpaint import StableDiffusionInpaint3DPipeline
from models.animatediff.animatediff_unet_models import AnimateDiffModel
from models.prompt_clip import PromptCLIP
from models.unet_models import UNet3DConditionModel
from train import get_flow
from utils.model_setting import get_caption_model
from utils.others import get_clip_score, get_lpips_score, get_captions
from diffusers import DDIMScheduler


def log_validation(accelerator, config, args, vae, text_encoder, tokenizer, unet, weight_dtype, val_dataloader,
                   prompt_text, device, caption_processor=None, caption_model=None, **kwargs):
    if accelerator.is_main_process:
        os.makedirs(f"outputs/{args.output_path}", exist_ok=True)
        accelerator.print(f"Validation start")

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
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    for l in range(args.repeat):
        random.seed(args.seed + l)
        np.random.seed(args.seed + l)
        torch.manual_seed(args.seed + l)
        torch.cuda.manual_seed_all(args.seed + l)

        psnr_scores = []
        lpips_scores = []
        clip_scores = []
        with torch.no_grad(), torch.autocast("cuda"):
            for batch in tqdm(val_dataloader, desc=f"Validation rank{accelerator.process_index}..."):

                if args.unsorted:
                    rdx = np.arange(config.n_frames_per_sequence - 1) + 1
                    np.random.shuffle(rdx)
                    batch.image_rgb[1:] = batch.image_rgb[rdx]
                    batch.filename[1:] = np.array(batch.filename)[rdx].tolist()

                init_image = torch.clamp((batch.image_rgb.to(device) + 1) / 2, 0, 1)
                if not hasattr(batch, "inpainted"):
                    inpainted = torch.zeros_like(batch.image_rgb)
                elif batch.inpainted.abs().sum() != 0:
                    inpainted = torch.clamp((batch.inpainted.to(device) + 1) / 2, 0, 1)
                else:
                    inpainted = batch.inpainted
                caption = batch.caption[::config.n_frames_per_sequence]
                if args.removal_prompt:
                    caption = ["nothing on the floor/ground/table"] * len(caption)
                    no_caption_index = []
                else:
                    no_caption_index = [j for j in range(len(caption)) if caption[j] == ""]  # "" means no caption from source data
                if caption_model is not None:
                    if inpainted.abs().sum() == 0:  # 没有参考图，用原图提取caption
                        image_caption = init_image[::config.n_frames_per_sequence] * 255
                    else:  # 提取参考图caption
                        image_caption = inpainted[::config.n_frames_per_sequence] * 255
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
                    final_text = caption_texts
                else:
                    final_text = prompt_text

                if args.save_text:
                    os.makedirs(f"outputs/{args.output_path}/caption", exist_ok=True)
                    with open(f"outputs/{args.output_path}/caption/{batch.sequence_category[0].split('*')[0]}.txt", 'w') as w:
                        w.write(final_text[0])

                print(batch.sequence_name[0], final_text)
                mask = batch.mask.to(device)  # [B,1,H,W]
                mask[::config.n_frames_per_sequence] = 0  # the first view of every group should not be masked

                if inpainted.abs().sum() != 0:  # 如果=0，则是用原图来inpainting
                    init_image[::config.n_frames_per_sequence] = inpainted[::config.n_frames_per_sequence]

                if flow_net is not None:  # 抽flow用原图or Inpaint后的图？
                    if args.flow_mask_dilate is None:
                        flow_mask = mask
                    else:
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

                preds = pipeline(final_text, init_image, mask, flows=flows,
                                 n_frames_per_sequence=config.n_frames_per_sequence,
                                 height=init_image.shape[2], width=init_image.shape[3],
                                 num_inference_steps=args.inference_step, cov_alpha=args.cov_alpha,
                                 guidance_scale=args.val_cfg, output_type="np").images

                init_image = einops.rearrange(init_image, "b c h w -> b h w c").cpu().numpy()
                mask = einops.rearrange(mask, "f c h w -> f h w c").cpu().numpy()
                masked_image = init_image * (1 - mask)
                preds = preds * mask + (1 - mask) * init_image

                if args.save_images:
                    save_results = preds[1:]
                    os.makedirs(f"outputs/{args.output_path}/{batch.sequence_category[0].split('*')[0]}", exist_ok=True)
                    for i in range(save_results.shape[0]):
                        save_result = (save_results[i, :, :, ::-1] * 255).astype(np.uint8)
                        cv2.imwrite(f"outputs/{args.output_path}/{batch.sequence_category[i + 1].split('*')[0]}/{batch.filename[i + 1].replace('.jpg', '.png')}", save_result)
                else:
                    if flows is not None:  # [1,3,f-1,h,w]
                        flows = einops.rearrange(flows[0, :2], "c f h w -> f h w c").cpu().numpy()
                        masked_flows = []
                        for fi in range(flows.shape[0]):
                            flow = visualize_flow(flows[fi]) / 255
                            masked_flows.append(flow)
                        masked_flows = [np.zeros_like(masked_flows[0])] + masked_flows
                        masked_flows =[cv2.resize(mf, (init_image.shape[2], init_image.shape[1])) for mf in masked_flows]
                        masked_flows = np.stack(masked_flows, axis=0)
                        show_image = np.concatenate([init_image, masked_image, masked_flows, preds], axis=1)
                    else:
                        show_image = np.concatenate([init_image, masked_image, preds], axis=1)
                    show_image = np.clip(np.concatenate([img for img in show_image], axis=1) * 255, 0, 255).astype(np.uint8)
                    if inpainted.abs().sum() == 0:
                        cv2.imwrite(f"outputs/{args.output_path}/{batch.sequence_category[0]}_{batch.sequence_name[0]}*origin_{l}.png", show_image[:, :, ::-1])
                    else:
                        cv2.imwrite(f"outputs/{args.output_path}/{batch.sequence_category[0]}_{batch.sequence_name[0]}_{l}.png", show_image[:, :, ::-1])

                if not args.save_images:
                    ref_image, gt_images = init_image[0], init_image[1:]
                    preds = preds[1:]

                    for i in range(preds.shape[0]):
                        psnr_ = torch.tensor(peak_signal_noise_ratio(gt_images[i], preds[i], data_range=1.0), dtype=torch.float32, device=device)
                        psnr_scores.append(accelerator.gather(psnr_).mean().item())
                        lpips_ = torch.tensor(get_lpips_score(loss_fn_alex, gt_images[i], preds[i], device), dtype=torch.float32, device=device)
                        lpips_scores.append(accelerator.gather(lpips_).mean().item())
                        clip_ = torch.tensor(get_clip_score(clip_model, ref_image, preds[i], device), dtype=torch.float32, device=device)
                        clip_scores.append(accelerator.gather(clip_).mean().item())

        if not args.save_images:
            # FIXME: dirty codes to align the result to multi-gpu evaluation
            if accelerator.num_processes == 1:
                psnr_scores = psnr_scores[:int(len(psnr_scores) // 4 * 4)]
                lpips_scores = lpips_scores[:int(len(lpips_scores) // 4 * 4)]
                clip_scores = clip_scores[:int(len(clip_scores) // 4 * 4)]

            accelerator.print(args.load_path)
            accelerator.print("PSNR", round(np.mean(psnr_scores), 3))
            accelerator.print("LPIPS", round(np.mean(lpips_scores), 3))
            accelerator.print("CLIP-Scores", round(np.mean(clip_scores), 3))


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
    parser.add_argument(
        "--val_mask_type",
        type=str,
        default="fix"
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
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
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--inference_step", type=int, default=50)
    parser.add_argument("--cov_alpha", type=float, default=0.0)
    parser.add_argument("--dataset_names", default=None, nargs="+", type=str, required=True)
    parser.add_argument("--dataset_root", default=None, type=str)
    parser.add_argument("--nframe", default=None, type=int)
    parser.add_argument("--mask_dilate", default=None, type=int)
    parser.add_argument("--removal_prompt", action="store_true")
    parser.add_argument("--reference_path", default="inpainted", type=str, help="path to save reference-guided images")
    parser.add_argument("--reference_num", default=-1, type=int, help="limit the reference num used for guidance")
    parser.add_argument("--reference_split", default=-1, type=int, help="split the origin images into N groups (should be <= nframe) to inpaint")
    parser.add_argument("--flow_after_inpaint", action="store_true")
    parser.add_argument("--save_images", action="store_true", help="save split image results")
    parser.add_argument("--save_text", action="store_true")
    parser.add_argument("--enable_bbox_mask", action="store_true")
    parser.add_argument("--caption_type", default=None, type=str)
    parser.add_argument("--flow_mask_dilate", default=None, type=float)
    parser.add_argument("--repeat", default=1, type=int)
    parser.add_argument("--unsorted", action="store_true")

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
    if args.unsorted:
        args.output_path += "_unsorted"

    cfg = dict()

    if "realworld" in args.dataset_names:
        cfg["realworld"] = {"dataset_root": args.dataset_root}
    else:
        for data_name in args.dataset_names:
            cfg[data_name] = EasyDict(OmegaConf.load(f"configs/datasets/{data_name}.yaml"))
            if args.mask_dilate is not None:
                cfg[data_name].val_masking_params.obj_dilate = args.mask_dilate

    # Sanity checks
    assert args.dataset_names is not None and len(args.dataset_names) > 0

    return args, config, cfg


if __name__ == '__main__':
    args, config, cfg = parse_args()

    accelerator = Accelerator()

    if "realworld" in cfg:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        val_dataset = RealWorldDataset(dataset_root=args.dataset_root,
                                       image_height=args.img_size, image_width=args.img_size,
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
    else:
        _, val_dataset, cfg = load_global_dataset(config, args.dataset_names,
                                                  rank=accelerator.process_index,
                                                  no_training_filter=True,
                                                  no_val_filter=True,
                                                  load_eval_pickle=True,
                                                  img_size=args.img_size, cfg=cfg)

        for i in range(len(val_dataset.datasets)):
            val_dataset.datasets[i].masking_type = args.val_mask_type
        val_dataset.cumulative_sizes = ConcatDataset.cumsum(val_dataset.datasets)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        val_sampler = GlobalConcatSampler(
            val_dataset,
            shuffle=False,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            n_frames_per_sample=config.n_frames_per_sequence,
            mode="val"
        )

        accelerator.print(f"Total Group num:{val_sampler.n_group_total}")

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


    caption_processor, caption_model = None, None
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        if hasattr(config, "caption_model") and config.caption_model is not None:
            caption_processor, caption_model = get_caption_model(config.caption_model)

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
    if caption_model is not None:
        caption_model.to(accelerator.device, dtype=weight_dtype)

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
        caption_processor=caption_processor,
        caption_model=caption_model,
        flow_net=flow_net
    )
