import torch
from diffusers.schedulers import PNDMScheduler
from transformers import AutoProcessor, BlipForConditionalGeneration

from schedulers.scheduling_ddim import DDIMScheduler


def get_weight_path(model_name):
    if model_name == 'SD2-inpainting':
        return "stabilityai/stable-diffusion-2-inpainting"
    elif model_name == "SD1.5-inpainting":
        return "runwayml/stable-diffusion-inpainting"
    else:
        raise NotImplementedError("Unknown model {}!".format(model_name))


def get_caption_model(model_name):
    if model_name == "blip":
        processor = AutoProcessor.from_pretrained("./check_points/huggingface/hub/models--Salesforce--blip-image-captioning-large/snapshots/2227ac38c9f16105cb0412e7cab4759978a8fd90",
                                                  local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained("./check_points/huggingface/hub/models--Salesforce--blip-image-captioning-large/snapshots/2227ac38c9f16105cb0412e7cab4759978a8fd90",
                                                             torch_dtype=torch.float16, local_files_only=True)
        return processor, model
    else:
        raise NotImplementedError


def get_scheduler(name):
    if name == "DDIM":
        # set by SD1.5
        return DDIMScheduler(beta_start=0.00085,
                             beta_end=0.02,
                             beta_schedule="scaled_linear",
                             clip_sample=False,
                             clip_sample_range=1.0,
                             dynamic_thresholding_ratio=0.995,
                             num_train_timesteps=1000,
                             prediction_type="epsilon",
                             rescale_betas_zero_snr=False,
                             sample_max_value=1.0,
                             set_alpha_to_one=False,
                             steps_offset=1,
                             thresholding=False,
                             timestep_spacing="leading")
    elif name == "PNDM":
        # set by SD2.0
        return PNDMScheduler(beta_start=0.00085,
                             beta_end=0.02,
                             beta_schedule="scaled_linear",
                             clip_sample=False,
                             num_train_timesteps=1000,
                             prediction_type="epsilon",
                             set_alpha_to_one=False,
                             skip_prk_steps=True,
                             steps_offset=1,
                             timestep_spacing="leading")
    else:  # use default sampler
        return None
