import torch
import torch.nn as nn

from models.vae.distributions import DiagonalGaussianDistribution
from models.vae.model import Encoder
from models.vae.my_decoder_large2 import Decoder


class AutoencoderKL(nn.Module):
    def __init__(self,
                 ddconfig,
                 embed_dim=4,
                 image_key="image"):
        super().__init__()
        self.config = ddconfig
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.dtype = torch.float16
        self.device = "cuda"

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, image, mask):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, image, mask)
        return dec

    def forward(self, input, sample_posterior=True):
        input_image = input[0]
        input_mask = input[1]
        posterior = self.encode(input_image)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, input_image, input_mask)
        return dec, posterior

    def forward_aug(self, image, mask, origin_image, sample_posterior=True):
        input_image = image  # maybe noised
        input_mask = mask
        posterior = self.encode(input_image)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, origin_image, input_mask)
        return dec, posterior