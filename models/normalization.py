import einops
import torch.nn as nn
from torch import Tensor


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class AdaLayerNormZero(nn.LayerNorm):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, dim, base_dim=1280):
        super().__init__(dim)

        self.zero_ada_linear = zero_module(nn.Linear(base_dim, dim * 2))
        self.silu = nn.SiLU()

    def forward(self, input: Tensor, temb: Tensor) -> Tensor:
        # input:[b*f,h*w,c], temb:[b,c,f]
        x = super().forward(input) # [b*f,h*w,c]
        temb = einops.rearrange(temb, "b c f -> (b f) c") # [b*f,c]
        temb = einops.repeat(temb, "bf c -> bf hw c", hw=input.shape[1])
        emb = self.zero_ada_linear(self.silu(temb))
        scale, shift = emb.chunk(2, dim=-1)
        x = x * (1 + scale) + shift
        return x
