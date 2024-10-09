import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionalEncodingNorm(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=12):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        self.pe_dict = dict()

    def reset_pe(self, new_len, device):
        position = torch.arange(new_len).unsqueeze(1).float() * self.max_len / new_len  # normalize to 0~max_len
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(1, self.max_len, self.d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return pe.to(device)  # [1,len,d]

    def forward(self, x):
        # x:[BHW,F,C]
        len = x.shape[1]
        if f"{len}" in self.pe_dict:
            pe = self.pe_dict[f"{len}"]
        else:
            pe = self.reset_pe(len, x.device)
        x = x + pe
        return self.dropout(x)


class PositionEncodingSine2DNorm(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(16, 16)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()
        assert len(max_shape) == 2
        self.d_model = d_model
        self.max_shape = max_shape
        self.pe_dict = dict()

    def reset_pe(self, new_shape, device):
        (H, W) = new_shape
        pe = torch.zeros((self.d_model, H, W))
        y_position = torch.ones((H, W)).cumsum(0).float().unsqueeze(0) * self.max_shape[0] / H
        x_position = torch.ones((H, W)).cumsum(1).float().unsqueeze(0) * self.max_shape[1] / W

        div_term = torch.exp(torch.arange(0, self.d_model // 2, 2).float() * (-math.log(10000.0) / (self.d_model // 2)))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        return pe.unsqueeze(0).to(device)

    def forward_pe(self, H, W, device):
        if f"{H}-{W}" in self.pe_dict:  # if the cache has this PE weights, use it
            pe = self.pe_dict[f"{H}-{W}"]
        else:  # or re-generate new PE weights for H-W
            pe = self.reset_pe((H, W), device)
            self.pe_dict[f"{H}-{W}"] = pe  # save new PE
        return pe

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        H, W = x.shape[-2:]
        # if in testing, and test_shape!=train_shape, reset PE
        if f"{H}-{W}" in self.pe_dict:  # if the cache has this PE weights, use it
            pe = self.pe_dict[f"{H}-{W}"]
        else:  # or re-generate new PE weights for H-W
            pe = self.reset_pe((H, W), x.device)
            self.pe_dict[f"{H}-{W}"] = pe  # save new PE

        return x + pe  # the shape must be the same


class PositionEncodingSine3DNorm(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(12, 16, 16)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()
        assert len(max_shape) == 3
        self.d_model = d_model
        self.max_shape = max_shape
        self.pe_dict = dict()

    def reset_pe(self, new_shape, device):
        (F, H, W) = new_shape

        hw_dim = int(self.d_model * 0.75)
        f_dim = int(self.d_model * 0.25)
        assert f_dim + hw_dim // 2 * 2 == self.d_model

        pe_hw = torch.zeros((hw_dim, F, H, W))
        pe_f = torch.zeros((f_dim, F, H, W))

        z_position = torch.ones((F, H, W)).cumsum(0).float().unsqueeze(0) * self.max_shape[0] / F
        y_position = torch.ones((F, H, W)).cumsum(1).float().unsqueeze(0) * self.max_shape[1] / H
        x_position = torch.ones((F, H, W)).cumsum(2).float().unsqueeze(0) * self.max_shape[2] / W

        div_term = torch.exp(torch.arange(0, hw_dim // 2, 2).float() * (-math.log(10000.0) / (hw_dim // 2)))
        div_term = div_term[:, None, None, None]  # [C//4, 1, 1, 1]
        pe_hw[0::4, ...] = torch.sin(x_position * div_term)
        pe_hw[1::4, ...] = torch.cos(x_position * div_term)
        pe_hw[2::4, ...] = torch.sin(y_position * div_term)
        pe_hw[3::4, ...] = torch.cos(y_position * div_term)

        div_term = torch.exp(torch.arange(0, f_dim, 2).float() * (-math.log(10000.0) / f_dim))
        div_term = div_term[:, None, None, None]  # [C//8, 1, 1, 1]
        pe_f[0::2, ...] = torch.sin(z_position * div_term)
        pe_f[1::2, ...] = torch.cos(z_position * div_term)

        pe = torch.cat([pe_hw, pe_f], dim=0)

        return pe.unsqueeze(0).to(device)

    def forward(self, x):
        """
        Args:
            x: [N, C, F, H, W]
        """
        F, H, W = x.shape[-3:]
        # if in testing, and test_shape!=train_shape, reset PE
        if f"{F}-{H}-{W}" in self.pe_dict:  # if the cache has this PE weights, use it
            pe = self.pe_dict[f"{F}-{H}-{W}"]
        else:  # or re-generate new PE weights for H-W
            pe = self.reset_pe((F, H, W), x.device)
            self.pe_dict[f"{F}-{H}-{W}"] = pe  # save new PE

        return x + pe  # the shape must be the same


class FlowPositionalEncodingNorm(nn.Module):
    def __init__(self, d_model, h_max=256, w_max=256):
        super().__init__()
        self.d_model = d_model
        self.h_max = h_max
        self.w_max = w_max
        self.div_term = torch.exp(torch.arange(0, self.d_model // 2, 2) * (-math.log(10000.0) / (self.d_model // 2)))

    def forward(self, x):
        # x:[b,3,f-1,h,w]
        b, _, f, h_new, w_new = x.shape
        x_pos = x[:, 0:1] * self.w_max / w_new
        y_pos = x[:, 1:2] * self.h_max / h_new

        x_pe = torch.zeros((b, self.d_model // 2, f, h_new, w_new), dtype=x.dtype, device=x.device)
        y_pe = torch.zeros((b, self.d_model // 2, f, h_new, w_new), dtype=x.dtype, device=x.device)
        div_term = self.div_term.reshape(1, -1, 1, 1, 1).to(x.device, dtype=x.dtype)

        x_pe[:, 0::2] = torch.sin(x_pos * div_term)
        x_pe[:, 1::2] = torch.cos(x_pos * div_term)

        y_pe[:, 0::2] = torch.sin(y_pos * div_term)
        y_pe[:, 1::2] = torch.cos(y_pos * div_term)

        return torch.cat([x_pe, y_pe, x[:, 2:3]], dim=1)
