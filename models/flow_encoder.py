import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.animatediff.attention import Attention, FeedForward
from models.position_encoding import PositionEncodingSine2DNorm, PositionEncodingSine3DNorm

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GateConv(nn.Module):
    def __init__(self, in_channels, out_channels, group_size, kernel_size, stride=1, padding=1):
        super(GateConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(num_groups=group_size, num_channels=out_channels, eps=1e-6)

    def forward(self, x):
        x = self.conv(x)  # [B,2C,H,W]
        [out, gate] = torch.chunk(x, 2, dim=1)
        out = F.silu(self.norm(out)) * F.sigmoid(gate)
        return out


def GateConvEncoder(encoder_type, encoder_dims):
    return nn.Sequential(
        GateConv(33, 16, group_size=8, kernel_size=3, padding=1),
        GateConv(16, 32, group_size=16, kernel_size=3, stride=2, padding=1),
        GateConv(32, 64, group_size=32, kernel_size=3, stride=2, padding=1),
        GateConv(64, 128, group_size=32, kernel_size=3, stride=2, padding=1),
        GateConv(128, encoder_dims, group_size=32, kernel_size=3, stride=2 if encoder_type == "gateconv_16" else 1, padding=1),
    )


def DefaultEncoder(encoder_type, encoder_dims):
    if "3d" in encoder_type:
        return nn.Sequential(
            nn.Conv3d(33, 16, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=16, eps=1e-6),
            nn.SiLU(),
            nn.Conv3d(16, 32, kernel_size=3, stride=[1, 2, 2], padding=1),  # 1/2
            nn.GroupNorm(num_groups=16, num_channels=32, eps=1e-6),
            nn.SiLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=[1, 2, 2], padding=1),  # 1/4
            nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6),
            nn.SiLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=[1, 2, 2], padding=1),  # 1/8
            nn.GroupNorm(num_groups=32, num_channels=128, eps=1e-6),
            nn.SiLU(),
            nn.Conv3d(128, encoder_dims, kernel_size=3, stride=[1, 2, 2] if encoder_type == "default3d_16" else 1, padding=1),  # 1/16 or 1/8
            nn.GroupNorm(num_groups=32, num_channels=encoder_dims, eps=1e-6),
            nn.SiLU(),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(33, 16, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=16, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 1/2
            nn.GroupNorm(num_groups=16, num_channels=32, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 1/4
            nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 1/8
            nn.GroupNorm(num_groups=32, num_channels=128, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(128, encoder_dims, kernel_size=3, stride=2 if encoder_type == "default_16" else 1, padding=1),  # 1/16 or 1/8
            nn.GroupNorm(num_groups=32, num_channels=encoder_dims, eps=1e-6),
            nn.SiLU(),
        )

class SlotAttention2D(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_slots, encoder_dims, hidden_dim=128, eps=1e-8, q_type="slot",
                 gru_iters=1):
        """Builds the Slot Attention module.
        Args:
            iters: Number of iterations.
            num_slots: Number of slots.
            encoder_dims: Dimensionality of slot feature vectors.
            hidden_dim: Hidden layer size of MLP.
            eps: Offset for attention coefficients before normalization.
        """
        super(SlotAttention2D, self).__init__()

        self.eps = eps
        self.num_slots = num_slots
        self.scale = encoder_dims ** -0.5
        self.q_type = q_type

        self.norm_input = nn.LayerNorm(encoder_dims)
        self.norm_slots = nn.LayerNorm(encoder_dims)
        self.norm_pre_ff = nn.LayerNorm(encoder_dims)

        self.slots_embedding = nn.Embedding(num_slots, encoder_dims)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(encoder_dims, encoder_dims)
        self.project_k = nn.Linear(encoder_dims, encoder_dims)
        self.project_v = nn.Linear(encoder_dims, encoder_dims)

        # Slot update functions.
        self.gru_iters = max(1, gru_iters)
        if self.gru_iters > 1:  # 超过一次iter，采用gru优化
            self.gru = nn.GRUCell(encoder_dims, encoder_dims)
        else:
            self.gru = None

        hidden_dim = max(encoder_dims, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dims, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, encoder_dims)
        )

    def forward(self, inputs, mask=None):
        # inputs has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_input(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        b, n, d = inputs.shape
        n_s = self.num_slots

        # learnable slots initialization
        slots = self.slots_embedding(torch.arange(0, n_s).expand(b, n_s).to(inputs.device))

        if mask is not None:
            mask = einops.rearrange(mask, "b c h w -> b c (h w)")
            valid = (1 - mask).mean(dim=-1, keepdim=True)  # [b,c,1]
            valid[valid < 0.003] = 0  # mask覆盖面积过大，就不适用attention_mask了(<=3个token)
            valid[valid >= 0.003] = 1
            mask = mask * valid

        for _ in range(self.gru_iters):
            slots_prev = slots
            # One round of attention
            slots = self.norm_slots(slots)
            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale  # [b,ns,hw]

            if self.q_type == "slot":
                attn = dots.softmax(dim=1) + self.eps
                attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean.
            elif self.q_type == "qformer":
                if mask is not None:
                    dots.masked_fill_(mask.bool(), -torch.inf)
                attn = dots.softmax(dim=2)
            else:
                raise NotImplementedError(f"q_type {self.q_type} not implemented")

            updates = torch.einsum('bjd,bij->bid', v, attn)  # [b,ns,c]

            # slot update.
            if self.gru is not None:
                slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))
                slots = slots.reshape(b, -1, d)
            else:
                slots = updates

            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots


class SlotAttention3DBlock(nn.Module):
    def __init__(self, encoder_dims, nhead=8, use_global_slots=False, max_shape=None):
        super(SlotAttention3DBlock, self).__init__()

        self.encoder_dims = encoder_dims
        self.nhead = nhead

        self.norm1 = nn.LayerNorm(encoder_dims)
        self.input_attn2d = Attention(query_dim=encoder_dims, nhead=nhead)

        self.norm2 = nn.LayerNorm(encoder_dims)
        self.input_attn3d = Attention(query_dim=encoder_dims, nhead=nhead)

        self.norm3 = nn.LayerNorm(encoder_dims)
        self.ff1 = FeedForward(encoder_dims)

        self.norm4 = nn.LayerNorm(encoder_dims)
        self.norm5 = nn.LayerNorm(encoder_dims)
        self.slot_attn = Attention(query_dim=encoder_dims, nhead=nhead)

        self.norm6 = nn.LayerNorm(encoder_dims)
        self.ff2 = FeedForward(encoder_dims)

        self.use_global_slots = use_global_slots
        if use_global_slots:
            self.pos_encoding = PositionEncodingSine3DNorm(encoder_dims, max_shape=max_shape)
            self.norm7 = nn.LayerNorm(encoder_dims)
            self.norm8 = nn.LayerNorm(encoder_dims)
            self.global_slot_attn = Attention(query_dim=encoder_dims, nhead=nhead)

            self.norm9 = nn.LayerNorm(encoder_dims)
            self.ff3 = FeedForward(encoder_dims)

    def forward(self, x, slots, mask, f, h, w, i_layer, global_slots=None):
        # x:[b,fhw,c], slots:[bf,ns,c], mask:[b,fhw,1], global_slots:[b,ns,c] or None
        x = einops.rearrange(x, "b (f h w) c -> (b f) (h w) c", f=f, h=h, w=w)
        x = x + self.input_attn2d(self.norm1(x))
        x = einops.rearrange(x, "(b f) (h w) c -> (b h w) f c", f=f, h=h, w=w)
        x = x + self.input_attn3d(self.norm2(x))
        x = x + self.ff1(self.norm3(x))

        x = einops.rearrange(x, "(b h w) f c -> (b f) (h w) c", f=f, h=h, w=w)
        if mask is not None:
            mask = mask.squeeze(-1)  # [b,fhw]
            mask = (1 - mask).bool()
            mask = einops.rearrange(mask, "b (f h w) -> (b f) (h w)", f=f, h=h, w=w)

        if i_layer == 0:
            slots = self.slot_attn(self.norm4(slots), self.norm5(x), attention_mask=mask)  # [b*f,ns,c]
        else:
            slots = slots + self.slot_attn(self.norm4(slots), self.norm5(x), attention_mask=mask)
        slots = slots + self.ff2(self.norm6(slots))

        if self.use_global_slots and global_slots is not None:
            x_temporal = einops.rearrange(x, "(b f) (h w) c -> b c f h w", f=f, h=h, w=w)
            x_temporal = self.pos_encoding(x_temporal)
            x_temporal = einops.rearrange(x_temporal, "b c f h w -> b (f h w) c", f=f, h=h, w=w)

            if i_layer == 0:
                global_slots = self.global_slot_attn(self.norm7(global_slots), self.norm8(x_temporal))
            else:
                global_slots = global_slots + self.global_slot_attn(self.norm7(global_slots), self.norm8(x_temporal))
            global_slots = global_slots + self.ff3(self.norm9(global_slots))  # [b,ns,c]

        x = einops.rearrange(x, "(b f) (h w) c -> b (f h w) c", f=f, h=h, w=w)

        return x, slots, global_slots


class SlotAttention3D(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_slots, encoder_dims, layers=1, num_global_slots=0, max_shape=None):
        """Builds the Slot Attention module.
        Args:
            iters: Number of iterations.
            num_slots: Number of slots.
            encoder_dims: Dimensionality of slot feature vectors.
            hidden_dim: Hidden layer size of MLP.
            eps: Offset for attention coefficients before normalization.
        """
        super(SlotAttention3D, self).__init__()

        self.num_slots = num_slots
        self.layers = layers
        self.num_global_slots = num_global_slots

        self.slots_embedding = nn.Embedding(num_slots, encoder_dims)
        if num_global_slots > 0:
            self.global_slots_embedding = nn.Embedding(num_global_slots, encoder_dims)
        else:
            self.global_slots_embedding = None

        # Slot update functions.
        self.slot_attns = nn.ModuleList()
        for i in range(self.layers):
            self.slot_attns.append(SlotAttention3DBlock(encoder_dims, use_global_slots=True if num_global_slots > 0 else False, max_shape=max_shape))

        self.out_norm = nn.LayerNorm(encoder_dims)

        if num_global_slots > 0:
            self.global_out_norm = nn.LayerNorm(encoder_dims)
        else:
            self.global_out_norm = None

    def forward(self, inputs, mask=None, f=None, h=None, w=None):
        # inputs:[b,fhw,c]
        # mask:[bf,1,h,w]

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        b, n, d = inputs.shape

        # learnable slots initialization
        slots = self.slots_embedding(torch.arange(0, self.num_slots).expand(b * f, self.num_slots).to(inputs.device))
        if self.global_slots_embedding is not None:
            global_slots = self.global_slots_embedding(torch.arange(0, self.num_global_slots).expand(b, self.num_global_slots).to(inputs.device))
        else:
            global_slots = None

        if mask is not None:
            mask = einops.rearrange(mask, "b c h w -> b c (h w)")
            valid = (1 - mask).mean(dim=-1, keepdim=True)  # [b,c,1]
            valid[valid < 0.003] = 0  # 完全mask的情况不适用attention_mask了(<=3个token)
            valid[valid >= 0.003] = 1
            mask = mask * valid
            mask = einops.rearrange(mask, "(b f) c (h w) -> b (f h w) c", h=h, w=w, f=f)

        x = inputs
        for i in range(self.layers):
            x, slots, global_slots = self.slot_attns[i](x, slots, mask, f=f, h=h, w=w, i_layer=i, global_slots=global_slots)

        slots = self.out_norm(slots)
        if global_slots is not None:
            global_slots = self.global_out_norm(global_slots)

        return slots, global_slots


class SlotAttentionEncoder(nn.Module):
    """Slot Attention-based auto-encoder for object discovery."""

    def __init__(self, num_slots, encoder_dims=320, out_channels=1280, zero_output=True, **kwargs):
        """Builds the Slot Attention-based Auto-encoder.

        Args:
            resolution: Tuple of integers specifying width and height of input image
            num_slots: Number of slots in Slot Attention.
            iters: Number of iterations in Slot Attention.
        """
        super(SlotAttentionEncoder, self).__init__()

        self.num_slots = num_slots
        self.num_global_slots = kwargs.get("num_global_slots", 0)
        self.encoder_dims = encoder_dims
        self.encoder_type = kwargs.get("encoder_type", "default_16")
        self.max_shape = kwargs.get("max_shape", [16, 16])
        self.use_mask = kwargs.get("use_mask", False)
        self.slot_type = kwargs.get("slot_type", "slot_attn2d")

        if "default" in self.encoder_type:
            self.encoder = DefaultEncoder(self.encoder_type, self.encoder_dims)
        elif "gateconv" in self.encoder_type:
            self.encoder = GateConvEncoder(self.encoder_type, self.encoder_dims)
        elif "sparseconv" in self.encoder_type:
            assert self.use_mask == True  # sparseconv do not encode masked features
            self.encoder = SparseConvEncoder(self.encoder_type, self.encoder_dims)
        else:
            raise NotImplementedError("Unknown encoder", self.encoder_type)

        self.layer_norm = nn.LayerNorm(self.encoder_dims)

        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_dims, self.encoder_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dims, self.encoder_dims)
        )

        if self.slot_type == "slot_attn2d":
            self.pos_encoding = PositionEncodingSine2DNorm(self.encoder_dims, max_shape=self.max_shape)
            self.slot_attention = SlotAttention2D(
                num_slots=self.num_slots,
                encoder_dims=self.encoder_dims,
                hidden_dim=self.encoder_dims,
                q_type=kwargs.get("q_type", "slot"),
                gru_iters=kwargs.get("gru_iters", 1)
            )
        elif self.slot_type == "slot_attn3d":
            assert "sparseconv" not in self.encoder_type  # slot3d不兼容sparseconv
            self.pos_encoding = PositionEncodingSine3DNorm(self.encoder_dims, max_shape=self.max_shape)
            self.slot_attention = SlotAttention3D(
                num_slots=self.num_slots,
                encoder_dims=self.encoder_dims,
                layers=kwargs.get("layers", 1),
                num_global_slots=kwargs.get("num_global_slots", 0),
                max_shape=self.max_shape,
            )
        else:
            raise NotImplementedError("Unknown slot_type", self.slot_type)

        if zero_output:
            self.out_proj = zero_module(nn.Linear(self.encoder_dims * self.num_slots, out_channels))
            if self.num_global_slots > 0:
                self.global_out_proj = zero_module(nn.Linear(self.encoder_dims * self.num_global_slots, out_channels))
            else:
                self.global_out_proj = None
        else:  # if we use flow as the condition for cross-attention, there is no need for zero_module
            self.out_proj = nn.Linear(self.encoder_dims * self.num_slots, out_channels)
            if self.num_global_slots > 0:
                self.global_out_proj = nn.Linear(self.encoder_dims * self.num_global_slots, out_channels)
            else:
                self.global_out_proj = None

    def forward(self, flow, flow_length):
        # `flow` has shape: [b*f,c,h,w].
        mask = flow[:, -1:]
        if "sparseconv" in self.encoder_type:
            x = self.encoder(flow, f=flow_length)  # CNN Backbone.
        else:
            if "3d" in self.encoder_type:
                flow = einops.rearrange(flow, "(b f) c h w -> b c f h w", f=flow_length)
            x = self.encoder(flow)
            if "3d" in self.encoder_type:
                x = einops.rearrange(x, "b c f h w -> (b f) c h w")

        if self.use_mask:
            h, w = x.shape[-2:]  # [b,1,h,w]

            if "sparseconv" in self.encoder_type:
                mask = F.avg_pool2d(mask, mask.shape[-1] // w)
                mask[mask < 1] = 0
            else:
                mask = F.max_pool2d(mask, mask.shape[-1] // w)
                mask[mask > 0] = 1
        else:
            mask = None

        if self.slot_type == "slot_attn2d":
            x = self.pos_encoding(x)  # position embedding
            x = einops.rearrange(x, "b c h w -> b (h w) c")  # Flatten spatial dimensions (treat image as set).
            x = self.mlp(self.layer_norm(x))  # Feedforward network on set. # `x` has shape: [b*f,h*w,c].
            x = self.slot_attention(x, mask)  # # Slot Attention module.
            x_global = None
        elif self.slot_type == "slot_attn3d":
            x = einops.rearrange(x, "(b f) c h w -> b c f h w", f=flow_length)
            x = self.pos_encoding(x)  # 3D position embedding
            f, h, w = x.shape[-3:]
            x = einops.rearrange(x, "b c f h w -> b (f h w) c")
            x = self.mlp(self.layer_norm(x))  # [b,fhw,c]
            x, x_global = self.slot_attention(x, mask, f=f, h=h, w=w)  # Slot Attention module. x_global is None if we do not use global_slot
        else:
            raise NotImplementedError("Unknown slot_type", self.slot_type)

        # `x` has shape: [b*f,num_slots,c].
        x = einops.rearrange(x, "b s c -> b (s c)")  # [b*f,num_slots*c]
        x = self.out_proj(x)

        if x_global is not None:
            x_global = einops.rearrange(x_global, "b s c -> b (s c)")  # [b,num_slots*c]
            x_global = self.global_out_proj(x_global)
            return x, x_global

        return x


def get_flow_encoder(flow_cfg):
    if flow_cfg['name'] == "default":
        encoder = nn.Sequential(
            nn.Conv2d(33, 48, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=12, num_channels=48, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # 1/2
            nn.GroupNorm(num_groups=16, num_channels=64, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 1/4
            nn.GroupNorm(num_groups=32, num_channels=128, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 1/8
            nn.GroupNorm(num_groups=32, num_channels=256, eps=1e-6),
            nn.SiLU(),
            zero_module(nn.Conv2d(256, 320, kernel_size=3, padding=1))
        )
    elif "slot_attention" in flow_cfg['name']:
        encoder = SlotAttentionEncoder(**flow_cfg)
    else:
        raise NotImplementedError(f"Not implemented flow encoder {flow_cfg['name']}")

    return encoder
