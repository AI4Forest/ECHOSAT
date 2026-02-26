import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DropPath(nn.Module):
    """
    Randomly drops entire layers in a residual path during training.

    According to https://arxiv.org/abs/1810.12890
    """
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


def window_partition_3d(x, window_size):
    """
    Partitions the input tensor into windows of size window_size.

    According to https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py

    Args:
        x: (B, T, H, W, E)
        window_size: (int, int, int)
    Returns:
        windows: (num_windows*B, window_size[0], window_size[1], window_size[2], E)
    """
    logger.debug(f"Window partition input shape: {x.shape}, window_size: {window_size}")
    
    B, T, H, W, E = x.shape
    x = x.view(B,
               T // window_size[0], window_size[0],
               H // window_size[1], window_size[1],
               W // window_size[2], window_size[2],
               E)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7) \
        .contiguous() \
        .view(-1, window_size[0], window_size[1], window_size[2], E)

    logger.debug(f"Window partition output shape: {windows.shape}")
    return windows


def window_reverse_3d(windows, window_size, T, H, W):
    """
    Undoes the window partitioning by reconstructing the original tensor.

    According to https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py

    Args:
        windows: (num_windows*B, window_size[0], window_size[1], window_size[2], E)
        window_size: (int, int, int)
        T: int
        H: int
        W: int
    Returns:
        x: (B, T, H, W, E)
    """
    logger.debug(f"Window reverse input shape: {windows.shape}, target T,H,W: {T},{H},{W}")
    
    B = int(windows.shape[0] / (T * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B,
                     T // window_size[0],
                     H // window_size[1],
                     W // window_size[2],
                     window_size[0],
                     window_size[1],
                     window_size[2],
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7) \
        .contiguous() \
        .view(B, T, H, W, -1)
    
    logger.debug(f"Window reverse output shape: {x.shape}")
    return x



class TemporalDownsample(nn.Module):
    """Temporal downsampling layer that reduces temporal dimension to target resolution.
    
    Args:
        dim (int): Number of input channels (E).
        input_temporal_dim (int): Input temporal dimension (T).
        target_resolution (int): Target temporal resolution after downsampling.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(
        self, dim, input_temporal_dim, target_resolution, norm_layer=nn.LayerNorm, downsample_per_year=False
    ):

        super().__init__()
        self.dim = dim
        self.input_temporal_dim = input_temporal_dim
        self.target_resolution = target_resolution
        self.downsample_per_year = downsample_per_year
        
        if downsample_per_year:
            # Temporal reduction: project T//T_target -> T_target
            self.temporal_proj = nn.Linear(dim * input_temporal_dim//7, dim * target_resolution//7, bias=False)
        else:
            # Temporal reduction: project T -> T_target
            self.temporal_proj = nn.Linear(dim * input_temporal_dim, dim * target_resolution, bias=False)
            

        # Spatial reduction: merge 2x2 patches, then project 4E -> 2E
        self.norm = norm_layer(4 * dim)
        self.spatial_reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        """
        x: (B, T, H, W, E)
        """
        B, T, H, W, E = x.shape
        assert (
            T == self.input_temporal_dim
        ), f"Expected temporal dim {self.input_temporal_dim}, got {T}"
        
        def process_temporal_reduction(x):
            B, T, H, W, E = x.shape
            if self.downsample_per_year:
                # (B, T, H, W, E) -> (B, T_target, H, W, E)
                x = x.view(B, 7, -1, H, W, E) 
                x = x.permute(0, 1, 3, 4, 5, 2).contiguous()
                x = x.view(B, 7, H, W, E*(T//7)) 
                x = self.temporal_proj(x)  # (B, 7, H, W, E * T_target)
                x = x.view(B, 7, H, W, E, self.target_resolution//7) # (B, 7, H, W, E, T_target)
                x = x.permute(0, 1, 5, 2, 3, 4).contiguous()  # (B, T_target, H, W, E)
                x = x.view(B, self.target_resolution, H, W, E)
            else:
                # (B, T, H, W, E) -> (B, T_target, H, W, E)
                x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, H, W, E, T)
                x = x.view(B, H, W, E*T)
                x = self.temporal_proj(x)  # (B, H, W, E, T_target)
                x = x.view(B, H, W, E, self.target_resolution)
                x = x.permute(0, 4, 1, 2, 3)  # (B, T_target, H, W, E)
            return x
            
            
        x = process_temporal_reduction(x)

        # ---- Spatial reduction ----
        # Pad if needed
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, 0))
            _, _, H, W, _ = x.shape

        # Extract 2x2 patches
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]

        # Concatenate along channel dim
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, T_target, H/2, W/2, 4E)

        # Normalize + project to 2E
        x = self.norm(x)
        x = self.spatial_reduction(x)  # (B, T_target, H/2, W/2, 2E)

        return x


class TemporalSkipConnection(nn.Module):
    """
    Handles skip connections between encoder and decoder with temporal dimension mismatch.
    
    Strategy: 
    1. Reshape encoder output: (B, T_enc, H, W, E) -> (B, H, W, T_enc*E)
    2. Apply Linear layer to reduce temporal dimension: T_enc*E -> T_dec*E
    3. Concatenate with decoder output: (B, H, W, 2*T_dec*E)
    4. Apply final Linear layer: 2*T_dec*E -> T_dec*E
    5. Reshape back: (B, H, W, T_dec*E) -> (B, T_dec, H, W, E)
    
    Uses simple linear transformations without activations or biases for minimal parameters.
    """
    
    def __init__(self, channels, encoder_temporal_dim, decoder_temporal_dim, temporal_skip_reduction="linear"):
        super().__init__()
        self.channels = channels  # E
        self.encoder_temporal_dim = encoder_temporal_dim  # T_enc
        self.decoder_temporal_dim = decoder_temporal_dim  # T_dec
        
        self.temporal_skip_reduction = temporal_skip_reduction
        
        if temporal_skip_reduction == "linear":
            # Simple linear layer to reduce temporal dimension in encoder features
            # T_enc*E -> T_dec*E
            self.temporal_reduction = nn.Linear(
                encoder_temporal_dim * channels, 
                decoder_temporal_dim * channels, 
                bias=False
            )
            
            # Simple linear layer to combine encoder and decoder features
            # 2*T_dec*E -> T_dec*E
            self.feature_fusion = nn.Linear(
                2 * decoder_temporal_dim * channels, 
                decoder_temporal_dim * channels, 
                bias=False
            )
            
            logger.info(f"TemporalSkipConnection init: channels={channels}, "
                    f"encoder_temporal={encoder_temporal_dim}, decoder_temporal={decoder_temporal_dim}")
            logger.debug(f"Parameters added: {encoder_temporal_dim * channels * decoder_temporal_dim * channels + 2 * decoder_temporal_dim * channels * decoder_temporal_dim * channels}")
        elif temporal_skip_reduction.startswith("transformer"):
            # Ensure nhead is even and at least 2 for proper attention mechanism
            #nhead = max(2, (channels // 16) if (channels // 16) % 2 == 0 else (channels // 16) + 1)
            nhead = channels // 16
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=nhead, dim_feedforward=channels*4, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        else:
            raise ValueError(f"Temporal skip reduction {temporal_skip_reduction} not implemented.")
        
    
    def forward(self, encoder_features, decoder_features):
        """
        Args:
            encoder_features: (B, T_enc, H, W, E)
            decoder_features: (B, T_dec, H, W, E)
        Returns:
            fused_features: (B, T_dec, H, W, E)
        """
        B_enc, T_enc, H_enc, W_enc, E_enc = encoder_features.shape
        B_dec, T_dec, H_dec, W_dec, E_dec = decoder_features.shape
        
        logger.debug(f"Skip connection - Encoder: {encoder_features.shape}, Decoder: {decoder_features.shape}")
        
        if self.temporal_skip_reduction == "linear":
            # Ensure spatial dimensions match (take minimum)
            min_H = min(H_enc, H_dec)
            min_W = min(W_enc, W_dec)
            
            encoder_features = encoder_features[:, :, :min_H, :min_W, :]
            decoder_features = decoder_features[:, :, :min_H, :min_W, :]
            
            # Step 1: Reshape encoder features (B, T_enc, H, W, E) -> (B, H, W, T_enc*E)
            encoder_reshaped = encoder_features.permute(0, 2, 3, 1, 4).contiguous()  # (B, H, W, T_enc, E)
            encoder_reshaped = encoder_reshaped.view(B_enc, min_H, min_W, T_enc * E_enc)  # (B, H, W, T_enc*E)
            
            # Step 2: Apply temporal reduction (T_enc*E -> T_dec*E)
            encoder_reduced = self.temporal_reduction(encoder_reshaped)  # (B, H, W, T_dec*E)
            
            # Step 3: Process decoder features (B, T_dec, H, W, E) -> (B, H, W, T_dec*E)
            decoder_reshaped = decoder_features.permute(0, 2, 3, 1, 4).contiguous()  # (B, H, W, T_dec, E)
            decoder_reshaped = decoder_reshaped.view(B_dec, min_H, min_W, T_dec * E_dec)  # (B, H, W, T_dec*E)
            
            # Step 4: Concatenate encoder and decoder features (B, H, W, 2*T_dec*E)
            combined_features = torch.cat([encoder_reduced, decoder_reshaped], dim=-1)  # (B, H, W, 2*T_dec*E)
            
            # Step 5: Apply fusion (2*T_dec*E -> T_dec*E)
            fused_features = self.feature_fusion(combined_features)  # (B, H, W, T_dec*E)
            
            # Step 6: Reshape back to (B, T_dec, H, W, E)
            fused_features = fused_features.view(B_enc, min_H, min_W, self.decoder_temporal_dim, E_enc)  # (B, H, W, T_dec, E)
            fused_features = fused_features.permute(0, 3, 1, 2, 4).contiguous()  # (B, T_dec, H, W, E)
        
        elif self.temporal_skip_reduction == "transformer_all":
            
            concat_features = torch.cat([decoder_features, encoder_features], dim=1)
            concat_features = concat_features.permute(0, 2, 3, 1, 4).contiguous()
            concat_features = concat_features.view(B_enc*H_enc*W_enc, T_dec+T_enc, E_enc)
            
            fused_features = self.transformer_encoder(concat_features)
            fused_features = fused_features.view(B_enc, H_enc, W_enc, T_dec+T_enc, E_dec)
            fused_features = fused_features.permute(0, 3, 1, 2, 4).contiguous()
            fused_features = fused_features[:, :T_dec, :, :, :]
            
        elif self.temporal_skip_reduction == "transformer_year":
            encoder_features = encoder_features.view(B_enc, 7, T_enc//7, H_enc, W_enc, E_enc)
            decoder_features = decoder_features.view(B_dec, 7, 1, H_dec, W_dec, E_dec)
            concat_features = torch.cat([decoder_features, encoder_features], dim=2)
            
            concat_features = concat_features.permute(0, 3, 4, 1, 2, 5).contiguous()
            concat_features = concat_features.view(B_enc * H_enc * W_enc * 7, (T_dec+T_enc)//7, E_enc)
            
            # Very bad solution to solve the problem of batch size beeing greather than 65535 here
            fused_features1 = self.transformer_encoder(concat_features[:concat_features.shape[0]//2])
            fused_features2 = self.transformer_encoder(concat_features[concat_features.shape[0]//2:])
            fused_features = torch.cat([fused_features1, fused_features2], dim=0)
            
            fused_features = fused_features.view(B_enc * H_enc * W_enc * 7, (T_dec+T_enc)//7, E_enc)
            fused_features = fused_features[:,0,:]
            fused_features = fused_features.view(B_enc, H_enc, W_enc, 7, E_dec)
            fused_features = fused_features.permute(0, 3, 1, 2, 4).contiguous()
        
        logger.debug(f"Skip connection output: {fused_features.shape}")
        return fused_features


class WindowAttention3D(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias for 3D.

    According to https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py
    
    Args:
        dim (int): Number of input channels.
        window_size (list[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        logger.debug(f"WindowAttention3D init: dim={dim}, window_size={window_size}, num_heads={num_heads}")

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        # Used as efficient implementation of relative position bias
        coords_t = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_t, coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # Projection layers for the multi-head self-attention (MHSA) block
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # Map input token to query, key, value (separated later)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)                     # Project concatenated (or summed) attention outputs back to original feature dimension
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B_, N, E) - tokens inside one or many windows
                B_ = batch_size x num_windows
                N = tokens per window (w_t·w_h·w_w)
                E = embedding dimension
            mask: (num_windows, N, N) - attention mask for shifted windows (optional)
        """
        B_, N, E = x.shape
        logger.debug(f"WindowAttention3D input shape: {x.shape}")
        
        # (B_, N, 3E) -> (B_, N, 3, h, E // h) -> (3, B_, h, N, E // h)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, E // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logger.debug(f"QKV shapes: q={q.shape}, k={k.shape}, v={v.shape}")

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))        # (B_, h, N, N)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)               # (B_, h, N, N)
        attn = self.attn_drop(attn)             # Dropout on attention weights

        x = attn @ v                            # (B_, h, N, E/h)
        x = x.transpose(1, 2).reshape(B_, N, E) # concat heads
        x = self.proj(x)                        # final linear
        x = self.proj_drop(x)                   # dropout on output
        
        logger.debug(f"WindowAttention3D output shape: {x.shape}")
        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block for 3D data.

    According to https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        encoder (bool, optional): If True, use encoder MLP (4->1). If False, use decoder MLP (4->2). Default: True
    """

    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, encoder=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.encoder = encoder

        logger.debug(f"SwinTransformerBlock3D init: dim={dim}, window_size={window_size}, shift_size={shift_size}, encoder={encoder}")

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        # Conditional MLP based on encoder/decoder
        mlp_hidden_dim = int(dim * mlp_ratio)
        if encoder:
            # Encoder: 1 -> 4 -> 1
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim),
                act_layer(),
                nn.Dropout(drop),
                nn.Linear(mlp_hidden_dim, dim),
                nn.Dropout(drop)
            )
            self.shortcut_proj = None
        else:
            # Decoder: 1 -> 4 -> 2 (no bottleneck!)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim),
                act_layer(),
                nn.Dropout(drop),
                nn.Linear(mlp_hidden_dim, dim * 2),  # Direct 4 -> 2
                nn.Dropout(drop)
            )
            # Project shortcut to match
            self.shortcut_proj = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x, mask_matrix):
        """
        Args:
            x: Input feature, tensor size (B, T, H, W, E).
            mask_matrix: Attention mask for cyclic shift.
        """
        B, T, H, W, E = x.shape
        window_size, shift_size = self.window_size, self.shift_size
        logger.debug(f"SwinTransformerBlock3D input shape: {x.shape}, encoder={self.encoder}")

        shortcut = x
        x = self.norm1(x)

        # Pad feature maps to multiples of window size
        pad_l = pad_t = pad_t0 = 0
        pad_t1 = (window_size[0] - T % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        
        if pad_t1 > 0 or pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_t0, pad_t1))
            logger.debug(f"After padding: {x.shape}, pads: t1={pad_t1}, b={pad_b}, r={pad_r}")
        
        _, Tp, Hp, Wp, _ = x.shape

        # Cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
            logger.debug(f"Applied cyclic shift: {shift_size}")
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows = window_partition_3d(shifted_x, window_size)
        x_windows = x_windows.view(-1, window_size[0] * window_size[1] * window_size[2], E)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], window_size[2], E)
        shifted_x = window_reverse_3d(attn_windows, window_size, Tp, Hp, Wp)

        # Reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_t1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :T, :H, :W, :].contiguous()

        # Residual connection after attention
        x = shortcut + self.drop_path(x)
        
        # MLP with appropriate residual connection
        if self.encoder:
            # Standard residual: input and output have same dimensions
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            # Project shortcut to match MLP output dimensions
            shortcut_projected = self.shortcut_proj(x)  # (B,T,H,W,E) -> (B,T,H,W,2*E)
            mlp_out = self.mlp(self.norm2(x))           # (B,T,H,W,E) -> (B,T,H,W,2*E)
            x = shortcut_projected + self.drop_path(mlp_out)

        logger.debug(f"SwinTransformerBlock3D output shape: {x.shape}")
        return x


class BasicLayer3D(nn.Module):
    """A basic Swin Transformer layer for one stage.

    According to https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        target_temporal_resolution (int, optional): Target temporal resolution. Default: None
    """

    def __init__(self, 
                 dim, 
                 depth, 
                 num_heads, 
                 window_size=(2, 7, 7),
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 norm_layer=nn.LayerNorm, 
                 downsample=True, 
                 target_temporal_resolution=None,
                 input_temporal_dim=None,
                 downsample_per_year=False
                ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.downsample_per_year = downsample_per_year
        
        logger.debug(f"BasicLayer3D init: dim={dim}, depth={depth}, window_size={window_size}")

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                encoder=True,
                )
            for i in range(depth)])

        if downsample is False:
            self.downsample = None
        elif target_temporal_resolution is not None:
            assert input_temporal_dim is not None, "input_temporal_dim must be provided when using TemporalDownsample"
            self.downsample = TemporalDownsample(
                dim=dim, 
                input_temporal_dim=input_temporal_dim,
                target_resolution=target_temporal_resolution, 
                norm_layer=norm_layer,
                downsample_per_year=downsample_per_year
            )
        else:
            self.downsample = PatchMerging(dim=dim, norm_layer=norm_layer)


    def forward(self, x):
        # Calculate attention mask for SW-MSA
        B, T, H, W, E = x.shape
        window_size, shift_size = self.window_size, self.shift_size
        logger.debug(f"BasicLayer3D input shape: {x.shape}")
        
        Tp = int(math.ceil(T / window_size[0])) * window_size[0]
        Hp = int(math.ceil(H / window_size[1])) * window_size[1]
        Wp = int(math.ceil(W / window_size[2])) * window_size[2]
        
        attn_mask = None
        if any(i > 0 for i in shift_size):
            img_mask = torch.zeros((1, Tp, Hp, Wp, 1), device=x.device)
            t_slices = (slice(0, -window_size[0]),
                       slice(-window_size[0], -shift_size[0]),
                       slice(-shift_size[0], None))
            h_slices = (slice(0, -window_size[1]),
                       slice(-window_size[1], -shift_size[1]),
                       slice(-shift_size[1], None))
            w_slices = (slice(0, -window_size[2]),
                       slice(-window_size[2], -shift_size[2]),
                       slice(-shift_size[2], None))
            cnt = 0
            for t in t_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, t, h, w, :] = cnt
                        cnt += 1

            mask_windows = window_partition_3d(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size[0] * window_size[1] * window_size[2])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x, attn_mask)
            
        if self.downsample is not None:
            x_down = self.downsample(x)
            logger.info(f"BasicLayer3D after downsample: {x_down.shape}")
            return x, x_down
        else:
            logger.debug(f"BasicLayer3D output (no downsample): {x.shape}")
            return x, x


class PatchMerging(nn.Module):
    """Patch Merging Layer for 3D data.

    H and W dimensions are halved; channels are doubled.
    Temporal dim T is kept unchanged.

    According to https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # 2x2 spatial reduction, double output channel dimension
        self.reduction = nn.Linear((2*2) * dim, 2 * dim, bias=False)
        self.norm = norm_layer((2*2) * dim)
        logger.debug(f"PatchMerging init: dim={dim}")

    def forward(self, x):
        B, T, H, W, E = x.shape
        logger.debug(f"PatchMerging input shape: {x.shape}")

        # Pad if any dimension is odd
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, 0))
            _, T, H, W, _ = x.shape
            logger.debug(f"After padding: {x.shape}")

        # Split into 4 non-overlapping 2x2 sub-volumes
        x0 = x[:, :, 0::2, 0::2, :]  # (B, T, H/2, W/2, E)
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, T, H/2, W/2, 4E)

        logger.debug(f"After concatenation: {x.shape}")

        # Linear projection to 2E channels
        x = self.norm(x)
        x = self.reduction(x)  # (B, T, H/2, W/2, 2E)

        logger.info(f"PatchMerging output shape: {x.shape}")
        return x


class PatchMerging3D(nn.Module):
    """Patch Merging Layer for 3D data.

    H, W, and T dimensions are halved; channels are doubled.

    According to https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # 2x2x2 spatial and temporal reduction, double output channel dimension
        self.reduction = nn.Linear((2*2*2) * dim, 2 * dim, bias=False)
        self.norm = norm_layer((2*2*2) * dim)
        logger.debug(f"PatchMerging3D init: dim={dim}")

    def forward(self, x):
        B, T, H, W, E = x.shape
        logger.debug(f"PatchMerging3D input shape: {x.shape}")

        # Pad if any dimension is odd
        pad_t = T % 2
        pad_h = H % 2
        pad_w = W % 2
        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
            _, T, H, W, _ = x.shape
            logger.debug(f"After padding: {x.shape}")

        # Split into 8 non-overlapping 2x2x2 sub-volumes
        x0 = x[:, 0::2, 0::2, 0::2, :]  # (B, T/2, H/2, W/2, E)
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)  # (B, T/2, H/2, W/2, 8E)

        logger.debug(f"After concatenation: {x.shape}")

        # Linear projection to 2E channels
        x = self.norm(x)
        x = self.reduction(x)  # (B, T/2, H/2, W/2, 2E)

        logger.info(f"PatchMerging3D output shape: {x.shape}")
        return x


class PatchExpand(nn.Module):
    """Patch expanding layer for 3-D data.

    H and W dimensions are doubled; channels are halved.
    Temporal dim T is kept unchanged.

    Works with SwinTransformerBlock3DDecoder that outputs 2x channels.
    """

    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # Input will be 2*dim from the decoder block, we need to reshape it
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        # Input x has shape (B, T, H, W, 2*E) from SwinTransformerBlock3DDecoder
        B, T, H, W, doubled_E = x.shape
        E = doubled_E // 2  # Original channel dimension
        
        # Reshape (2 -> 0.5, P_W * 2, P_H * 2)
        # We have 2*E channels, reshape to get spatial upsampling
        x = x.view(B, T, H, W, 2, 2, E // 2)    # (B, T, H, W, 2, 2, E/2)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = x.view(B, T, H * 2, W * 2, E // 2)  # (B, T, 2H, 2W, E/2)
        
        # Normalization
        x = self.norm(x)
        return x


class BasicLayerUp3D(nn.Module):
    """A basic Swin Transformer layer for upsampling.

    According to https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py
    but with added 3D support.
    """
    
    def __init__(self, dim, depth, num_heads, window_size=(2, 7, 7),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=True, force_encoder_blocks=False, downsample_per_year=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.downsample_per_year = downsample_per_year

        logger.debug(f"BasicLayerUp3D init: dim={dim}, depth={depth}")

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                encoder=True if force_encoder_blocks else (i != depth - 1),
                )
            for i in range(depth)])

        if upsample is False:
            self.upsample = None
        else:
            self.upsample = PatchExpand(dim, norm_layer=norm_layer)


    def forward(self, x):
        # Calculate attention mask for SW-MSA
        B, T, H, W, E = x.shape
        window_size, shift_size = self.window_size, self.shift_size
        logger.debug(f"BasicLayerUp3D input shape: {x.shape}")
        
        Tp = int(math.ceil(T / window_size[0])) * window_size[0]
        Hp = int(math.ceil(H / window_size[1])) * window_size[1]
        Wp = int(math.ceil(W / window_size[2])) * window_size[2]
        
        attn_mask = None
        if any(i > 0 for i in shift_size):
            img_mask = torch.zeros((1, Tp, Hp, Wp, 1), device=x.device)
            t_slices = (slice(0, -window_size[0]),
                       slice(-window_size[0], -shift_size[0]),
                       slice(-shift_size[0], None))
            h_slices = (slice(0, -window_size[1]),
                       slice(-window_size[1], -shift_size[1]),
                       slice(-shift_size[1], None))
            w_slices = (slice(0, -window_size[2]),
                       slice(-window_size[2], -shift_size[2]),
                       slice(-shift_size[2], None))
            cnt = 0
            for t in t_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, t, h, w, :] = cnt
                        cnt += 1

            mask_windows = window_partition_3d(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size[0] * window_size[1] * window_size[2])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x, attn_mask)
            
        if self.upsample is not None:
            x = self.upsample(x)
            
        logger.info(f"BasicLayerUp3D output shape: {x.shape}")
        return x


class SwinVideoUnet(nn.Module):
    """
    SwinVideoUnet: A U-Net architecture with Video Swin Transformer blocks.
    
    Args:
        input_shape: Tuple of (years, months, channels, height, width)
        embed_dim: Patch embedding dimension
        encoder_depths: Depth of each encoder Swin Transformer layer
        decoder_depths: Depth of each decoder Swin Transformer layer
        num_heads: Number of attention heads in different layers
        window_size: Window size for attention computation
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        qkv_bias: If True, add a learnable bias to query, key, value
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        norm_layer: Normalization layer
        reduce_time: Reduction of temporal dimension in encoder (unchanged in decoder)
        patch_size_time: Patch size for temporal dimension
        patch_size_image: Patch size for spatial dimensions
    """
    
    def __init__(self, 
                 input_shape: tuple[int, int, int, int, int] = (4, 6, 12, 256, 256),
                 embed_dim: int = 96,
                 encoder_depths: list[int] = [2, 2, 6, 2],
                 decoder_depths: list[int] = [2, 6, 2, 2],
                 num_heads: list[int] = [3, 6, 12, 24],
                 window_size_temporal: int = 2,
                 window_size_spatial: int = 7,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.1,
                 attn_drop_rate: float = 0.1,
                 drop_path_rate: float = 0.1,
                 norm_layer=nn.LayerNorm,
                 reduce_time: list[int] = [16, 8, 4],
                 patch_size_time: int = 1,
                 patch_size_image: int = 1,
                 temporal_skip_reduction="linear",
                 use_final_convs: bool = True,
                 downsample_per_year: bool = True,
                 ):
        super().__init__()
        self.input_shape = input_shape

        print("######## Input shape", input_shape, "########")

        years, months, C, H, W = input_shape
        self.num_layers = len(encoder_depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.reduce_time = reduce_time
        self.patch_size_time = patch_size_time
        self.patch_size_image = patch_size_image
        self.window_size = (window_size_temporal, window_size_spatial, window_size_spatial)
        
        self.temporal_skip_reduction = temporal_skip_reduction
        self.use_final_convs = use_final_convs

        logger.info(f"SwinVideoUnet init: input_shape={input_shape}, embed_dim={embed_dim}")
        logger.info(f"Encoder depths: {encoder_depths}, Decoder depths: {decoder_depths}")
        logger.info(f"Num heads: {num_heads}, window_size: {self.window_size}")
        logger.info(f"Temporal reductions: {reduce_time}")
        logger.info(f"Patch sizes: time={patch_size_time}, image={patch_size_image}")

        # Calculate temporal dimensions at each encoder level
        initial_temporal_dim = years * months // patch_size_time
        self.encoder_temporal_dims = [initial_temporal_dim]
        
        current_temporal_dim = initial_temporal_dim
        for i in range(len(reduce_time)):
            if i < len(reduce_time):
                current_temporal_dim = reduce_time[i]
            self.encoder_temporal_dims.append(current_temporal_dim)
        
        # Pad with the last value if needed
        while len(self.encoder_temporal_dims) < self.num_layers + 1:
            self.encoder_temporal_dims.append(self.encoder_temporal_dims[-1])
        
        logger.info(f"Calculated encoder temporal dimensions: {self.encoder_temporal_dims}")

        # Patch embedding with configurable patch sizes
        self.patch_embed = nn.Conv3d(C, embed_dim, 
                                   kernel_size=(patch_size_time, patch_size_image, patch_size_image), 
                                   stride=(patch_size_time, patch_size_image, patch_size_image))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr_encoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(encoder_depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(decoder_depths))]

        # Build encoder layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # Determine target temporal resolution and input temporal dimension for this layer
            target_temporal_resolution = None
            input_temporal_dim = None
            
            if i_layer < len(reduce_time):
                target_temporal_resolution = reduce_time[i_layer]
                input_temporal_dim = self.encoder_temporal_dims[i_layer]

            layer = BasicLayer3D(
                dim=int(embed_dim * 2 ** i_layer),
                depth=encoder_depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_encoder[sum(encoder_depths[:i_layer]):sum(encoder_depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=not(i_layer == self.num_layers - 1),  # Only downsample in intermediate layers
                target_temporal_resolution=target_temporal_resolution,
                input_temporal_dim=input_temporal_dim,
                downsample_per_year=downsample_per_year,
            )
            self.layers.append(layer)
        
        # Build decoder layers
        self.layers_up = nn.ModuleList()

        # Create temporal skip connections for each decoder layer (except the first one)
        self.skip_connections = nn.ModuleList()

        for i_layer in range(self.num_layers):
            # Skip connection (not needed for the first decoder layer)
            if i_layer > 0:
                encoder_level = self.num_layers - 1 - i_layer
                skip_conn = TemporalSkipConnection(
                    channels=int(embed_dim * 2 ** encoder_level),
                    encoder_temporal_dim=self.encoder_temporal_dims[encoder_level],
                    temporal_skip_reduction=temporal_skip_reduction,
                    decoder_temporal_dim=years  # Assuming decoder works with temporal dim of 7
                )
                self.skip_connections.append(skip_conn)
            else:
                self.skip_connections.append(None)

            layer_up = BasicLayerUp3D(
                dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                depth=decoder_depths[i_layer],
                num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                window_size=self.window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(decoder_depths[:i_layer]):sum(decoder_depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=not(i_layer == self.num_layers - 1),  # Only upsample in intermediate layers
                force_encoder_blocks=(i_layer == self.num_layers - 1),
                downsample_per_year=downsample_per_year,
            )
            self.layers_up.append(layer_up)
        
        self.norm = norm_layer(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.norm_up = norm_layer(self.embed_dim)
        
        # Final output layer
        if self.use_final_convs:
            self.final_conv = nn.Sequential(
                nn.Conv3d(embed_dim, embed_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1), padding_mode='reflect'),
                nn.GroupNorm((embed_dim//32)+1, embed_dim),
                nn.GELU(),
                nn.Conv3d(embed_dim, embed_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode='reflect'),
                nn.GroupNorm((embed_dim//32)+1, embed_dim),
                nn.GELU(),
            )
        self.final_conv3 = nn.Conv3d(embed_dim, 1, kernel_size=(1, 1, 1))
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Input format: (batch, years, months, channels, H, W)
        B, C, Y, M, H, W = x.shape
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()

        logger.info(f"Forward pass input shape: (B={B}, Y={Y}, M={M}, C={C}, H={H}, W={W})")
        
        # Input validation
        assert M % self.patch_size_time == 0 and H % self.patch_size_image == 0 and W % self.patch_size_image == 0, \
            f"Input dimensions must be compatible with patch embedding: M={M}, H={H}, W={W}, patch_time={self.patch_size_time}, patch_image={self.patch_size_image}"
        
        # Reshape to: (B, Y * M, C, H, W)
        x = x.view(B, Y * M, C, H, W)
        logger.debug(f"After reshape: (B={B}, T={Y * M}, C={C}, H={H}, W={W})")

        # Permute to Conv3d format: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        logger.debug(f"After permute for patch embedding: (B={B}, C={C}, T={Y * M}, H={H}, W={W})")

        x = self.patch_embed(x)
        logger.info(f"After patch embedding: {x.shape}")

        # Permute to format: (B, T, H, W, E)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.pos_drop(x)
        
        logger.debug(f"After permute to (B, T, H, W, E): {x.shape}")

        # Store skip connections
        x_downsample = []
        
        # Encoder
        logger.info("=== ENCODER ===")
        for i, layer in enumerate(self.layers):
            x_downsample.append(x)
            logger.info(f"Encoder layer {i} input: {x.shape}")
            x, x_down = layer(x)
            if x_down is not x:
                x = x_down
        
        x = self.norm(x)
        logger.info(f"After encoder norm: {x.shape}")
        
        # Decoder
        logger.info("=== DECODER ===")
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                skip_connection = x_downsample[self.num_layers - 1 - inx]
                
                # Use the temporal skip connection
                skip_conn_module = self.skip_connections[inx]
                if skip_conn_module is not None:
                    x = skip_conn_module(skip_connection, x)
                else:
                    # Fallback to simple concatenation (shouldn't happen)
                    logger.warning(f"No skip connection module for layer {inx}, using fallback")
                    B, T_x, H_x, W_x, E_x = x.shape
                    B, T_skip, H_skip, W_skip, E_skip = skip_connection.shape
                    
                    min_T = min(T_x, T_skip)
                    min_H = min(H_x, H_skip)
                    min_W = min(W_x, W_skip)
                    
                    x = x[:, :min_T, :min_H, :min_W, :]
                    skip_connection = skip_connection[:, :min_T, :min_H, :min_W, :]
                    x = torch.cat([x, skip_connection], -1)
                x = layer_up(x)
            logger.info(f"Decoder layer {inx} output: {x.shape}")
        
        x = self.norm_up(x)
        logger.info(f"After decoder norm: {x.shape}")

        # Current shape: (B, T, H, W, E)
        # Apply conv over E dimension
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, E, T, H, W)
        logger.debug(f"After permute for final conv: {x.shape}")
        
        if self.use_final_convs:
            x = self.final_conv(x)
        x = self.final_conv3(x)
        logger.debug(f"After final conv: {x.shape}")
        x = x.squeeze(1)  # (B, T, H, W)
        logger.info(f"Final output shape: (B={x.shape[0]}, T={x.shape[1]}, H={x.shape[2]}, W={x.shape[3]})")
        
        return x

# # Example usage
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
    
#     # Create model for your specific use case
#     model = SwinVideoUnet(
#         # 4 years, 6 months, 12 channels, 64x64 spatial
#         input_shape=(4, 6, 12, 64, 64),
#         embed_dim=96,
#         encoder_depths=[2, 2, 6, 2],
#         decoder_depths=[2, 2, 2, 6],
#         num_heads=[3, 6, 12, 24],
#         # Temporal window of 12, spatial windows of 4x4
#         window_size_temporal=12, # 2
#         window_size_spatial=4,   # 6 or 8
#         drop_path_rate=0.2,
#         reduce_time=[16, 8, 4],  # Temporal reductions: 24 -> 16 -> 8 -> 4 (unchanged in Decoder)
#         patch_size_time=1,
#         patch_size_image=1,
#     )
    
#     # Test forward pass (B, C, Y, M, H, W)
#     x = torch.randn(2, 12, 4, 6, 64, 64)  # Batch size 2
#     output = model(x)

#     print(f"\nFinal Results:")
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
#     print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
#     print(f"Model size (MB): {sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2):.2f} MB")
