from basicsr.utils.registry import ARCH_REGISTRY
import numpy as np
import torch
from torch.nn.functional import silu
import torch.nn.functional as F
from torch import nn
from basicsr.utils.basic_ops import normalization
from basicsr.utils.swin_transformer import BasicLayer
from torch import Tensor

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

def pad_dims_like(x: Tensor, other: Tensor) -> Tensor:
    """Pad dimensions of tensor `x` to match the shape of tensor `other`.

    Parameters
    ----------
    x : Tensor
        Tensor to be padded.
    other : Tensor
        Tensor whose shape will be used as reference for padding.

    Returns
    -------
    Tensor
        Padded tensor with the same shape as other.
    """
    ndim = other.ndim - x.ndim
    return x.view(*x.shape, *((1,) * ndim))
#----------------------------------------------------------------------------
# Fully-connected layer.

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        # f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = torch.as_tensor(resample_filter, dtype=torch.float16)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

#----------------------------------------------------------------------------


class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True, resolution=256,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None, swin=False, swin_params=dict(),
    ):
        super().__init__()
        self.attention = attention
        self.swin = swin
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads and swin == False:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)
        elif attention and swin == True:
            # swin_params['in_chans'] = out_channels
            self.swin_layer = BasicLayer(img_size=resolution,
                                         in_chans=out_channels,
                                         norm_layer=normalization,
                                         **swin_params)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads and self.swin == False:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        elif self.attention and self.swin == True:
            x = self.swin_layer(x)
            x = x * self.skip_scale
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------


class SongUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.
        cond_lq             = True,

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16, 32],         # List of resolutions with self-attention. 原值为16，现改为32
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        swin                = False,
        swin_params         = None,
        fourier_scale       = 16,
        scale               = 4,
        use_enc             = False,
        upsampler           = 'nearest',
        res_con             = False,
        use_fp16            = False,
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.use_enc = use_enc
        self.label_dropout = label_dropout
        self.cond_lq = cond_lq
        self.res_con = res_con
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn, swin=swin, swin_params=swin_params
        )
        # print(f'embedding_type:{embedding_type}')
        # print(f'fourier_scale:{fourier_scale}')
        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels, scale=fourier_scale)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.upsampler = upsampler

        if use_enc:
            cout = in_channels * scale ** 2
            # out_channels = out_channels * scale ** 2
            out_channels = model_channels
        else:
            cout = in_channels
        
        if use_enc:
            self.encoder = nn.Sequential(nn.PixelUnshuffle(scale))
            if upsampler == 'pixelshuffle':
                self.decoder = nn.Sequential(nn.PixelShuffle(scale))
                self.conv_last = nn.Conv2d(in_channels, in_channels, 3, 1, 1,)
            elif upsampler == 'nearest':
                self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
                self.conv_up1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1,)
                self.conv_up2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1,)
                self.conv_last = nn.Conv2d(out_channels, in_channels, 3, 1, 1,)
            
            # self.conv_last = nn.Conv2d(out_channels, out_channels, 3, 1, 1,)
        self.scale = scale
        
        
        # Encoder.
        self.enc = torch.nn.ModuleDict()

        
        if cond_lq:
            cout += in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            block_kwargs['resolution'] = res
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, lq=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        # if self.map_label is not None:
        #     tmp = class_labels
        #     if self.training and self.label_dropout:
        #         tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
        #     emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        # if self.map_augment is not None and augment_labels is not None:
        #     emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))
        if self.use_enc:
            x = self.encoder(x)
        if self.cond_lq:
            x = torch.cat([x, lq], dim=1)
        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        if self.use_enc:
            if self.upsampler == 'pixelshuffle':
                aux = self.decoder(aux)
            elif self.upsampler == 'nearest':
                aux = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(aux, scale_factor=2, mode='nearest')))
                aux = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(aux, scale_factor=2, mode='nearest')))
                
            aux = self.conv_last(aux)
        if self.res_con:
            lq_up = F.interpolate(lq, scale_factor=self.scale, mode='bicubic')
            aux = lq_up + aux
        else:
            aux = aux
        return aux
    
class TimeEmbeddingLayer(torch.nn.Module):
    def __init__(self,
                 depth, 
                 swin,
                 unet_block_params,
                 swin_params,
    ):
        super().__init__()
        self.depth = depth
        self.swin = swin
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(UNetBlock(**unet_block_params))
        if swin:
            self.swin_block = BasicLayer(**swin_params)

    def forward(self, x, emb):
        for layer in self.layers:
            x = layer(x, emb)
        if self.swin:
            x = self.swin_block(x)
        
        return x

class SRNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        cond_lq             = False,
        use_enc             = False,
        upsampler           = 'nearest',
        scale               = 4,
        drop_cond_lq        = -1,

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        depths              = [6, 6, 6, 6],
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        skip_scale          = np.sqrt(0.5),
        swin                = False,
        swin_params         = None,
        fourier_scale       = 16,
    ):
        assert embedding_type in ['fourier', 'positional']


        super().__init__()
        self.label_dropout = label_dropout
        self.cond_lq = cond_lq
        self.drop_cond_lq = drop_cond_lq
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=skip_scale, eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # print(f'embedding_type:{embedding_type}')
        # print(f'fourier_scale:{fourier_scale}')

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels, scale=fourier_scale)
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        self.upsampler = upsampler
        self.use_enc = use_enc
        
        
        if use_enc:
            cout = in_channels * scale ** 2
            out_channels = out_channels * scale ** 2
        else:
            cout = in_channels
        
        
        if cond_lq:
            cout += in_channels
        
        self.conv_first = nn.Conv2d(cout, model_channels, 3, 1, 1)
        
        if use_enc:
            self.encoder = nn.Sequential(nn.PixelUnshuffle(scale))
            if upsampler == 'pixelshuffle':
                self.decoder = nn.Sequential(nn.PixelShuffle(scale))
                self.conv_last = nn.Conv2d(in_channels, in_channels, 3, 1, 1,)
            elif upsampler == 'nearest':
                self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
                self.conv_up1 = nn.Conv2d(model_channels, model_channels, 3, 1, 1,)
                self.conv_up2 = nn.Conv2d(model_channels, model_channels, 3, 1, 1,)
                self.conv_last = nn.Conv2d(model_channels, in_channels, 3, 1, 1,)
        else:
            self.conv_last = nn.Conv2d(model_channels, in_channels, 3, 1, 1,)
            # self.conv_last = nn.Conv2d(out_channels, out_channels, 3, 1, 1,)
        
        unet_block_params = dict(
            in_channels=model_channels,
            out_channels=model_channels,
            attention=False,
            **block_kwargs
        )
        if swin:
            swin_params = dict(
                img_size=img_resolution,
                in_chans=model_channels,
                norm_layer=normalization,
                **swin_params
            )
        self.layers = nn.ModuleList()
        for num_layers in depths:
            self.layers.append(TimeEmbeddingLayer(num_layers, swin=swin, unet_block_params=unet_block_params, swin_params=swin_params if swin else None))
        # for num_layers in depths:
        #     for i in range(num_layers):
        #         self.layers.append(UNetBlock(in_channels=model_channels, out_channels=model_channels, attention=False, **block_kwargs))
        #     if swin:
        #         self.layers.append(BasicLayer(img_size=img_resolution,
        #                                       in_chans=model_channels,
        #                                       norm_layer=normalization,
        #                                       **swin_params))
                
        self.scale = scale
        
        

        

        
    def forward(self, x, noise_labels, lq=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        
        if self.training and torch.rand(1) <= self.drop_cond_lq:
            lq = torch.zeros_like(x, device=x.device, dtype=x.dtype)
        else:
            lq = lq

        if self.use_enc:
            x = self.encoder(x)
        
        if self.cond_lq:
            x = torch.cat([x, lq], dim=1)
        
        x = self.conv_first(x)
        for layer in self.layers:
            x = layer(x, emb)
            
        if self.use_enc:
            if self.upsampler == 'pixelshuffle':
                x = self.decoder(x)
            elif self.upsampler == 'nearest':
                x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
                
            x = self.conv_last(x)
        else:
            x = self.conv_last(x)
        
        

        return x


@ARCH_REGISTRY.register()
class EDMUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        # label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'SongUnet',   # Class name of the underlying model.
        up_list         = None,
        down_list       = None,
        use_skip        = True,
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        # self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.up_list = up_list
        self.down_list = down_list
        self.use_skip = use_skip
        self.model_type = model_type
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, **model_kwargs)


    def forward(self, x, sigma, lq, force_fp32=False, **model_kwargs):

        x = x.to(torch.float32)
        x_lr = lq
        # x_cond = torch.cat([x, x_lr], dim=1).to(torch.float32)
        # sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        sigma = sigma.reshape(-1, 1, 1, 1).to(torch.float32)
        # class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        # sigma = sigma.reshape(-1, 1, 1, 1)
        if self.use_skip:
            c_skip = self.sigma_data ** 2 / (((sigma / 0.1)) ** 2 + self.sigma_data ** 2)
            # c_skip = self.sigma_data ** 2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)
            c_out = (sigma / 0.1) / ((sigma / 0.1) ** 2 + self.sigma_data ** 2).sqrt()
            # c_out = (sigma - self.sigma_min) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
            # c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        else:
            c_skip = self.sigma_data ** 2 / (((sigma / 0.001)) ** 2 + self.sigma_data ** 2)
            c_out = (sigma / 0.001) / ((sigma / 0.001) ** 2 + self.sigma_data ** 2).sqrt()
        c_noise = ((sigma + 0.002).log() / 4).to(dtype)
        c_skip.to(dtype)
        c_out.to(dtype)
        
        # c_noise = sigma
        c_in = 1
        ##
        # print(c_noise.dtype)
        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), lq=x_lr, **model_kwargs)
        # F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), lq=x_lr, **model_kwargs)
        # print(F_x.dtype)
        # assert F_x.dtype == dtype
        # D_x = c_skip * x.to(dtype) + c_out * F_x.to(dtype)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        # print(f'x_type:{x.dtype}')
        
        # print((f'Dx_dtype:{D_x.dtype}'))
    
        
        return D_x

    def forward_new(self, x, sigma, lq, force_fp32=False, **model_kwargs):

        x = x.to(torch.float32)
        x_lr = lq
        # x_cond = torch.cat([x, x_lr], dim=1).to(torch.float32)
        # sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        sigma = sigma.reshape(-1, 1, 1, 1).to(torch.float32)
        # class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        # sigma = sigma.reshape(-1, 1, 1, 1)
        if self.use_skip:
            c_skip = self.sigma_data ** 2 / (((sigma / 0.1)) ** 2 + self.sigma_data ** 2)
            # c_skip = self.sigma_data ** 2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data ** 2)
            c_out = (sigma / 0.1) / ((sigma / 0.1) ** 2 + self.sigma_data ** 2).sqrt()
            # c_out = (sigma - self.sigma_min) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
            # c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        else:
            c_skip = self.sigma_data ** 2 / (((sigma / 0.001)) ** 2 + self.sigma_data ** 2)
            c_out = (sigma / 0.001) / ((sigma / 0.001) ** 2 + self.sigma_data ** 2).sqrt()
        c_noise = ((sigma + 0.002).log() / 4)
        # c_skip.to(dtype)
        # c_out.to(dtype)
        
        # c_noise = sigma
        c_in = 1
        ##
        # print(c_noise.dtype)
        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), lq=x_lr, **model_kwargs)
        # F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), lq=x_lr, **model_kwargs)
        # print(F_x.dtype)
        # assert F_x.dtype == dtype
        # D_x = c_skip * x.to(dtype) + c_out * F_x.to(dtype)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        # print(f'x_type:{x.dtype}')
        
        # print((f'Dx_dtype:{D_x.dtype}'))
    
        
        return D_x
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------