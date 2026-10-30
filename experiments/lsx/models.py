"""LeafSpike-S (SDT-v3 E-SpikeFormer + max-pool shortcuts + time-aligned attention) and ViT baselines.

The LeafSpike code is an independent implementation; its module names match the official SDT-v3 models.py
(https://github.com/BICLab/Spike-Driven-Transformer-V3) key for key, so the ImageNet checkpoint loads 1:1.
"""
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeMonitor:
    def __init__(self):
        self.collect_stats = False
        self.collect_reg = False
        self.stats = defaultdict(lambda: [0.0, 0.0, 0.0])   # name -> [sum counts, nonzero, numel]
        self.attn_ops = 0.0       # estimated synaptic ops in attention matmuls (per sample, summed over batches)
        self.attn_dense = 0.0
        self.samples = 0
        self.reg_terms = []
    def reset(self):
        self.stats.clear(); self.attn_ops = 0.0; self.attn_dense = 0.0; self.samples = 0; self.reg_terms = []

MON = SpikeMonitor()

class MultiSpike(nn.Module):
    # I-LIF / MultiSpike: integer spike count round(clamp(x, 0, D)), straight-through gradient inside [0, D].
    def __init__(self, min_value=0, max_value=4, Norm=None):
        super().__init__()
        self.min_value, self.max_value = min_value, max_value
        self.Norm = max_value if Norm is None else Norm
        self.tag = ''
    def forward(self, x):
        xc = x.clamp(self.min_value, self.max_value)
        q = xc + (torch.round(xc) - xc).detach()
        if MON.collect_reg:
            MON.reg_terms.append(q.float().mean() / self.max_value)
        if MON.collect_stats:
            with torch.no_grad():
                s = MON.stats[self.tag]
                s[0] += q.float().sum().item(); s[1] += (q > 0).sum().item(); s[2] += q.numel()
        return q / self.Norm
    def extra_repr(self):
        return f'max={self.max_value}, Norm={self.Norm}'

class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__(); self.p = p
    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep

class SepConv_Spike(nn.Module):
    def __init__(self, dim, expansion_ratio=2, kernel_size=7, padding=3, D=4):
        super().__init__()
        med = int(expansion_ratio * dim)
        self.spike1 = MultiSpike(max_value=D)
        self.pwconv1 = nn.Sequential(nn.Conv2d(dim, med, 1, 1, bias=False), nn.BatchNorm2d(med))
        self.spike2 = MultiSpike(max_value=D)
        self.dwconv = nn.Sequential(nn.Conv2d(med, med, kernel_size, padding=padding, groups=med, bias=False),
                                    nn.BatchNorm2d(med))
        self.spike3 = MultiSpike(max_value=D)
        self.pwconv2 = nn.Sequential(nn.Conv2d(med, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))
    def forward(self, x):
        x = self.pwconv1(self.spike1(x))
        x = self.dwconv(self.spike2(x))
        return self.pwconv2(self.spike3(x))

class MS_ConvBlock_spike_SepConv(nn.Module):
    def __init__(self, dim, mlp_ratio=4, D=4):
        super().__init__()
        self.Conv = SepConv_Spike(dim=dim, D=D)
        self.mlp_ratio = mlp_ratio
        self.spike1 = MultiSpike(max_value=D)
        self.conv1 = nn.Conv2d(dim, dim * mlp_ratio, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)
        self.spike2 = MultiSpike(max_value=D)
        self.conv2 = nn.Conv2d(dim * mlp_ratio, dim, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
    def forward(self, x):
        x = self.Conv(x) + x
        x_feat = x
        x = self.bn1(self.conv1(self.spike1(x)))
        x = self.bn2(self.conv2(self.spike2(x)))
        return x_feat + x

class MS_MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, D=4):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, 1, 1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = MultiSpike(max_value=D)
        self.fc2_conv = nn.Conv1d(hidden_features, out_features, 1, 1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_spike = MultiSpike(max_value=D)
        self.c_hidden, self.c_output = hidden_features, out_features
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2)
        x = self.fc1_bn(self.fc1_conv(self.fc1_spike(x)))
        x = self.fc2_bn(self.fc2_conv(self.fc2_spike(x)))
        return x.reshape(B, C, H, W).contiguous()

class MS_Attention_linear(nn.Module):
    def __init__(self, dim, num_heads=8, lamda_ratio=4, D=4, mode='aligned'):
        super().__init__()
        assert dim % num_heads == 0
        self.dim, self.num_heads, self.lamda_ratio, self.D, self.mode = dim, num_heads, lamda_ratio, D, mode
        self.scale = (dim // num_heads) ** -0.5
        self.head_spike = MultiSpike(max_value=D)
        self.q_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))
        self.q_spike = MultiSpike(max_value=D)
        self.k_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))
        self.k_spike = MultiSpike(max_value=D)
        cv = int(dim * lamda_ratio)
        self.v_conv = nn.Sequential(nn.Conv2d(dim, cv, 1, 1, bias=False), nn.BatchNorm2d(cv))
        self.v_spike = MultiSpike(max_value=D)
        self.attn_spike = MultiSpike(max_value=D)
        self.proj_conv = nn.Sequential(nn.Conv2d(cv, dim, 1, 1, bias=False), nn.BatchNorm2d(dim))

    def _heads(self, t, B, N, c):
        return t.flatten(2).transpose(-1, -2).reshape(B, N, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

    def forward(self, x):
        B, C, H, W = x.shape
        N, Cv = H * W, int(C * self.lamda_ratio)
        x = self.head_spike(x)
        q = self._heads(self.q_spike(self.q_conv(x)), B, N, C)
        k = self._heads(self.k_spike(self.k_conv(x)), B, N, C)
        v = self._heads(self.v_spike(self.v_conv(x)), B, N, Cv)
        if self.mode == 'integer':
            out = q @ (k.transpose(-2, -1) @ v)            # == (q @ k^T) @ v, linear order
        else:
            D = self.D
            qc, kc, vc = q * D, k * D, v * D                  # integer counts (STE gradients pass through)
            out = 0
            for d in range(D):                                # thermometer planes: spike at step d iff count > d
                qd, kd, vd = (qc - d).clamp(0, 1), (kc - d).clamp(0, 1), (vc - d).clamp(0, 1)
                out = out + qd @ (kd.transpose(-2, -1) @ vd)
            out = out / float(D ** 3)
        if MON.collect_stats:
            with torch.no_grad():
                dh, dv = C // self.num_heads, Cv // self.num_heads
                dense = 2.0 * self.num_heads * N * dh * dv
                MON.attn_dense += dense * B
                MON.attn_ops += 0.5 * dense * (k.float().mean().item() + q.float().mean().item()) * self.D * B
        out = out * (self.scale * 2)
        out = out.transpose(2, 3).reshape(B, Cv, N).contiguous()
        out = self.attn_spike(out).reshape(B, Cv, H, W)
        return self.proj_conv(out).reshape(B, C, H, W)

class MS_Block_Spike_SepConv(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, drop_path=0.0, D=4, attn_mode='aligned'):
        super().__init__()
        self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1, D=D)
        self.attn = MS_Attention_linear(dim, num_heads=num_heads, lamda_ratio=4, D=D, mode=attn_mode)
        self.drop_path = DropPath(drop_path)
        self.mlp = MS_MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), D=D)
    def forward(self, x):
        x = x + self.drop_path(self.conv(x))
        x = x + self.drop_path(self.attn(x))
        return x + self.drop_path(self.mlp(x))

class MS_DownSampling(nn.Module):
    def __init__(self, in_channels, embed_dims, kernel_size=3, stride=2, padding=1, first_layer=True,
                 D=4, maxpool_shortcut=False):
        super().__init__()
        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size, stride, padding)
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = MultiSpike(max_value=D)
        self.use_max_short = maxpool_shortcut and not first_layer and stride == 2
        if self.use_max_short:
            self.max_short = nn.Sequential(nn.MaxPool2d(3, 2, 1), nn.Conv2d(in_channels, embed_dims, 1, bias=False),
                                           nn.BatchNorm2d(embed_dims))
            nn.init.zeros_(self.max_short[2].weight)   # gamma = 0 -> exact pretrained function at init
    def forward(self, x):
        if hasattr(self, 'encode_spike'):
            x = self.encode_spike(x)
        y = self.encode_bn(self.encode_conv(x))
        if self.use_max_short:
            y = y + self.max_short(x)
        return y

class LeafSpike(nn.Module):
    # Same structure / names as SDT-v3 Spiking_vit_MetaFormer_Spike_SepConv
    def __init__(self, embed_dim=(32, 64, 128, 192), num_heads=8, mlp_ratios=4, in_channels=3, num_classes=10,
                 drop_path_rate=0.0, D=4, attn_mode='aligned', maxpool_shortcut=True):
        super().__init__()
        e = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 8)]
        kw = dict(D=D, maxpool_shortcut=maxpool_shortcut)
        self.downsample1_1 = MS_DownSampling(in_channels, e[0] // 2, 7, 2, 3, first_layer=True, **kw)
        self.ConvBlock1_1 = nn.ModuleList([MS_ConvBlock_spike_SepConv(e[0] // 2, mlp_ratios, D)])
        self.downsample1_2 = MS_DownSampling(e[0] // 2, e[0], 3, 2, 1, first_layer=False, **kw)
        self.ConvBlock1_2 = nn.ModuleList([MS_ConvBlock_spike_SepConv(e[0], mlp_ratios, D)])
        self.downsample2 = MS_DownSampling(e[0], e[1], 3, 2, 1, first_layer=False, **kw)
        self.ConvBlock2_1 = nn.ModuleList([MS_ConvBlock_spike_SepConv(e[1], mlp_ratios, D)])
        self.ConvBlock2_2 = nn.ModuleList([MS_ConvBlock_spike_SepConv(e[1], mlp_ratios, D)])
        self.downsample3 = MS_DownSampling(e[1], e[2], 3, 2, 1, first_layer=False, **kw)
        self.block3 = nn.ModuleList([MS_Block_Spike_SepConv(e[2], num_heads, mlp_ratios, dpr[j], D, attn_mode)
                                     for j in range(6)])
        self.downsample4 = MS_DownSampling(e[2], e[3], 3, 1, 1, first_layer=False, **kw)
        self.block4 = nn.ModuleList([MS_Block_Spike_SepConv(e[3], num_heads, mlp_ratios, dpr[6 + j], D, attn_mode)
                                     for j in range(2)])
        self.head = nn.Linear(e[3], num_classes)
        self.spike = MultiSpike(max_value=D, Norm=1)
        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, MultiSpike):
                m.tag = name
        self.downsample1_1.encode_conv._is_stem = True
        self.head._in_norm = 1.0

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def set_attn_mode(self, mode):
        for m in self.modules():
            if isinstance(m, MS_Attention_linear):
                m.mode = mode

    def set_max_short(self, on):
        for m in self.modules():
            if isinstance(m, MS_DownSampling) and hasattr(m, 'max_short'):
                m.use_max_short = on

    def forward_features(self, x):
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1: x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2: x = blk(x)
        x = self.downsample2(x)
        for blk in self.ConvBlock2_1: x = blk(x)
        for blk in self.ConvBlock2_2: x = blk(x)
        x = self.downsample3(x)
        for blk in self.block3: x = blk(x)
        x = self.downsample4(x)
        for blk in self.block4: x = blk(x)
        return x

    def forward(self, x):
        x = self.forward_features(x).flatten(2).mean(2)
        return self.head(self.spike(x))

SIZES = {'t': (24, 48, 96, 128), 's': (32, 64, 128, 192), 'm': (48, 96, 192, 240), 'l': (64, 128, 256, 360)}
CKPT_DRIVE_IDS = {'s': '1LMkOTPehDNpQE79bvB7jFTf6UzDjpAHQ', 'm': '1pHrampLjyE1kLr-4DS1WgSdnCVPzL6Tq',
                  'l': '1pSGCOzrZNgHDxQXAp-Uelx61snIbQC1H'}




def build_leafspike(num_classes, size='s', D=4, attn_mode='aligned', maxpool_shortcut=True, drop_path=0.1):
    return LeafSpike(SIZES[size], num_classes=num_classes, drop_path_rate=drop_path, D=D,
                     attn_mode=attn_mode, maxpool_shortcut=maxpool_shortcut)


def torch_load_any(p):
    try:
        return torch.load(p, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(p, map_location='cpu')


def load_sdt_checkpoint(model, path):
    """Load SDT-v3 ImageNet weights (all tensors whose name and shape match; head and new layers skipped).
    Returns the fraction of backbone tensors loaded."""
    ck = torch_load_any(path)
    if isinstance(ck, dict):
        for k in ('model', 'state_dict', 'model_ema'):
            if k in ck and isinstance(ck[k], dict):
                ck = ck[k]
                break
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in ck.items() if torch.is_tensor(v)}
    msd = model.state_dict()
    backbone = [k for k in msd if not k.startswith('head.') and 'max_short' not in k]
    ok = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape and not k.startswith('head.')}
    frac = len([k for k in backbone if k in ok]) / max(1, len(backbone))
    if frac >= 0.9:
        model.load_state_dict(ok, strict=False)
    return frac


# ---------------------------------------------------------------- ViT baselines (timm)

VIT_ARCHS = {
    'deit_s': 'deit_small_patch16_224',
    'vit_s': 'vit_small_patch16_224',
    'swin_t': 'swin_tiny_patch4_window7_224',
    'mobilevit_xs': 'mobilevit_xs',
    'mobilevit_xxs': 'mobilevit_xxs',
}


def build_vit(arch, num_classes, drop_path=0.1, pretrained=True):
    """Returns (model, pretrained_loaded, mean, std)."""
    import timm
    name = VIT_ARCHS.get(arch, arch)
    loaded = False
    try:
        m = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path)
        loaded = pretrained
    except Exception as e:
        print(f'   [warn] {name}: pretrained weights unavailable ({type(e).__name__}: {e}); training from scratch')
        m = timm.create_model(name, pretrained=False, num_classes=num_classes, drop_path_rate=drop_path)
    cfg = m.pretrained_cfg
    return m, loaded, tuple(cfg.get('mean', IMAGENET_MEAN)), tuple(cfg.get('std', IMAGENET_STD))


IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def new_param_prefixes(model):
    """Names of freshly initialised classifier parameters (get a higher learning rate)."""
    import timm  # noqa: F401
    if isinstance(model, LeafSpike):
        return ['head.', 'max_short']
    name = model.default_cfg.get('classifier', 'head') if hasattr(model, 'default_cfg') else 'head'
    return [name + '.']
