# =====================================================================
# KurdFake - kf_antideepfake.py
# MMS-300M-AntiDeepfake (Ge, Wang, Liu & Yamagishi, 2025; github.com/nii-yamagishilab/AntiDeepfake,
# checkpoint CC BY-NC-SA 4.0) re-implemented in plain PyTorch, so it runs WITHOUT fairseq on Windows.
#   * The network below is the fairseq wav2vec 2.0 "layer_norm_first" architecture (as in models/W2V.py
#     and W2V_configs['mms_300m'] of the official repo), with IDENTICAL parameter names, so the official
#     model.safetensors loads with strict name and shape checking: no renaming, no guessing.
#   * Head (official): mean over time of the last-layer features -> Linear(1024, 2). Index 1 = real.
#   * Input (official test pipeline): whole utterance at 16 kHz, per-utterance layer_norm of the
#     waveform, no cropping or padding. Score = logit_real - logit_fake (= log-odds of 'real').
# Python 3.8 + torch 2.x; needs the 'safetensors' package OR falls back to a pure-Python reader.
# =====================================================================
import json, struct, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONV = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
D, FFN, HEADS, LAYERS, POS_K, POS_G = 1024, 4096, 16, 24, 128, 16


class _T(nn.Module):                                  # fairseq TransposeLast
    def forward(self, x):
        return x.transpose(-2, -1)


class FeatureExtractor(nn.Module):                    # fairseq ConvFeatureExtractionModel, mode="layer_norm"
    def __init__(self):
        super().__init__()
        layers, cin = [], 1
        for (c, k, s) in CONV:
            layers.append(nn.Sequential(nn.Conv1d(cin, c, k, stride=s, bias=True), nn.Dropout(0.0),
                                        nn.Sequential(_T(), nn.LayerNorm(c, elementwise_affine=True), _T()), nn.GELU()))
            cin = c
        self.conv_layers = nn.ModuleList(layers)

    def forward(self, x):                             # (B, T) -> (B, 512, frames)
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


class WNConv(nn.Module):                              # Conv1d with weight norm over dim=2 (fairseq pos_conv)
    def __init__(self):
        super().__init__()
        self.weight_g = nn.Parameter(torch.ones(1, 1, POS_K))
        self.weight_v = nn.Parameter(torch.randn(D, D // POS_G, POS_K) * 0.01)
        self.bias = nn.Parameter(torch.zeros(D))

    def forward(self, x):
        w = self.weight_v * (self.weight_g / self.weight_v.norm(dim=(0, 1), keepdim=True))
        return F.conv1d(x, w, self.bias, padding=POS_K // 2, groups=POS_G)


class MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(D, D) for _ in range(4))

    def forward(self, x):                             # (B, T, D)
        B, T, _ = x.shape
        sh = lambda t: t.view(B, T, HEADS, D // HEADS).transpose(1, 2)
        o = F.scaled_dot_product_attention(sh(self.q_proj(x)), sh(self.k_proj(x)), sh(self.v_proj(x)))
        return self.out_proj(o.transpose(1, 2).reshape(B, T, D))


class Layer(nn.Module):                               # fairseq TransformerSentenceEncoderLayer, layer_norm_first=True
    def __init__(self):
        super().__init__()
        self.self_attn = MHA()
        self.self_attn_layer_norm = nn.LayerNorm(D)
        self.fc1, self.fc2 = nn.Linear(D, FFN), nn.Linear(FFN, D)
        self.final_layer_norm = nn.LayerNorm(D)

    def forward(self, x):
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_conv = nn.ModuleList([WNConv()])     # name 'pos_conv.0.*' as in fairseq
        self.layers = nn.ModuleList([Layer() for _ in range(LAYERS)])
        self.layer_norm = nn.LayerNorm(D)

    def forward(self, x):                             # (B, T, D)
        pc = F.gelu(self.pos_conv[0](x.transpose(1, 2))[:, :, :-1]).transpose(1, 2)
        x = x + pc
        for l in self.layers:
            x = l(x)
        return self.layer_norm(x)


class W2V(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.layer_norm = nn.LayerNorm(512)
        self.post_extract_proj = nn.Linear(512, D)
        self.encoder = Encoder()

    def forward(self, wav):                           # features_only=True, mask=False
        f = self.feature_extractor(wav).transpose(1, 2)
        return self.encoder(self.post_extract_proj(self.layer_norm(f)))


class _SSL(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = W2V()


class AntiDeepfake(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_ssl = _SSL()
        self.proj_fc = nn.Linear(D, 2)

    def forward(self, wav):                           # wav: (B, T) already layer-normed; equal lengths in a batch
        emb = self.m_ssl.model(wav)
        return self.proj_fc(emb.mean(dim=1))          # AdaptiveAvgPool1d(1) over time == mean


UNUSED = ("m_ssl.model.final_proj.", "m_ssl.model.project_q.", "m_ssl.model.quantizer.", "m_ssl.model.mask_emb")


def read_safetensors(path):
    try:
        from safetensors.torch import load_file
        return load_file(path)
    except ImportError:
        pass
    out = {}
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n)); base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            assert v["dtype"] == "F32", v["dtype"]
            s, e = v["data_offsets"]; fh.seek(base + s)
            out[k] = torch.from_numpy(np.frombuffer(fh.read(e - s), dtype="<f4").copy()).view(*v["shape"])
    return out


def load_antideepfake(path, device="cpu"):
    sd = read_safetensors(path)
    sd = {k: v for k, v in sd.items() if not k.startswith(UNUSED)}
    model = AntiDeepfake()
    own = model.state_dict()
    missing = sorted(set(own) - set(sd)); extra = sorted(set(sd) - set(own))
    bad = [k for k in own if k in sd and tuple(own[k].shape) != tuple(sd[k].shape)]
    if missing or extra or bad:
        raise RuntimeError(f"checkpoint mismatch: missing {missing[:5]} extra {extra[:5]} shape {bad[:5]}")
    model.load_state_dict(sd, strict=True)
    n = sum(p.numel() for p in model.parameters())
    return model.to(device).eval(), n


def prep(x):
    """official test pipeline: per-utterance layer_norm of the 16 kHz waveform (float32 numpy, 1-D)."""
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    return F.layer_norm(t, t.shape)


def score_batch(model, wavs, device, max_batch_seconds=60.0):
    """Scores a list of 1-D float32 waveforms one length-group at a time (no padding, like batch_size=1)."""
    out = [None] * len(wavs)
    order = sorted(range(len(wavs)), key=lambda i: len(wavs[i]))
    i = 0
    with torch.no_grad():
        while i < len(order):
            L = len(wavs[order[i]]); j = i
            while j < len(order) and len(wavs[order[j]]) == L and (j - i + 1) * L <= max_batch_seconds * 16000:
                j += 1
            j = max(j, i + 1)
            x = torch.stack([prep(wavs[k]) for k in order[i:j]]).to(device)
            lg = model(x).float().cpu()
            for k, s in zip(order[i:j], (lg[:, 1] - lg[:, 0]).tolist()):
                out[k] = s
            i = j
    return out
