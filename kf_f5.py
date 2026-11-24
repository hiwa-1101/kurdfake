# =====================================================================
# KurdFake - kf_f5.py : run the Central Kurdish F5-TTS model (aranemini/central-kurdish-tts)
# inside the existing Python 3.8 environment (gpu_env), fully offline.
#
# It reproduces the authors' infer.py (same asosoft normalisation + G2P, same f5_tts inference
# functions and default settings) but adds four small compatibility shims, because f5-tts targets
# Python >= 3.9:
#   1. importlib.resources.files  -> importlib_resources backport
#   2. f5_tts.model.trainer / dataset (training only, not Python-3.8 compatible) -> empty stubs
#   3. rjieba / librosa (Chinese segmentation, BigVGAN mel) -> tiny stubs, never used for Kurdish+Vocos
#   4. transformers.pipeline (only for automatic ASR of a reference without text) -> stub
# Usage:
#   import kf_f5
#   tts = kf_f5.F5Kurdish()                      # loads model + Vocos once
#   wav24k = tts.synth("دەقی کوردی", ref_wav_path, ref_text)
#   python kf_f5.py --selftest                   # synthesises the authors' own example
# =====================================================================
import os, sys, re, types, importlib, argparse
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README

F5_DIR    = os.path.join(KF_ROOT, "models", "f5_ckb")
VOCOS_DIR = os.path.join(KF_ROOT, "models", "vocos")
CKPT      = "model-studio-male.pt"

# ---------------- shims (must run before importing f5_tts) ----------------
import importlib.resources as _ir
if not hasattr(_ir, "files"):
    import importlib_resources
    _ir.files = importlib_resources.files

def _stub(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m

_stub("f5_tts.model.trainer", Trainer=None)
_stub("f5_tts.model.dataset", DynamicBatchSampler=None, collate_fn=None, load_dataset=None)
try:
    import rjieba  # noqa
except ImportError:
    _stub("rjieba", cut=lambda text: [text])
try:
    import librosa.filters  # noqa
except ImportError:
    def _no_librosa(*a, **k):
        raise RuntimeError("librosa mel is only needed for BigVGAN, not for Vocos")
    lib = _stub("librosa"); lib.filters = _stub("librosa.filters", mel=_no_librosa)
try:
    import matplotlib  # noqa
except ImportError:
    mpl = _stub("matplotlib", use=lambda *a, **k: None); mpl.pylab = _stub("matplotlib.pylab")
try:
    from transformers import pipeline  # noqa
except Exception:
    def _no_pipeline(*a, **k):
        raise RuntimeError("automatic ASR of reference audio is disabled - always pass ref_text")
    _stub("transformers", pipeline=_no_pipeline)

import numpy as np
import torch
import yaml
import asosoft
from f5_tts.infer.utils_infer import (infer_process, load_model, load_vocoder, preprocess_ref_audio_text,
                                      cfg_strength, cross_fade_duration, device, nfe_step,
                                      sway_sampling_coef, speed, target_rms, fix_duration)
import f5_tts.model as f5m

# ---------------- text front-end: identical to the authors' infer.py ----------------
# The three replacement pairs below are the authors' (their file showed them mis-encoded in the
# Windows console); they map Latin phoneme variants that asosoft emits onto the model's vocabulary.
REPLACEMENTS = {"\u1e8d": "\u03c7", "\u01b9": "\u00bf", "\u1e27": "\u1e25"}   # ẍ->χ, ƹ->¿, ḧ->ḥ (decoded from the authors file)

def normalize_and_g2p(text):
    text = re.sub(r"(\d{1,8})\s*[-–]\s*(\d{1,8})", r"\1 تا \2", text)
    text = re.sub(r"\b\d{9,}\b", lambda m: f"<NUM:{m.group(0)}>", text)
    norm = asosoft.Normalize(text, changeInitialR=True, deepUnicodeCorrectios=True, additionalUnicodeCorrections=True)
    norm = asosoft.NormalizePunctuations(norm, seprateAllPunctuations=True)
    try:
        norm = asosoft.Number2Word(norm)
    except Exception as e:
        print(f"[Number2Word Error] Skipping conversion: {e}")
    g2p = asosoft.KurdishG2P(norm).replace("ˈ", "")
    for old, new in REPLACEMENTS.items():
        g2p = g2p.replace(old, new)
    return g2p

def slim_checkpoint(path):
    """The released .pt (5.4 GB) is a full training checkpoint. Keep only the EMA weights (~1.3 GB)."""
    slim = path.replace(".pt", "_ema.pt")
    if os.path.exists(slim):
        return slim
    print(f"creating {os.path.basename(slim)} (EMA weights only) - one time ...")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    torch.save({"ema_model_state_dict": ck["ema_model_state_dict"]}, slim)
    del ck
    return slim

class F5Kurdish:
    def __init__(self, f5_dir=F5_DIR, vocos_dir=VOCOS_DIR, ckpt=CKPT):
        cfg = yaml.safe_load(open(os.path.join(f5_dir, "F5TTS_v1_Base.yaml"), encoding="utf-8"))
        model_cls = getattr(f5m, cfg["model"]["backbone"])
        self.vocoder = load_vocoder(vocoder_name="vocos", is_local=True, local_path=vocos_dir, device=device)
        self.model = load_model(model_cls, cfg["model"]["arch"], slim_checkpoint(os.path.join(f5_dir, ckpt)),
                                mel_spec_type="vocos", vocab_file=os.path.join(f5_dir, "vocab.txt"), device=device)
        self.sr = 24000
        self._ref_cache = {}

    def synth(self, text, ref_wav, ref_text, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        key = (ref_wav, ref_text)
        if key not in self._ref_cache:
            self._ref_cache[key] = preprocess_ref_audio_text(ref_wav, normalize_and_g2p(ref_text))
        ra, rt = self._ref_cache[key]
        wav, sr, _ = infer_process(ra, rt, normalize_and_g2p(text), self.model, self.vocoder, mel_spec_type="vocos",
                                   target_rms=target_rms, cross_fade_duration=cross_fade_duration, nfe_step=nfe_step,
                                   cfg_strength=cfg_strength, sway_sampling_coef=sway_sampling_coef, speed=speed,
                                   fix_duration=fix_duration, device=device, show_info=lambda *a, **k: None)
        self.sr = sr
        return np.asarray(wav, dtype=np.float32)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    import soundfile as sf
    tts = F5Kurdish()
    text = open(os.path.join(F5_DIR, "input_text.txt"), encoding="utf-8").read().strip()
    ref_t = open(os.path.join(F5_DIR, "prompt-studio-male.txt"), encoding="utf-8").read().strip()
    print("G2P:", normalize_and_g2p(text)[:200])
    y = tts.synth(text, os.path.join(F5_DIR, "prompt-studio-male.wav"), ref_t, seed=1337)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_f5.wav")
    sf.write(out, y, tts.sr)
    print(f"wrote {out}  ({len(y)/tts.sr:.1f} s)  - compare with models\\f5_ckb\\output-studio-male.wav")
