# =====================================================================
# KurdFake - channel_test.py   (shortcut test: is the detector judging the SPEECH or the RAIL CHANNEL?)
#
# Problem: every RAIL bona fide clip carries the RAIL field channel (50 Hz mains hum, noise floor,
# level, frequent clipping), while every spoof is clean synthetic audio at -16 LUFS. A detector can then
# reach ~0% EER on RAIL simply by recognising the channel, without learning anything about synthesis.
#
# Test ("channel transplant"): give each RAIL test spoof the channel of a REAL recording of the same
# target speaker, then score again. For every spoof we take one of that speaker's ENROLLMENT clips
# (never a test trial), keep its real leading/trailing background (hum + noise), and replace its speech
# region by: the spoof, scaled to the same speech RMS as the real speech + that clip's own background
# noise looped underneath; finally hard-clip at full scale exactly as the recorder does.
# Bona fide test clips are left untouched. Only the spoofs change, so any rise in EER = channel shortcut.
#
#   python channel_test.py                 (fine-tuned E1 model + zero-shot multilingual model, about 20 min)
# v4 recipe (23 Nov, click-free). ABLATION version: which part of the channel is the shortcut? Each condition adds ONE component.
# Writes: results\channel_ablation.txt
# =====================================================================
import os, sys, csv, json, wave, time, zlib, importlib.util, platform
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np
import torch

ROOT = KF_ROOT
DATA = os.path.join(ROOT, "Dataset")
RAIL = os.path.join(DATA, "RAIL_clean")
RES = os.path.join(ROOT, "results")
E1_CKPT = os.path.join(RES, "ft", "AASIST_raw_train-both_b100_s1", "best.pth")
ADF_CKPT = os.path.join(ROOT, "models", "antideepfake_mms300m", "model.safetensors")
AASIST = os.path.join(ROOT, "models", "aasist")
SPOOF_MANIFESTS = [os.path.join(DATA, "Spoofs", f) for f in ["spoof_manifest_knnvc_final.csv", "spoof_manifest_knnvc_D2.csv",
                                                             "spoof_manifest_tts_vekol.csv", "spoof_manifest_f5.csv"]]
SR, NB_SAMP, PAD_S = 16000, 64600, 0.05
TRAIN_SYSTEMS = {"D", "G", "C1", "C1D"}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kf_metrics import eer
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"python {platform.python_version()}  torch {torch.__version__}  device {dev}")

def load(path):
    with wave.open(path, "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return x.mean(axis=1).astype(np.float32) / 32768.0

def save(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 32767 / 32768) * 32768).astype(np.int16).tobytes())

# ---------------- data ----------------
bona, enroll = [], {}
with open(os.path.join(RAIL, "manifest.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["split"] != "test":
            continue
        if r["role"] == "enroll":
            enroll.setdefault(r["speaker"], []).append(r)
        else:
            bona.append(r)
spoofs = []
for mp in SPOOF_MANIFESTS:
    with open(mp, encoding="utf-8") as fh:
        spoofs += [r for r in csv.DictReader(fh) if r["dataset"] == "rail" and r["split"] == "test"]
print(f"RAIL test: bona fide {len(bona)}  spoofs {len(spoofs)}  enrollment references {sum(len(v) for v in enroll.values())}")

ref_cache = {}
def reference(r):
    if r["utt_id"] not in ref_cache:
        x = load(os.path.join(RAIL, r["clean_path"]))
        on, off = int(max(0.0, float(r["onset_s"]) - PAD_S) * SR), int(min(len(x) / SR, float(r["offset_s"]) + PAD_S) * SR)
        bg = np.concatenate([x[:on], x[off:]])
        sp = x[on:off]
        ref_cache[r["utt_id"]] = (x, on, off, bg, float(np.sqrt(np.mean(sp ** 2)) + 1e-9))
    return ref_cache[r["utt_id"]]

def hp(x, fc=100):
    X = np.fft.rfft(x); X[np.fft.rfftfreq(len(x), 1 / SR) < fc] = 0
    return np.fft.irfft(X, n=len(x)).astype(np.float32)

XF = int(0.02 * SR)
def xfade(a, b, f):
    f = min(f, len(a), len(b))
    if f < 2:
        return np.concatenate([a, b])
    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f))
    return np.concatenate([a[:-f], a[-f:] * (1 - w) + b[:f] * w, b[f:]])

def loop_noise(bg, n, rng):
    y = bg.copy()
    while len(y) < n + len(bg):
        y = xfade(y, bg, min(XF, len(bg) // 4))
    k = int(rng.integers(0, len(y) - n + 1)); return y[k:k + n]

def transplant(s, r, rng, mode):
    """v4 recipe (continuous real background under the whole file, spoof added on top; no cut-and-paste)."""
    x, on, off, bg, rms_ref = reference(r)
    sp = x[on:off]; e = int(0.02 * SR)
    bgv = xfade(x[e:on], x[off:len(x) - e], XF) if on > e and len(x) - e > off else bg
    if mode == "full":
        g = min(rms_ref / (np.sqrt(np.mean(s ** 2)) + 1e-9), 0.98 * float(np.max(np.abs(sp))) / (np.max(np.abs(s)) + 1e-9))
        s = s * g
    elif mode in ("level", "level+clip"):
        s = s * (rms_ref / (np.sqrt(np.mean(s ** 2)) + 1e-9))      # level only (no peak cap, so clipping can happen)
    if mode in ("level", "level+clip"):
        y = s
    else:
        n = on + len(s) + (len(x) - off)
        y = np.zeros(n, dtype=np.float32); y[on:on + len(s)] = s
        if len(bgv) >= int(0.1 * SR):
            b = bgv if mode == "full" else (hp(bgv) if mode == "noise" else bgv - hp(bgv))
            y = y + loop_noise(b, n, rng)
        f = int(0.005 * SR); w = np.linspace(0, 1, f); y[:f] *= w; y[-f:] *= w[::-1]
    if mode in ("level+clip", "full"):
        y = np.clip(y, -1.0, 32767 / 32768)
    return y.astype(np.float32)

MODES = ["hum", "noise", "level", "level+clip", "full"]
ref_of = []
for r in spoofs:
    rng = np.random.default_rng(zlib.crc32(r["utt_id"].encode())); refs = enroll.get(r["target_speaker"])
    ref_of.append((refs[int(rng.integers(0, len(refs)))], zlib.crc32(r["utt_id"].encode())))
spoof_x = [load(os.path.join(DATA, "Spoofs", r["path"])) for r in spoofs]
t0 = time.time()
bona_x = [load(os.path.join(RAIL, r["clean_path"])) for r in bona]
orig_x = None

# ---------------- models ----------------
def aasist_scorer():
    spec = importlib.util.spec_from_file_location("aasist_model", os.path.join(AASIST, "models", "AASIST.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    m = mod.Model(json.load(open(os.path.join(AASIST, "config", "AASIST.conf")))["model_config"])
    m.load_state_dict(torch.load(E1_CKPT, map_location="cpu", weights_only=True)); m = m.to(dev).eval()
    def pad(x):
        return x[:NB_SAMP] if len(x) >= NB_SAMP else np.tile(x, int(NB_SAMP / len(x)) + 1)[:NB_SAMP]
    def f(xs):
        out = []
        with torch.no_grad():
            for b in range(0, len(xs), 32):
                _, o = m(torch.from_numpy(np.stack([pad(x) for x in xs[b:b + 32]])).to(dev)); out += o[:, 1].float().cpu().tolist()
        return np.array(out)
    return f

def adf_scorer():
    from kf_antideepfake import load_antideepfake, score_batch
    m, _ = load_antideepfake(ADF_CKPT, dev)
    def f(xs):
        out = []
        for b in range(0, len(xs), 64):
            out += score_batch(m, xs[b:b + 64], dev)
        return np.array(out)
    return f

systems = sorted({r["system"] for r in spoofs})
sysarr = np.array([r["system"] for r in spoofs])
L = [f"KurdFake channel ablation (RAIL test, fine-tuned AASIST E1)  {time.strftime('%Y-%m-%d %H:%M')}",
     "Each condition gives the spoofs ONE part of the real channel of an enrollment clip of the same speaker.",
     "hum = <100 Hz background only; noise = >100 Hz background only; level = speech RMS only; level+clip = level then full-scale clipping; full = all.", ""]
sc = aasist_scorer(); sb = sc(bona_x); so = sc(spoof_x)
L.append("condition    " + "  ".join(f"{s:>5}" for s in systems) + "   pooled   mean spoof score (bona fide %.2f)" % sb.mean())
L.append("original     " + "  ".join(f"{100*eer(sb, so[sysarr==s])[0]:5.1f}" for s in systems) + f"   {100*eer(sb, so)[0]:5.1f}    {so.mean():.2f}")
for mode in MODES:
    xs = [transplant(x, rf, np.random.default_rng(seed), mode) for x, (rf, seed) in zip(spoof_x, ref_of)]
    st = sc(xs)
    L.append(f"{mode:12s} " + "  ".join(f"{100*eer(sb, st[sysarr==s])[0]:5.1f}" for s in systems) + f"   {100*eer(sb, st)[0]:5.1f}    {st.mean():.2f}")
    print(L[-1], flush=True)
txt = "\n".join(L)
open(os.path.join(RES, "channel_ablation_v4.txt"), "w", encoding="utf-8").write(txt + "\n")
print("\n" + txt + f"\n\nSaved to {RES}\\channel_ablation_v4.txt   ({(time.time()-t0)/60:.1f} min)")
