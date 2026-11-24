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
# v4 (23 Nov): uses the click-free channel-matched files in Dataset\Spoofs_CM (make_channel_matched.py v4).
# Writes: results\channel_test_v4.txt
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

def transplant(s, r, rng):
    x, on, off, bg, rms_ref = reference(r)
    s = s * (rms_ref / (np.sqrt(np.mean(s ** 2)) + 1e-9))          # same speech level as the real recording
    if len(bg) >= int(0.1 * SR):                                     # real background (hum + noise) under the spoof
        k = rng.integers(0, len(bg)); n = np.resize(np.roll(bg, -k), len(s))
        s = s + n
    y = np.concatenate([x[:on], s, x[off:]])                          # real lead-in and tail of the same clip
    return np.clip(y, -1.0, 32767 / 32768)                            # recorder clipping at full scale

print("loading channel-matched (+ch, v4) test spoofs from Spoofs_CM ...", flush=True)
t0 = time.time()
cm_path = {}
with open(os.path.join(DATA, "Spoofs_CM", "spoof_manifest_channel_matched.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        cm_path[r["source_spoof"]] = os.path.join(DATA, "Spoofs_CM", r["path"])
trans = [load(cm_path[r["utt_id"]]) for r in spoofs]
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
L = [f"KurdFake channel-transplant test (RAIL test)  {time.strftime('%Y-%m-%d %H:%M')}",
     "Spoofs given the real channel (hum, noise, level, clipping) of an enrollment recording of the same speaker.",
     "Bona fide test clips unchanged. EER per system: original -> transplanted.", ""]
for name, make in (("AASIST fine-tuned E1 (raw)", aasist_scorer), ("MMS-300M-AntiDeepfake zero-shot", adf_scorer)):
    print(f"\nscoring with {name} ...", flush=True)
    sc = make()
    sb = sc(bona_x)
    so = sc([load(os.path.join(DATA, "Spoofs", r["path"])) for r in spoofs])
    st = sc(trans)
    L.append(f"== {name}")
    for s in systems:
        m = sysarr == s
        L.append(f"  {s:4s}{' (held-out)' if s not in TRAIN_SYSTEMS else '           '}  EER {100*eer(sb, so[m])[0]:5.1f}%  ->  {100*eer(sb, st[m])[0]:5.1f}%")
    L.append(f"  pooled            EER {100*eer(sb, so)[0]:5.1f}%  ->  {100*eer(sb, st)[0]:5.1f}%")
    L.append(f"  mean score: bona fide {sb.mean():.2f}   spoof original {so.mean():.2f}   spoof transplanted {st.mean():.2f}")
    L.append("")
    del sc; torch.cuda.empty_cache() if dev == "cuda" else None
L.append("Reading: a large rise in EER means the detector relied on the recording channel, not on synthesis artefacts.")
txt = "\n".join(L)
open(os.path.join(RES, "channel_test_v4.txt"), "w", encoding="utf-8").write(txt + "\n")
print("\n" + txt + f"\n\nSaved to {RES}\\channel_test_v4.txt   ({(time.time()-t0)/60:.1f} min)")
