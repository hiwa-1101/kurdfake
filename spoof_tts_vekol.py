# =====================================================================
# KurdFake - spoof_tts_vekol.py   (Prompt 2, TTS family: system C1 and cascade C1D)
#   C1  = Vekol-TTS (RevgeAI/vekol-tts-ckb-edge, Piper/VITS, one male Sorani voice, CC BY-NC 4.0)
#         -> same CONTENT as the bona fide data (RAIL: digit words; CV: the target speaker's own sentences)
#   C1D = C1 output converted to the TARGET speaker's voice with kNN-VC (TTS + VC cascade attack)
#         attacker references: adapt = enrollment reps (as D); dev/test = DISJOINT recordings (as D2)
#
# Anaconda Prompt:
#   conda activate gpu_env
#   cd /d <KURDFAKE_ROOT>
#   python spoof_tts_vekol.py --smoke      (2 test speakers per dataset)
#   python spoof_tts_vekol.py              (all; resumes if interrupted)
# Needs: models\vekol\  model.onnx  model.onnx.json  vekol_tts.py      (+ onnxruntime from wheels_tts)
#        models\knnvc\  (already there)
# Writes: Dataset\Spoofs\C1\..., Dataset\Spoofs\C1D\...,  Dataset\Spoofs\spoof_manifest_tts_vekol.csv
# =====================================================================
import os, sys, csv, json, wave, random, argparse, time, collections, platform, tempfile, math
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np
import torch
from scipy.signal import resample_poly

DATA = os.path.join(KF_ROOT, "Dataset")
SETS = {"rail": os.path.join(DATA, "RAIL_clean"), "cv": os.path.join(DATA, "CV_ckb_clean")}
OUT  = os.path.join(DATA, "Spoofs")
KNN_DIR   = os.path.join(KF_ROOT, "models", "knnvc")
VEKOL_DIR = os.path.join(KF_ROOT, "models", "vekol")
SEED, SR, TOPK, PAD_S = 1337, 16000, 4, 0.05
N_PER_SPK = {"rail": 50, "cv": 20}
RAIL_ENROLL_REPS, RAIL_DISJOINT_REPS, CV_N_REF = {1, 2, 3}, {4, 5, 6}, 5
# Sorani digit words (same spelling as Vekol's own number reader)
DIGIT_WORDS = ["سفر", "یەک", "دوو", "سێ", "چوار", "پێنج", "شەش", "حەوت", "هەشت", "نۆ"]

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true"); ap.add_argument("--datasets", default="rail,cv")
args = ap.parse_args()
rng = random.Random(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"python {platform.python_version()}  torch {torch.__version__}  device {dev}")

# ---------------- Vekol TTS (official helper, unchanged; runs on CPU via onnxruntime) ----------------
sys.path.insert(0, VEKOL_DIR)
import vekol_tts                                   # loads model.onnx from VEKOL_DIR
print(f"Vekol-TTS loaded ({vekol_tts.SR} Hz)")
_tmp = os.path.join(tempfile.gettempdir(), "kf_vekol_tmp.wav")

def tts(text):
    vekol_tts.speak(text, _tmp)                     # Vekol's own pipeline: normalise, synth, denoise, trim
    with wave.open(_tmp, "rb") as w:
        sr = w.getframerate(); y = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    if sr != SR:                                     # 22.05 kHz -> 16 kHz (ratio 320/441; fixed v2)
        g = math.gcd(SR, sr); y = resample_poly(y, SR // g, sr // g)
    return y.astype(np.float32)

# ---------------- kNN-VC (same weights and code path as spoof_knnvc.py) ----------------
sys.path.insert(0, KNN_DIR)
from wavlm.WavLM import WavLM, WavLMConfig
from hifigan.models import Generator
from hifigan.utils import AttrDict
ck = torch.load(os.path.join(KNN_DIR, "WavLM-Large.pt"), map_location="cpu")
wavlm = WavLM(WavLMConfig(ck["cfg"])); wavlm.load_state_dict(ck["model"]); wavlm = wavlm.to(dev).eval(); del ck
with open(os.path.join(KNN_DIR, "hifigan", "config_v1_wavlm.json")) as fh:
    hifigan = Generator(AttrDict(json.load(fh)))
hifigan.load_state_dict(torch.load(os.path.join(KNN_DIR, "prematch_g_02500000.pt"), map_location="cpu")["generator"])
hifigan = hifigan.to(dev).eval(); hifigan.remove_weight_norm()
print("kNN-VC loaded")

def feats_of(w):
    with torch.inference_mode():
        return wavlm.extract_features(torch.from_numpy(w).float()[None].to(dev), output_layer=6, ret_layer_results=False)[0].squeeze(0)
def convert(q, m):
    with torch.inference_mode():
        qn, mn = torch.nn.functional.normalize(q, dim=-1), torch.nn.functional.normalize(m, dim=-1)
        idx = (qn @ mn.T).topk(min(TOPK, m.shape[0]), dim=-1).indices
        return hifigan(m[idx].mean(dim=1)[None]).squeeze().float().cpu().numpy()

def load(base, rel):
    with wave.open(os.path.join(base, rel), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return x.mean(axis=1).astype(np.float32) / 32768.0
def trim(x, r):
    try:
        return x[max(0, int((float(r["onset_s"]) - PAD_S) * SR)): int((float(r["offset_s"]) + PAD_S) * SR)]
    except (ValueError, KeyError):
        return x
def loudnorm(y, tgt=-16.0):
    try:
        import torchaudio
        t = torch.from_numpy(y).float(); return torchaudio.functional.gain(t, tgt - float(torchaudio.functional.loudness(t[None], SR))).numpy()
    except Exception:
        return y * (10 ** (tgt / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-9))
def save(path, y):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    y = np.clip(loudnorm(y), -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((y * 32767).astype(np.int16).tobytes())

# ---------------- manifest ----------------
os.makedirs(OUT, exist_ok=True)
man = os.path.join(OUT, "spoof_manifest_tts_vekol.csv")
done = set()
if os.path.exists(man):
    with open(man, encoding="utf-8") as fh:
        done = {r["utt_id"] for r in csv.DictReader(fh)}
cols = ["utt_id", "path", "dataset", "system", "split", "target_speaker", "target_gender", "source_utt_id",
        "source_speaker", "digit", "sentence", "text", "ref_mode", "n_ref", "ref_speech_s", "dur_s", "label"]
mf = open(man, "a", newline="", encoding="utf-8"); mw = csv.DictWriter(mf, fieldnames=cols)
if not done:
    mw.writeheader()

t0, made = time.time(), 0
for ds in args.datasets.split(","):
    base = SETS[ds]
    with open(os.path.join(base, "manifest.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["speaker"]].append(r)
    spks = sorted(by)
    if args.smoke:
        spks = [s for s in spks if by[s][0]["split"] == "test"][:2]
    print(f"\n== {ds}: {len(spks)} target speakers ==")
    for k, s in enumerate(spks):
        rs = by[s]; split, gender = rs[0]["split"], rs[0]["gender"]
        disjoint = split in ("dev", "test")
        if ds == "rail":
            refs = [r for r in rs if int(r["rep"]) in (RAIL_DISJOINT_REPS if disjoint else RAIL_ENROLL_REPS)]
            jobs = [(d, None) for d in range(10) for _ in range(N_PER_SPK["rail"] // 10)]
        else:
            enroll = [r for r in rs if r["role"] == "enroll"] or rs[:CV_N_REF]
            pool = sorted([r for r in rs if r["role"] in ("trial", "train") and r not in enroll], key=lambda r: r["utt_id"])
            refs = pool[:CV_N_REF] if disjoint else enroll[:CV_N_REF]
            texts = [r for r in pool if r not in refs]; rng.shuffle(texts)
            jobs = [(None, r) for r in texts[:N_PER_SPK["cv"]]]
        if not refs:
            print(f"  {s}: no reference clips -> skipped"); continue
        mset = torch.cat([feats_of(trim(load(base, r["clean_path"]), r) if ds == "rail" else load(base, r["clean_path"])) for r in refs])
        ref_speech = round(sum(float(r["speech_s"] or 0) for r in refs), 2)
        for j, (d, src) in enumerate(jobs):
            text = DIGIT_WORDS[d] if ds == "rail" else src["sentence"]
            tag = f"d{d}_{j:02d}" if ds == "rail" else src["utt_id"]
            uid1, uid2 = f"spf_C1_{ds}_{s}_{tag}", f"spf_C1D_{ds}_{s}_{tag}"
            if uid1 in done and uid2 in done:
                continue
            y = tts(text)
            common = {"dataset": ds, "split": split, "target_speaker": s, "target_gender": gender,
                      "digit": "" if d is None else d, "sentence": "" if src is None else src["sentence"], "text": text, "label": "spoof"}
            if uid1 not in done:
                save(os.path.join(OUT, "C1", ds, s, uid1 + ".wav"), y)
                mw.writerow({**common, "utt_id": uid1, "path": os.path.join("C1", ds, s, uid1 + ".wav"), "system": "C1",
                             "source_utt_id": "", "source_speaker": "vekol", "ref_mode": "", "n_ref": 0, "ref_speech_s": "",
                             "dur_s": round(len(y) / SR, 3)})
            z = convert(feats_of(y), mset)
            save(os.path.join(OUT, "C1D", ds, s, uid2 + ".wav"), z)
            mw.writerow({**common, "utt_id": uid2, "path": os.path.join("C1D", ds, s, uid2 + ".wav"), "system": "C1D",
                         "source_utt_id": uid1, "source_speaker": "vekol", "ref_mode": "disjoint" if disjoint else "enroll",
                         "n_ref": len(refs), "ref_speech_s": ref_speech, "dur_s": round(len(z) / SR, 3)})
            made += 2
        mf.flush()
        print(f"  [{ds}] {k+1}/{len(spks)} {s} ({split})  files so far: {made}  {(time.time()-t0)/60:.1f} min", flush=True)
mf.close()
print(f"\nDone. {made} new files.  Manifest: {man}")
