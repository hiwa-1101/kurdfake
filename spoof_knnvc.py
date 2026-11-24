# =====================================================================
# KurdFake - spoof_knnvc.py   (Prompt 2, systems D and G)
#   D = kNN-VC voice conversion (Baas et al., 2023; WavLM-Large + prematched HiFi-GAN)
#   G = vocoder resynthesis: the target's own speech -> WavLM features -> same HiFi-GAN
#
# Runs in the existing conda env gpu_env (Python 3.8, torch 2.4.1+cu121); needs NO internet and NO extra packages.
# Model files must be in MODEL_DIR (folders wavlm\ and hifigan\ + WavLM-Large.pt + prematch_g_02500000.pt).
# Usage (Anaconda Prompt):
#   conda activate gpu_env
#   cd /d <KURDFAKE_ROOT>
#   python spoof_knnvc.py --smoke        (2 speakers per dataset, quick test)
#   python spoof_knnvc.py                (everything; resumes if interrupted)
#
# Reads  : Dataset\RAIL_clean\manifest.csv, Dataset\CV_ckb_clean\manifest.csv
# Writes : Dataset\Spoofs\<system>\<dataset>\<speaker>\*.wav  +  Dataset\Spoofs\spoof_manifest_knnvc.csv
# Enrollment clips are the ONLY reference audio of a target speaker that is ever used.
# =====================================================================

import os, csv, sys, random, argparse, time, collections, platform
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import wave, json
import numpy as np
import torch

DATA = os.path.join(KF_ROOT, "Dataset")
SETS = {"rail": os.path.join(DATA, "RAIL_clean"), "cv": os.path.join(DATA, "CV_ckb_clean")}
OUT  = os.path.join(DATA, "Spoofs")
MODEL_DIR = os.path.join(KF_ROOT, "models", "knnvc")
SEED = 1337
SR   = 16000
TOPK = 4                     # kNN-VC default
N_PER_SPK = {"rail": 50, "cv": 20}          # spoofs per target speaker and system
RAIL_REF_REPS = {1, 2, 3}                    # same reps as the enrollment role
CV_N_REF = 5
PAD_S = 0.05                                 # keep 50 ms around detected speech in RAIL references

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true", help="only 2 speakers per dataset")
ap.add_argument("--systems", default="D,G")
ap.add_argument("--datasets", default="rail,cv")
ap.add_argument("--disjoint", action="store_true",
                help="system D2: kNN-VC with attacker references DISJOINT from the ASV enrollment clips (dev/test only)")
args = ap.parse_args()
systems = args.systems.split(","); datasets = args.datasets.split(",")
if args.disjoint:
    systems = ["D"]                  # written as system "D2"
RAIL_D2_REF_REPS = {4, 5, 6}         # D2: attacker material = reps 4-6 (never used for ASV enrollment)
rng = random.Random(SEED); torch.manual_seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"python {platform.python_version()}  torch {torch.__version__}  device {dev}"
      + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else "  <-- WARNING: no GPU"))

# ---------------- model (loaded from local files; same weights as torch.hub bshall/knn-vc, prematched) ----------------
sys.path.insert(0, MODEL_DIR)
from wavlm.WavLM import WavLM, WavLMConfig
from hifigan.models import Generator
from hifigan.utils import AttrDict
ck = torch.load(os.path.join(MODEL_DIR, "WavLM-Large.pt"), map_location="cpu")
wavlm = WavLM(WavLMConfig(ck["cfg"])); wavlm.load_state_dict(ck["model"]); wavlm = wavlm.to(dev).eval()
with open(os.path.join(MODEL_DIR, "hifigan", "config_v1_wavlm.json")) as fh:
    h = AttrDict(json.load(fh))
hifigan = Generator(h); hifigan.load_state_dict(torch.load(os.path.join(MODEL_DIR, "prematch_g_02500000.pt"), map_location="cpu")["generator"])
hifigan = hifigan.to(dev).eval(); hifigan.remove_weight_norm()
del ck
print("models loaded: WavLM-Large + prematched HiFi-GAN")
LAYER = 6                     # kNN-VC uses WavLM layer 6

def vocode(feats):
    with torch.inference_mode():
        y = hifigan(feats[None].to(dev))
    return y.squeeze().float().cpu().numpy()

def feats_of(wav):
    with torch.inference_mode():
        x = torch.from_numpy(wav).float()[None].to(dev)
        return wavlm.extract_features(x, output_layer=LAYER, ret_layer_results=False)[0].squeeze(0)

def knn_convert(q, m):
    # cosine kNN regression (same as kNN-VC match(), written out to avoid API drift)
    with torch.inference_mode():
        qn = torch.nn.functional.normalize(q.to(dev), dim=-1)
        mn = torch.nn.functional.normalize(m.to(dev), dim=-1)
        idx = (qn @ mn.T).topk(min(TOPK, m.shape[0]), dim=-1).indices
        return m.to(dev)[idx].mean(dim=1)

def load(base, rel):
    with wave.open(os.path.join(base, rel), "rb") as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, rel
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return x.mean(axis=1).astype(np.float32) / 32768.0

def trim(x, r):
    try:
        on, off = float(r["onset_s"]), float(r["offset_s"])
        return x[max(0, int((on - PAD_S) * SR)): int((off + PAD_S) * SR)]
    except (ValueError, KeyError):
        return x

TGT_LUFS = -16.0             # kNN-VC default output loudness (applied to D and G alike)
def loudnorm(y):
    try:
        import torchaudio
        t = torch.from_numpy(y).float()
        g = TGT_LUFS - float(torchaudio.functional.loudness(t[None], SR))
        return torchaudio.functional.gain(t, g).numpy()
    except Exception:
        rms = np.sqrt(np.mean(y ** 2)) + 1e-9
        return y * (10 ** (TGT_LUFS / 20) / rms)

def save(path, y):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    y = np.clip(loudnorm(y), -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())

# ---------------- manifest of spoofs ----------------
man_path = os.path.join(OUT, "spoof_manifest_knnvc_D2.csv" if args.disjoint else "spoof_manifest_knnvc.csv")
os.makedirs(OUT, exist_ok=True)
done = set()
if os.path.exists(man_path):
    with open(man_path, encoding="utf-8") as fh:
        done = {r["utt_id"] for r in csv.DictReader(fh)}
cols = ["utt_id", "path", "dataset", "system", "split", "target_speaker", "target_gender",
        "source_utt_id", "source_speaker", "digit", "sentence", "n_ref", "ref_speech_s", "dur_s", "label"]
mf = open(man_path, "a", newline="", encoding="utf-8")
mw = csv.DictWriter(mf, fieldnames=cols)
if not done:
    mw.writeheader()

d_sources = collections.defaultdict(list)
if args.disjoint:
    fm = os.path.join(OUT, "spoof_manifest_knnvc_final.csv")
    with open(fm, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["system"] == "D":
                d_sources[(r["dataset"], r["target_speaker"])].append(r["source_utt_id"])
    print(f"D2: re-using the source clips of {len(d_sources)} D targets from {fm}")
t0 = time.time(); made = 0
for ds in datasets:
    base = SETS[ds]
    with open(os.path.join(base, "manifest.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_spk = collections.defaultdict(list)
    for r in rows:
        by_spk[r["speaker"]].append(r)
    spks = sorted(by_spk)
    if args.smoke:
        spks = [s for s in spks if by_spk[s][0]["split"] == "test"][:2]
    print(f"\n== {ds}: {len(spks)} target speakers ==")
    for k, s in enumerate(spks):
        rs = by_spk[s]; split = rs[0]["split"]; gender = rs[0]["gender"]
        if args.disjoint and split not in ("dev", "test"):
            continue
        if args.disjoint:
            # D2: attacker references are other recordings of the target, NOT the ASV enrollment clips
            if ds == "rail":
                refs = [r for r in rs if int(r["rep"]) in RAIL_D2_REF_REPS]
            else:
                refs = sorted([r for r in rs if r["role"] == "trial"], key=lambda r: r["utt_id"])[:CV_N_REF]
        elif ds == "rail":
            # references = enrollment material only
            refs = [r for r in rs if int(r["rep"]) in RAIL_REF_REPS]
        else:
            refs = [r for r in rs if r["role"] == "enroll"] or rs[:CV_N_REF]
            refs = refs[:CV_N_REF]
        ref_ids = {r["utt_id"] for r in refs}
        ref_wavs = [trim(load(base, r["clean_path"]), r) if ds == "rail" else load(base, r["clean_path"]) for r in refs]
        ref_speech = sum(float(r["speech_s"] or 0) for r in refs)
        mset = None
        # query material
        own = [r for r in rs if r["utt_id"] not in ref_ids]
        others = [r for r in rows if r["split"] == split and r["speaker"] != s
                  and (r["gender"] == gender or gender == "unknown") and r["role"] != "enroll"]
        if ds == "rail":
            others = [r for r in others if int(r["rep"]) not in RAIL_REF_REPS]
        n = N_PER_SPK[ds]
        def pick(pool):
            if ds == "rail":   # balanced digits: n/10 per digit, prefer clips not touching the edges
                out = []
                for d in range(10):
                    p = [r for r in pool if r["digit"] == str(d)]
                    p.sort(key=lambda r: (r.get("flag_truncated") == "1", rng.random()))
                    out += p[: n // 10]
                return out
            p = pool[:]; rng.shuffle(p); return p[:n]
        jobs = []
        if args.disjoint:
            # D2 converts exactly the same source clips as D did for this target -> paired comparison D vs D2
            src_ids = d_sources.get((ds, s), [])
            by_id = {r["utt_id"]: r for r in rows}
            jobs = [("D", by_id[u]) for u in src_ids if u in by_id] if refs else []
            systems_this = []
        else:
            systems_this = systems
        if "G" in systems_this: jobs += [("G", q) for q in pick(own)]
        if "D" in systems_this:
            if refs:
                jobs += [("D", q) for q in pick(others)]
            else:
                print(f"  [{ds}] {s}: no enrollment clips -> D skipped")
        for sysname, q in jobs:
            if args.disjoint:
                sysname = "D2"
            uid = f"spf_{sysname}_{ds}_{s}_{q['utt_id']}"
            if uid in done:
                continue
            x = load(base, q["clean_path"])
            fq = feats_of(x)
            if sysname == "G":
                y = vocode(fq)
            else:   # D or D2
                if mset is None:
                    mset = torch.cat([feats_of(w) for w in ref_wavs], dim=0)
                y = vocode(knn_convert(fq, mset))
            rel = os.path.join(sysname, ds, s, uid + ".wav")
            save(os.path.join(OUT, rel), y)
            mw.writerow({"utt_id": uid, "path": rel, "dataset": ds, "system": sysname, "split": split,
                         "target_speaker": s, "target_gender": gender, "source_utt_id": q["utt_id"],
                         "source_speaker": q["speaker"], "digit": q.get("digit", ""), "sentence": q.get("sentence", ""),
                         "n_ref": len(refs) if sysname != "G" else 0, "ref_speech_s": round(ref_speech, 2) if sysname != "G" else "",
                         "dur_s": round(len(y) / SR, 3), "label": "spoof"})
            made += 1
        mf.flush()
        el = time.time() - t0
        print(f"  [{ds}] {k+1}/{len(spks)} {s} ({split})  made so far: {made}  elapsed {el/60:.1f} min", flush=True)
mf.close()
print(f"\nDone. {made} new spoof files in {OUT}\nManifest: {man_path}")
