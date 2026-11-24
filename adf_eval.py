# =====================================================================
# KurdFake - adf_eval.py   (RQ2 extension: does a modern MULTILINGUAL detector solve Kurdish zero-shot?)
# Model: MMS-300M-AntiDeepfake (Ge et al., 2025), loaded by kf_antideepfake.py (plain PyTorch, no fairseq).
#
#   python adf_eval.py --sanity     loading check on 2,000 random ASVspoof 2019 LA eval files (about 3 min).
#                                   That corpus was part of the model's post-training data, so the EER must be
#                                   very low (a few percent at most). INTERNAL CHECK ONLY - never reported.
#   python adf_eval.py              zero-shot scores for KurdFake dev+test, conditions raw and control
#                                   (same utterances, same conditions and same report format as cm_eval.py)
# Needs: models\antideepfake_mms300m\model.safetensors, kf_antideepfake.py, kf_metrics.py
# Writes: results\adf_scores.csv, results\adf_results.txt   (sanity: results\adf_sanity.txt)
# =====================================================================
import os, io, sys, csv, wave, time, random, zipfile, argparse, collections, platform
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np
import torch

ROOT = KF_ROOT
DATA = os.path.join(ROOT, "Dataset")
RES = os.path.join(ROOT, "results")
CKPT = os.path.join(ROOT, "models", "antideepfake_mms300m", "model.safetensors")
LA_ZIP = os.path.join(DATA, "LA.zip")
SETS = {"rail": os.path.join(DATA, "RAIL_clean"), "cv": os.path.join(DATA, "CV_ckb_clean")}
SPOOF_MANIFESTS = [os.path.join(DATA, "Spoofs", f) for f in ["spoof_manifest_knnvc_final.csv", "spoof_manifest_knnvc_D2.csv",
                                                             "spoof_manifest_tts_vekol.csv", "spoof_manifest_f5.csv"]]
SR, TGT_LUFS, HP_HZ, CHUNK = 16000, -16.0, 100, 64

ap = argparse.ArgumentParser()
ap.add_argument("--sanity", action="store_true")
ap.add_argument("--conds", default="raw,control")
args = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kf_antideepfake import load_antideepfake, score_batch
from kf_metrics import eer
os.makedirs(RES, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"python {platform.python_version()}  torch {torch.__version__}  device {dev}")
model, n_par = load_antideepfake(CKPT, dev)
print(f"MMS-300M-AntiDeepfake loaded: {n_par:,} parameters, every tensor matched by name and shape")

# ---------------- sanity check on ASVspoof 2019 LA eval ----------------
if args.sanity:
    import soundfile as sf
    z = zipfile.ZipFile(LA_ZIP); names = z.namelist()
    proto = [n for n in names if n.endswith("ASVspoof2019.LA.cm.eval.trl.txt")][0]
    flac = {os.path.basename(n)[:-5]: n for n in names if "ASVspoof2019_LA_eval/flac/" in n and n.endswith(".flac")}
    items = [(p[1], p[4]) for p in (l.split() for l in z.read(proto).decode().splitlines())]
    random.Random(1337).shuffle(items); items = items[:2000]
    t0 = time.time(); sc = []
    for b in range(0, len(items), CHUNK):
        wavs = [sf.read(io.BytesIO(z.read(flac[u])), dtype="float32")[0] for u, _ in items[b:b + CHUNK]]
        sc += score_batch(model, wavs, dev)
        print(f"  {b + len(wavs)}/{len(items)}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    sc = np.array(sc); y = np.array([k == "bonafide" for _, k in items])
    e, _ = eer(sc[y], sc[~y])
    txt = (f"AntiDeepfake MMS-300M loading check on 2,000 random ASVspoof 2019 LA eval files (in the model's training data)\n"
           f"EER = {100*e:.2f}%   mean score bona fide {sc[y].mean():.2f}  spoof {sc[~y].mean():.2f}\n"
           f"PASS if EER is a few percent or lower; around 50% would mean the weights did not load correctly.\n")
    print("\n" + txt); open(os.path.join(RES, "adf_sanity.txt"), "w").write(txt); sys.exit(0)

# ---------------- KurdFake zero-shot ----------------
def load(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return x.mean(axis=1).astype(np.float32) / 32768.0

def control(x):                                       # identical to cm_eval.py
    X = np.fft.rfft(x); X[np.fft.rfftfreq(len(x), 1 / SR) < HP_HZ] = 0
    y = np.fft.irfft(X, n=len(x)).astype(np.float32)
    try:
        import torchaudio
        t = torch.from_numpy(y); g = TGT_LUFS - float(torchaudio.functional.loudness(t[None], SR))
        y = torchaudio.functional.gain(t, g).numpy()
    except Exception:
        y = y * (10 ** (TGT_LUFS / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-9))
    return np.clip(y, -1, 1).astype(np.float32)

items = []
for ds, base in SETS.items():
    with open(os.path.join(base, "manifest.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] in ("dev", "test"):
                items.append((r["utt_id"], os.path.join(base, r["clean_path"]), ds, r["split"], "bonafide", "bonafide"))
for mp in SPOOF_MANIFESTS:
    with open(mp, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] in ("dev", "test"):
                items.append((r["utt_id"], os.path.join(DATA, "Spoofs", r["path"]), r["dataset"], r["split"], r["system"], "spoof"))
conds = args.conds.split(",")
print(f"utterances: {len(items)}   conditions: {conds}")
scores = {}
for cond in conds:
    t0 = time.time()
    for b in range(0, len(items), CHUNK):
        chunk = items[b:b + CHUNK]
        wavs = [control(load(p)) if cond == "control" else load(p) for (_, p, *_r) in chunk]
        for (u, *_r), s in zip(chunk, score_batch(model, wavs, dev)):
            scores[(cond, u)] = s
        if (b // CHUNK) % 50 == 0:
            print(f"  {cond}: {b+len(chunk)}/{len(items)}  ({(time.time()-t0)/60:.1f} min)", flush=True)

with open(os.path.join(RES, "adf_scores.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["utterance_id", "dataset", "split", "system", "label"] + [f"ADF-MMS300M_{c}" for c in conds])
    for (u, p, ds, sp, sysn, lab) in items:
        w.writerow([u, ds, sp, sysn, lab] + [round(scores[(c, u)], 5) for c in conds])

L = [f"KurdFake CM evaluation  {time.strftime('%Y-%m-%d %H:%M')}  zero-shot, MMS-300M-AntiDeepfake (multilingual post-training)", ""]
systems = sorted({i[4] for i in items if i[5] == "spoof"})
for cond in conds:
    L.append(f"===== ADF-MMS300M / {cond} =====")
    for ds in SETS:
        def sc(split, system):
            return np.array([scores[(cond, i[0])] for i in items if i[2] == ds and i[3] == split and i[4] == system])
        bona_d, bona_t = sc("dev", "bonafide"), sc("test", "bonafide")
        parts = [f"{s} {100*eer(bona_t, sc('test', s))[0]:5.1f}%" for s in systems]
        spoof_d = np.concatenate([sc("dev", s) for s in systems]); spoof_t = np.concatenate([sc("test", s) for s in systems])
        e_pool, _ = eer(bona_t, spoof_t); _, thr = eer(bona_d, spoof_d)
        L.append(f"  {ds.upper():4s} test EER per system: " + "  ".join(parts) + f"  | pooled {100*e_pool:5.1f}%")
        L.append(f"        dev-threshold on test: bona fide rejected {100*(bona_t < thr).mean():5.1f}%   spoof accepted {100*(spoof_t >= thr).mean():5.1f}%")
        L.append(f"        mean score: bona fide {bona_t.mean():.2f}  " + "  ".join(f"{s} {sc('test', s).mean():.2f}" for s in systems))
    L.append("")
L.append("EER: 50% = chance. Score = logit(real) - logit(fake); higher = more bona fide.")
txt = "\n".join(L)
open(os.path.join(RES, "adf_results.txt"), "w", encoding="utf-8").write(txt)
print("\n" + txt + f"\n\nSaved to {RES}")
