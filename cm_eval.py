# =====================================================================
# KurdFake - cm_eval.py   (Prompt 4 / RQ2: do English-trained spoofing detectors work on Kurdish?)
# Countermeasures: AASIST and AASIST-L (Jung et al., ICASSP 2022), official weights trained on
#                  ASVspoof 2019 LA (English). Used as-is: zero-shot, no Kurdish training.
# Two input conditions:
#   raw     : files as they are
#   control : every file (bona fide and spoof) high-passed at 100 Hz (removes RAIL mains hum)
#             and loudness-normalised to -16 LUFS  -> tests the hum/loudness shortcut
#
# Anaconda Prompt:
#   conda activate gpu_env
#   cd /d <KURDFAKE_ROOT>
#   python cm_eval.py
# Needs: models\aasist\ (code + weights from github.com/clovaai/aasist, MIT licence) - sent as a zip
# Writes: results\cm_scores.csv, results\cm_results.txt
# =====================================================================
import os, sys, csv, json, wave, time, collections, platform
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np
import torch

DATA   = os.path.join(KF_ROOT, "Dataset")
AASIST = os.path.join(KF_ROOT, "models", "aasist")
RES    = os.path.join(KF_ROOT, "results")
SETS   = {"rail": os.path.join(DATA, "RAIL_clean"), "cv": os.path.join(DATA, "CV_ckb_clean")}
SPOOF_MANIFESTS = [os.path.join(DATA, "Spoofs", f) for f in ["spoof_manifest_knnvc_final.csv", "spoof_manifest_knnvc_D2.csv", "spoof_manifest_tts_vekol.csv", "spoof_manifest_f5.csv"]]
SPLITS = ["dev", "test"]
MODELS = ["AASIST", "AASIST-L"]
CONDITIONS = ["raw", "control"]
SR, NB_SAMP, BATCH = 16000, 64600, 16      # batch 16 fits easily in 16 GB GPU memory
TGT_LUFS, HP_HZ = -16.0, 100

os.makedirs(RES, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"python {platform.python_version()}  torch {torch.__version__}  device {dev}")
import importlib.util
_spec = importlib.util.spec_from_file_location("aasist_model", os.path.join(AASIST, "models", "AASIST.py"))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
AASISTModel = _mod.Model
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def load(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return x.mean(axis=1).astype(np.float32) / 32768.0

def pad(x, max_len=NB_SAMP):           # identical to aasist/data_utils.pad (evaluation padding)
    if len(x) >= max_len:
        return x[:max_len]
    return np.tile(x, int(max_len / len(x)) + 1)[:max_len]

def control(x):
    X = np.fft.rfft(x); X[np.fft.rfftfreq(len(x), 1 / SR) < HP_HZ] = 0
    y = np.fft.irfft(X, n=len(x)).astype(np.float32)
    try:
        import torchaudio
        t = torch.from_numpy(y)
        g = TGT_LUFS - float(torchaudio.functional.loudness(t[None], SR))
        y = torchaudio.functional.gain(t, g).numpy()
    except Exception:
        y = y * (10 ** (TGT_LUFS / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-9))
    return np.clip(y, -1, 1).astype(np.float32)

# ---------------- utterance list ----------------
items = []   # (utt_id, path, dataset, split, system, label)
for ds, base in SETS.items():
    with open(os.path.join(base, "manifest.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] in SPLITS:
                items.append((r["utt_id"], os.path.join(base, r["clean_path"]), ds, r["split"], "bonafide", "bonafide"))
for mp in SPOOF_MANIFESTS:
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["split"] in SPLITS:
                    items.append((r["utt_id"], os.path.join(DATA, "Spoofs", r["path"]), r["dataset"], r["split"], r["system"], "spoof"))
print("utterances:", dict(collections.Counter((i[2], i[4]) for i in items)))

# ---------------- scoring ----------------
scores = {}   # (model, condition, utt_id) -> score  (higher = more bona fide)
for mname in MODELS:
    cfg = json.load(open(os.path.join(AASIST, "config", f"{mname}.conf")))
    model = AASISTModel(cfg["model_config"])
    model.load_state_dict(torch.load(os.path.join(AASIST, "models", "weights", f"{mname}.pth"), map_location="cpu"))
    model = model.to(dev).eval()
    print(f"\n{mname} loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    for cond in CONDITIONS:
        t0 = time.time()
        for b in range(0, len(items), BATCH):
            chunk = items[b:b + BATCH]
            xs = []
            for (u, p, *_rest) in chunk:
                x = load(p)
                if cond == "control":
                    x = control(x)
                xs.append(pad(x))
            with torch.no_grad():
                _, out = model(torch.from_numpy(np.stack(xs)).to(dev))
            for (u, *_r), s in zip(chunk, out[:, 1].float().cpu().numpy()):
                scores[(mname, cond, u)] = float(s)
            if (b // BATCH) % 200 == 0:
                print(f"  {mname}/{cond}: {b+len(chunk)}/{len(items)}  ({(time.time()-t0)/60:.1f} min)", flush=True)

with open(os.path.join(RES, "cm_scores.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["utterance_id", "dataset", "split", "system", "label"] + [f"{m}_{c}" for m in MODELS for c in CONDITIONS])
    for (u, p, ds, sp, sysn, lab) in items:
        w.writerow([u, ds, sp, sysn, lab] + [round(scores[(m, c, u)], 5) for m in MODELS for c in CONDITIONS])

# ---------------- metrics ----------------
def eer(bona, spoof):
    s = np.concatenate([bona, spoof]); y = np.concatenate([np.ones(len(bona)), np.zeros(len(spoof))])
    o = np.argsort(s, kind="mergesort"); s, y = s[o], y[o]
    frr = np.cumsum(y) / y.sum(); far = 1 - np.cumsum(1 - y) / (1 - y).sum()
    i = np.argmin(np.abs(frr - far)); return float((frr[i] + far[i]) / 2), float(s[i])

L = [f"KurdFake CM evaluation  {time.strftime('%Y-%m-%d %H:%M')}  zero-shot, ASVspoof 2019 LA weights", ""]
systems = sorted({i[4] for i in items if i[5] == "spoof"})
for mname in MODELS:
    for cond in CONDITIONS:
        L.append(f"===== {mname} / {cond} =====")
        for ds in SETS:
            def sc(split, system):
                return np.array([scores[(mname, cond, i[0])] for i in items if i[2] == ds and i[3] == split and i[4] == system])
            bona_d, bona_t = sc("dev", "bonafide"), sc("test", "bonafide")
            parts = []
            for system in systems:
                e, _ = eer(bona_t, sc("test", system)); parts.append(f"{system} {100*e:5.1f}%")
            spoof_d = np.concatenate([sc("dev", s) for s in systems]); spoof_t = np.concatenate([sc("test", s) for s in systems])
            e_pool, _ = eer(bona_t, spoof_t)
            _, thr = eer(bona_d, spoof_d)          # threshold fixed on dev, applied to test
            miss = 100 * (bona_t < thr).mean(); fa = 100 * (spoof_t >= thr).mean()
            L.append(f"  {ds.upper():4s} test EER per system: " + "  ".join(parts) + f"  | pooled {100*e_pool:5.1f}%")
            L.append(f"        dev-threshold on test: bona fide rejected {miss:5.1f}%   spoof accepted {fa:5.1f}%")
            L.append(f"        mean score: bona fide {bona_t.mean():.2f}  " + "  ".join(f"{s} {sc('test', s).mean():.2f}" for s in systems))
        L.append("")
L.append("EER: 50% = chance. Scores: higher = more bona fide (AASIST output logit for the bona fide class).")
txt = "\n".join(L)
open(os.path.join(RES, "cm_results.txt"), "w", encoding="utf-8").write(txt)
print("\n" + txt + f"\n\nSaved to {RES}")
