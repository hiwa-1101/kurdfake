# =====================================================================
# KurdFake - asv_eval.py   (Prompt 3 / RQ1: how often do spoofs fool speaker verification?)
# ASV = ECAPA-TDNN (SpeechBrain spkrec-ecapa-voxceleb, trained on VoxCeleb, used as-is, no Kurdish tuning)
#
# Anaconda Prompt:
#   conda activate gpu_env
#   cd /d <KURDFAKE_ROOT>
#   python asv_eval.py
#
# Needs: models\ecapa\embedding_model.ckpt  (from huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
#        speechbrain installed offline from the wheels folder
# Writes: results\asv_scores.csv  (utterance_id, speaker, label, system, score, ...)
#         results\asv_results.txt (EER, threshold, SPF-FAR per system with 95% bootstrap CI over speakers)
# =====================================================================
import os, csv, wave, json, random, time, collections, platform
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np
import torch

DATA   = os.path.join(KF_ROOT, "Dataset")
ECAPA  = os.path.join(KF_ROOT, "models", "ecapa", "embedding_model.ckpt")
RES    = os.path.join(KF_ROOT, "results")
SPOOF_MANIFESTS = [os.path.join(DATA, "Spoofs", f) for f in ["spoof_manifest_knnvc_final.csv", "spoof_manifest_knnvc_D2.csv", "spoof_manifest_tts_vekol.csv", "spoof_manifest_f5.csv"]]
SETS   = {"rail": os.path.join(DATA, "RAIL_clean"), "cv": os.path.join(DATA, "CV_ckb_clean")}
SPLITS = ["dev", "test"]
SEED, N_BOOT = 1337, 1000
SR = 16000

os.makedirs(RES, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"python {platform.python_version()}  torch {torch.__version__}  device {dev}")

# ---------------- model: same configuration as spkrec-ecapa-voxceleb/hyperparams.yaml ----------------
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
fbank = Fbank(n_mels=80).to(dev)
norm = InputNormalization(norm_type="sentence", std_norm=False).to(dev)
ecapa = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1],
                   dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=192)
ecapa.load_state_dict(torch.load(ECAPA, map_location="cpu"), strict=True)
ecapa = ecapa.to(dev).eval()
print("ECAPA-TDNN loaded")

def load(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return x.mean(axis=1).astype(np.float32) / 32768.0

@torch.no_grad()
def embed(path):
    x = torch.from_numpy(load(path))[None].to(dev)
    lens = torch.ones(1, device=dev)
    f = norm(fbank(x), lens)
    e = ecapa(f, lens).squeeze()
    return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

# ---------------- collect utterances ----------------
bona = {}          # utt_id -> row
for ds, base in SETS.items():
    with open(os.path.join(base, "manifest.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] in SPLITS:
                r["dataset"] = ds; r["abs"] = os.path.join(base, r["clean_path"]); bona[r["utt_id"]] = r
spoofs = []
for mp in SPOOF_MANIFESTS:
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as fh:
            spoofs += [r for r in csv.DictReader(fh) if r["split"] in SPLITS]
for r in spoofs:
    r["abs"] = os.path.join(DATA, "Spoofs", r["path"])
CM_MANIFEST = os.path.join(DATA, "Spoofs_CM", "spoof_manifest_channel_matched.csv")   # +ch spoofs (v4), added 24 Nov
if os.path.exists(CM_MANIFEST):
    with open(CM_MANIFEST, encoding="utf-8") as fh:
        ch = [r for r in csv.DictReader(fh) if r["split"] in SPLITS]
    for r in ch:
        r["abs"] = os.path.join(DATA, "Spoofs_CM", r["path"])
    spoofs += ch
print(f"bona fide utterances: {len(bona)}   spoof utterances: {len(spoofs)}")

# ---------------- embeddings (cached) ----------------
cache_path = os.path.join(RES, "ecapa_embeddings.npz")
emb = {}
if os.path.exists(cache_path):
    z = np.load(cache_path, allow_pickle=False); emb = dict(zip(z["ids"].tolist(), z["E"]))
todo = [(u, r["abs"]) for u, r in bona.items() if u not in emb] + [(r["utt_id"], r["abs"]) for r in spoofs if r["utt_id"] not in emb]
t0 = time.time()
for i, (u, p) in enumerate(todo):
    emb[u] = embed(p)
    if (i + 1) % 2000 == 0:
        print(f"  embedded {i+1}/{len(todo)}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        np.savez(cache_path, ids=np.array(list(emb)), E=np.stack(list(emb.values())))
if todo:
    np.savez(cache_path, ids=np.array(list(emb)), E=np.stack(list(emb.values())))
print(f"embeddings ready ({len(emb)})")

# ---------------- speaker models = mean of enrollment embeddings ----------------
spk_model = {}
for ds in SETS:
    acc = collections.defaultdict(list)
    for u, r in bona.items():
        if r["dataset"] == ds and r["role"] == "enroll":
            acc[r["speaker"]].append(emb[u])
    for s, v in acc.items():
        m = np.mean(v, axis=0); spk_model[(ds, s)] = m / np.linalg.norm(m)

# ---------------- score all trials ----------------
rows, skipped = [], 0
for ds, base in SETS.items():
    for sp in SPLITS:
        with open(os.path.join(base, f"trials_{sp}.csv"), encoding="utf-8") as fh:
            for t in csv.DictReader(fh):
                u = t["test_utt_id"]; m = spk_model.get((ds, t["enroll_speaker"]))
                if m is None or u not in emb:
                    skipped += 1; continue
                rows.append({"dataset": ds, "split": sp, "utterance_id": u, "speaker": t["enroll_speaker"],
                             "true_speaker": bona[u]["speaker"], "label": t["label"], "system": "bonafide",
                             "score": float(emb[u] @ m)})
for r in spoofs:
    m = spk_model.get((r["dataset"], r["target_speaker"]))
    if m is None:
        continue
    rows.append({"dataset": r["dataset"], "split": r["split"], "utterance_id": r["utt_id"], "speaker": r["target_speaker"],
                 "true_speaker": r["source_speaker"], "label": "spoof", "system": r["system"],
                 "score": float(emb[r["utt_id"]] @ m)})
if skipped:
    print(f"WARNING: {skipped} trials skipped (speaker without enrollment clips)")
with open(os.path.join(RES, "asv_scores_with_ch.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# ---------------- metrics ----------------
def eer(tar, non):
    s = np.concatenate([tar, non]); y = np.concatenate([np.ones(len(tar)), np.zeros(len(non))])
    o = np.argsort(s, kind="mergesort"); s, y = s[o], y[o]
    frr = np.cumsum(y) / y.sum()                       # targets at or below threshold are rejected
    far = 1 - np.cumsum(1 - y) / (1 - y).sum()
    i = np.argmin(np.abs(frr - far))
    return float((frr[i] + far[i]) / 2), float(s[i])

rng = np.random.default_rng(SEED)
L = [f"KurdFake ASV evaluation  {time.strftime('%Y-%m-%d %H:%M')}  ECAPA-TDNN (VoxCeleb, no adaptation)", ""]
summary = {}
for ds in SETS:
    for sp in SPLITS:
        R = [r for r in rows if r["dataset"] == ds and r["split"] == sp]
        tar = np.array([r["score"] for r in R if r["label"] == "target"])
        non = np.array([r["score"] for r in R if r["label"] == "nontarget"])
        e, thr = eer(tar, non)
        L.append(f"== {ds.upper()} / {sp}:  target={len(tar)}  nontarget={len(non)}  EER={100*e:.2f}%  threshold={thr:.4f}")
        L.append(f"   mean score: target={tar.mean():.3f}  nontarget={non.mean():.3f}")
        res = {"eer": e, "thr": thr}
        spk = sorted({r["speaker"] for r in R})
        for system in sorted({r["system"] for r in spoofs}):
            S = [r for r in R if r["system"] == system]
            if not S:
                continue
            sc = np.array([r["score"] for r in S]); far = float((sc >= thr).mean())
            # bootstrap over target speakers (threshold re-estimated in every resample)
            by = collections.defaultdict(lambda: {"t": [], "n": [], "s": []})
            for r in R:
                k = "t" if r["label"] == "target" else "n" if r["label"] == "nontarget" else ("s" if r["system"] == system else None)
                if k: by[r["speaker"]][k].append(r["score"])
            boot = []
            for _ in range(N_BOOT):
                pick = rng.choice(spk, size=len(spk), replace=True)
                t = [x for s in pick for x in by[s]["t"]]; n = [x for s in pick for x in by[s]["n"]]
                ss = np.array([x for s in pick for x in by[s]["s"]])
                if not t or not n or not len(ss):
                    continue
                t, n = np.array(t), np.array(n)
                _, th = eer(t, n); boot.append((ss >= th).mean())
            lo, hi = np.percentile(boot, [2.5, 97.5])
            L.append(f"   spoof {system}: n={len(sc)}  mean score={sc.mean():.3f}  SPF-FAR at EER threshold = {100*far:.1f}%  (95% CI {100*lo:.1f}-{100*hi:.1f})")
            res[system] = {"n": len(sc), "mean": float(sc.mean()), "spf_far": far, "ci": [float(lo), float(hi)]}
        summary[f"{ds}_{sp}"] = res
        L.append("")
if any(r["dataset"] == "rail" for r in rows):
    L.append("RAIL test, SPF-FAR per digit (D | D2 if present):")
    R = [r for r in rows if r["dataset"] == "rail" and r["split"] == "test"]
    tar = np.array([r["score"] for r in R if r["label"] == "target"]); non = np.array([r["score"] for r in R if r["label"] == "nontarget"])
    _, thr = eer(tar, non)
    dig = {s["utt_id"]: s["digit"] for s in spoofs}
    for d in range(10):
        parts = []
        for system in ["D", "D2"]:
            sc = np.array([r["score"] for r in R if r["system"] == system and dig.get(r["utterance_id"]) == str(d)])
            if len(sc): parts.append(f"{system}: n={len(sc)} SPF-FAR={100*(sc >= thr).mean():.1f}%")
        if parts: L.append(f"   digit {d}:  " + "   ".join(parts))
txt = "\n".join(L)
open(os.path.join(RES, "asv_results_with_ch.txt"), "w", encoding="utf-8").write(txt)
json.dump(summary, open(os.path.join(RES, "asv_results.json"), "w"), indent=2)
print("\n" + txt + f"\n\nSaved to {RES}")
