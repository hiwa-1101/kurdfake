# KurdFake - select_model.py   (locked rule 1: choose the final model by DEV only, on one identical dev subset)
# Rebuilds exactly the fixed 3,000-utterance dev subset used by train_adf.py (bona fide vs the four training
# systems, raw and +ch) and computes every run's EER on it from the run's own saved dev scores. No GPU, no test data.
#   python select_model.py      -> results\model_selection_dev.txt
import os, csv, random
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np
ROOT = KF_ROOT; DATA = os.path.join(ROOT, "Dataset"); FT = os.path.join(ROOT, "results", "ft")
SETS = {"rail": os.path.join(DATA, "RAIL_clean"), "cv": os.path.join(DATA, "CV_ckb_clean")}
SPOOF = [os.path.join(DATA, "Spoofs", f) for f in ["spoof_manifest_knnvc_final.csv", "spoof_manifest_knnvc_D2.csv",
                                                   "spoof_manifest_tts_vekol.csv", "spoof_manifest_f5.csv"]]
CM = os.path.join(DATA, "Spoofs_CM", "spoof_manifest_channel_matched.csv")
TRAIN_SYSTEMS = {"D", "G", "C1", "C1D"}
base = lambda s: s[:-3] if s.endswith("+ch") else s

items = []
for ds, b in SETS.items():
    with open(os.path.join(b, "manifest.csv"), encoding="utf-8") as fh:
        items += [(r["utt_id"], ds, r["split"], "bonafide", 1) for r in csv.DictReader(fh) if r["split"] in ("adapt", "dev", "test")]
for mp in SPOOF + [CM]:
    with open(mp, encoding="utf-8") as fh:
        items += [(r["utt_id"], r["dataset"], r["split"], r["system"], 0) for r in csv.DictReader(fh)]
dev = [i for i in items if i[2] == "dev" and (i[3] == "bonafide" or base(i[3]) in TRAIN_SYSTEMS)]
subset = random.Random(1234).sample(dev, min(3000, len(dev)))
lab = {i[0]: i[4] for i in subset}
print(f"dev subset: {len(subset)} utterances ({sum(lab.values())} bona fide)")

def eer(pos, neg):
    s = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y) / len(pos); fp = np.cumsum(1 - y) / len(neg); fnr = 1 - tp
    k = np.argmin(np.abs(fnr - fp)); return (fnr[k] + fp[k]) / 2

rows = []
for run in sorted(os.listdir(FT)):
    if run.endswith("_smoke"):
        continue
    for fn in ("scores.csv", "scores_v4eval.csv"):
        p = os.path.join(FT, run, fn)
        if not os.path.exists(p):
            continue
        sc = {}
        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["utterance_id"] in lab:
                    sc[r["utterance_id"]] = float(r["score"])
        if len(sc) < len(lab):
            rows.append((run, fn, None, f"only {len(sc)}/{len(lab)} subset utterances scored (older run without +ch dev) - not comparable"))
            continue
        pos = np.array([sc[u] for u in lab if lab[u] == 1]); neg = np.array([sc[u] for u in lab if lab[u] == 0])
        rows.append((run, fn, eer(pos, neg), ""))
rows.sort(key=lambda r: (r[2] is None, r[2] if r[2] is not None else 9))
L = ["Model selection on the identical fixed dev subset (raw + ch, training systems only). Locked rule: lowest dev EER wins.", ""]
for run, fn, e, note in rows:
    L.append(f"  {('%.2f%%' % (100 * e)) if e is not None else '   -  ':>8}  {run}  [{fn}]  {note}")
txt = "\n".join(L); print(txt)
open(os.path.join(ROOT, "results", "model_selection_dev.txt"), "w", encoding="utf-8").write(txt + "\n")
