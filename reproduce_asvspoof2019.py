# =====================================================================
# KurdFake - reproduce_asvspoof2019.py   (pipeline sanity check before trusting RQ2)
# Scores the official AASIST / AASIST-L weights on ASVspoof 2019 LA EVAL with EXACTLY the same
# loading, padding and scoring code path that cm_eval.py uses for Kurdish.
# Published (Jung et al., ICASSP 2022): AASIST EER 0.83%, AASIST-L EER 0.99%  (LA eval).
#
# Reads LA.zip directly (no need to extract 7.6 GB).  Needs soundfile (installed from wheels).
# Anaconda Prompt:
#   conda activate gpu_env
#   cd /d <KURDFAKE_ROOT>
#   python reproduce_asvspoof2019.py            (full eval set, 71,237 files)
#   python reproduce_asvspoof2019.py --quick    (random 5,000 files, ~5 min, for a first check)
# =====================================================================
import os, io, sys, json, time, zipfile, random, argparse, platform
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np
import soundfile as sf
import torch

LA_ZIP = os.path.join(KF_ROOT, "Dataset", "LA.zip")
AASIST = os.path.join(KF_ROOT, "models", "aasist")
RES    = os.path.join(KF_ROOT, "results")
MODELS = ["AASIST", "AASIST-L"]
PUBLISHED = {"AASIST": 0.83, "AASIST-L": 0.99}
NB_SAMP, BATCH, SEED = 64600, 16, 1337

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); args = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kf_metrics import eer, min_dcf
import importlib.util
_spec = importlib.util.spec_from_file_location("aasist_model", os.path.join(AASIST, "models", "AASIST.py"))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod); AASISTModel = _mod.Model
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"python {platform.python_version()}  torch {torch.__version__}  device {dev}")
os.makedirs(RES, exist_ok=True)

def pad(x, max_len=NB_SAMP):           # identical to cm_eval.pad and aasist/data_utils.pad
    if len(x) >= max_len:
        return x[:max_len]
    return np.tile(x, int(max_len / len(x)) + 1)[:max_len]

z = zipfile.ZipFile(LA_ZIP)
names = z.namelist()
proto = [n for n in names if n.endswith("ASVspoof2019.LA.cm.eval.trl.txt")][0]
flac = {os.path.basename(n)[:-5]: n for n in names if "ASVspoof2019_LA_eval/flac/" in n and n.endswith(".flac")}
items = []
for line in z.read(proto).decode().splitlines():
    p = line.split()                     # SPEAKER UTT - ATTACK KEY
    items.append((p[1], p[3], p[4]))     # utt, attack (A07..A19 or -), bonafide/spoof
print(f"protocol: {len(items)} trials, flac files in zip: {len(flac)}")
if args.quick:
    random.Random(SEED).shuffle(items); items = items[:5000]

out_lines = {}
summary = {}
for mname in MODELS:
    cfg = json.load(open(os.path.join(AASIST, "config", f"{mname}.conf")))
    model = AASISTModel(cfg["model_config"])
    model.load_state_dict(torch.load(os.path.join(AASIST, "models", "weights", f"{mname}.pth"), map_location="cpu"))
    model = model.to(dev).eval()
    scores = {}
    t0 = time.time()
    for b in range(0, len(items), BATCH):
        chunk = items[b:b + BATCH]
        xs = []
        for (u, _a, _k) in chunk:
            x, sr = sf.read(io.BytesIO(z.read(flac[u])), dtype="float32")
            assert sr == 16000
            xs.append(pad(x))
        with torch.no_grad():
            _, o = model(torch.from_numpy(np.stack(xs)).to(dev))
        for (u, _a, _k), s in zip(chunk, o[:, 1].float().cpu().numpy()):
            scores[u] = float(s)
        if (b // BATCH) % 500 == 0:
            print(f"  {mname}: {b+len(chunk)}/{len(items)}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    bona = np.array([scores[u] for u, a, k in items if k == "bonafide"])
    spoof = np.array([scores[u] for u, a, k in items if k == "spoof"])
    e, _ = eer(bona, spoof); md, _ = min_dcf(bona, spoof)
    per = {}
    for att in sorted({a for _, a, k in items if k == "spoof"}):
        per[att] = 100 * eer(bona, np.array([scores[u] for u, a, k in items if a == att]))[0]
    summary[mname] = {"eer": 100 * e, "min_dcf_asvspoof5": md, "per_attack_eer": per, "n": len(items)}
    with open(os.path.join(RES, f"asvspoof2019_LA_eval_{mname}{'_quick' if args.quick else ''}_scores.txt"), "w") as fh:
        for u, a, k in items:
            fh.write(f"{u} {a} {k} {scores[u]:.6f}\n")

L = [f"ASVspoof 2019 LA eval reproduction  {time.strftime('%Y-%m-%d %H:%M')}  {'QUICK subset 5000' if args.quick else 'FULL'}", ""]
for m, r in summary.items():
    ok = abs(r["eer"] - PUBLISHED[m]) < (0.5 if not args.quick else 1.0)
    L.append(f"{m:9s} EER = {r['eer']:.2f}%   (published {PUBLISHED[m]:.2f}%)   minDCF(ASVspoof5 costs) = {r['min_dcf_asvspoof5']:.4f}   -> {'MATCH' if ok else 'CHECK!'}")
    L.append("          per attack EER: " + "  ".join(f"{a} {v:.2f}" for a, v in r["per_attack_eer"].items()))
txt = "\n".join(L)
open(os.path.join(RES, f"asvspoof2019_reproduction{'_quick' if args.quick else ''}.txt"), "w").write(txt)
print("\n" + txt)
