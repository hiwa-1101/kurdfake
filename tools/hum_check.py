# KurdFake - hum_check.py   (objective check of the listening notes; reads files only, changes nothing)
# For RAIL test: real recordings, original spoofs and channel-matched (+ch) spoofs, per system, it measures
#   hum_db  : share of energy below 100 Hz (dB; 0 = all energy is hum; RAIL real median was about -3.3 dB)
#   clicks  : sample-to-sample jumps larger than 12x the clip's median jump, per second (splice clicks)
# on a random sample of 300 files per group.      python hum_check.py   (about 2-3 minutes)
import os, csv, wave, random
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np
ROOT = KF_ROOT; DATA = os.path.join(ROOT, "Dataset")
SR = 16000

def load(p):
    with wave.open(p, "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return x.mean(axis=1).astype(np.float64) / 32768.0

def measure(x):
    X = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(x), 1 / SR)
    hum = 10 * np.log10(X[f < 100].sum() / (X.sum() + 1e-12) + 1e-12)
    d = np.abs(np.diff(x)); thr = 12 * (np.median(d) + 1e-6)
    return hum, (d > thr).sum() / (len(x) / SR)

groups = {}
with open(os.path.join(DATA, "RAIL_clean", "manifest.csv"), encoding="utf-8") as fh:
    groups["REAL"] = [os.path.join(DATA, "RAIL_clean", r["clean_path"]) for r in csv.DictReader(fh) if r["split"] == "test" and r["role"] == "trial"]
for f in ["spoof_manifest_knnvc_final.csv", "spoof_manifest_tts_vekol.csv", "spoof_manifest_f5.csv"]:
    with open(os.path.join(DATA, "Spoofs", f), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["dataset"] == "rail" and r["split"] == "test":
                groups.setdefault(r["system"], []).append(os.path.join(DATA, "Spoofs", r["path"]))
with open(os.path.join(DATA, "Spoofs_CM", "spoof_manifest_channel_matched.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["dataset"] == "rail" and r["split"] == "test" and r["system"] in ("D+ch", "C1D+ch", "BC+ch", "B+ch", "C1+ch"):
            groups.setdefault(r["system"], []).append(os.path.join(DATA, "Spoofs_CM", r["path"]))
print(f"{'group':8s} {'n':>4s}  {'hum_db median':>13s}  {'hum_db p25..p75':>16s}  {'clicks/s median':>15s}  {'files with clicks':>17s}")
for g in ["REAL", "D", "D+ch", "C1D", "C1D+ch", "BC", "BC+ch", "B", "B+ch", "C1", "C1+ch"]:
    if g not in groups:
        continue
    fs = random.Random(1).sample(groups[g], min(300, len(groups[g])))
    m = np.array([measure(load(p)) for p in fs])
    print(f"{g:8s} {len(fs):4d}  {np.median(m[:,0]):13.1f}  {np.percentile(m[:,0],25):7.1f}..{np.percentile(m[:,0],75):6.1f}  "
          f"{np.median(m[:,1]):15.1f}  {100*np.mean(m[:,1] > 0):16.0f}%")
