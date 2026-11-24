# KurdFake - diagnose stereo channels + pick 60 original clips for threshold tuning (read-only)
# Usage:  py diag_channels.py
# Output: diag_channels.txt  and  rail_sample60.zip  (about 4 MB) next to this script
import os, random, wave, zipfile
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np

SRC = os.path.join(KF_ROOT, "Dataset", "RAIL_full")
here = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(7)
files = [os.path.join(SRC, str(d), f) for d in range(10) for f in os.listdir(os.path.join(SRC, str(d))) if f.endswith(".wav")]
pick = rng.sample(files, 400)

def db(v): return 20 * np.log10(v + 1e-9)
L, stats = [], []
for p in pick:
    with wave.open(p, "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels()).astype(np.float32) / 32768
    a, b = x[:, 0], x[:, 1]
    ra, rb, rm = np.sqrt((a**2).mean()), np.sqrt((b**2).mean()), np.sqrt((((a + b) / 2)**2).mean())
    c = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    pa, pb = np.abs(a).max(), np.abs(b).max()
    stats.append((ra, rb, rm, c))
    L.append(f"{os.path.relpath(p, SRC):22s} L_rms={db(ra):7.1f} R_rms={db(rb):7.1f} mix_rms={db(rm):7.1f} "
             f"L_peak={db(pa):6.1f} R_peak={db(pb):6.1f} corr={c:6.3f}")
s = np.array(stats)
head = [f"files analysed: {len(s)}",
        f"corr: min={s[:,3].min():.3f} median={np.median(s[:,3]):.3f} max={s[:,3].max():.3f}",
        f"corr < -0.5 (anti-phase): {(s[:,3] < -0.5).sum()}   |corr| < 0.2: {(np.abs(s[:,3]) < 0.2).sum()}   corr > 0.5: {(s[:,3] > 0.5).sum()}",
        f"mix quieter than louder channel by >6 dB: {(db(s[:,2]) < db(np.maximum(s[:,0], s[:,1])) - 6).sum()}",
        f"L louder than R by >6 dB: {(db(s[:,0]) > db(s[:,1]) + 6).sum()}   R louder than L by >6 dB: {(db(s[:,1]) > db(s[:,0]) + 6).sum()}", ""]
open(os.path.join(here, "diag_channels.txt"), "w").write("\n".join(head + L))
print("\n".join(head))
with zipfile.ZipFile(os.path.join(here, "rail_sample60.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for p in pick[:60]:
        z.write(p, os.path.relpath(p, SRC))
print("Saved diag_channels.txt and rail_sample60.zip")
