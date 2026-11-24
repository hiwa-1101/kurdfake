# KurdFake - diag_clicks.py   (find WHICH step of the channel transplant makes the faint clicks)
# For the 3 listening sets in results\listen_channel it rebuilds the +ch file step by step and saves:
#   A_level_only   the spoof alone, only made louder exactly as in +ch     (no noise, no real lead-in/tail)
#   B_level_noise  A + the looped real background noise under it           (no real lead-in/tail)
#   C_full         B placed inside the real recording (= the +ch file, v3)
#   V4_new         the new v4 recipe: one continuous real background under the whole file, spoof added on top
# If A already clicks, the clicks are inside the spoof itself and only become audible when it is louder.
#   python diag_clicks.py     -> results\diag_clicks\   (9 files, a few seconds)
import os, csv, wave, zlib
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np
ROOT = KF_ROOT; DATA = os.path.join(ROOT, "Dataset"); RAIL = os.path.join(DATA, "RAIL_clean")
OUT = os.path.join(ROOT, "results", "diag_clicks"); os.makedirs(OUT, exist_ok=True)
SR, PAD_S, XF = 16000, 0.05, int(0.02 * 16000)

def load(p):
    with wave.open(p, "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return x.mean(axis=1).astype(np.float32) / 32768.0

def save(p, x):
    with wave.open(p, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 32767 / 32768) * 32768).astype(np.int16).tobytes())

def xfade(a, b, f):
    f = min(f, len(a), len(b))
    if f < 2:
        return np.concatenate([a, b])
    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f))
    return np.concatenate([a[:-f], a[-f:] * (1 - w) + b[:f] * w, b[f:]])

def loop_noise(bg, n, rng):
    y = bg.copy()
    while len(y) < n + len(bg):
        y = xfade(y, bg, min(XF, len(bg) // 4))
    k = int(rng.integers(0, len(y) - n + 1)); return y[k:k + n]

rail = {}
with open(os.path.join(RAIL, "manifest.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        rail[r["utt_id"]] = r
spoof = {}
for f in ["spoof_manifest_knnvc_final.csv", "spoof_manifest_tts_vekol.csv", "spoof_manifest_f5.csv"]:
    with open(os.path.join(DATA, "Spoofs", f), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            spoof[r["utt_id"]] = r
cm = {}
with open(os.path.join(DATA, "Spoofs_CM", "spoof_manifest_channel_matched.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        cm[r["path"].replace("\\", "/").split("/")[-1]] = r
sets = sorted(f for f in os.listdir(os.path.join(ROOT, "results", "listen_channel")) if f.endswith("_3_FAKE_channel.wav"))
print("listening sets:", sets)
# find the +ch rows behind the 3 listening files by comparing audio
for fn in sets:
    target = load(os.path.join(ROOT, "results", "listen_channel", fn))
    tag = fn.split("_3_")[0]; system = tag.split("_")[1] + "+ch"; spk = tag.split("spk")[1]
    row = None
    for r in cm.values():
        if r["system"] == system and r["target_speaker"] == spk and r["split"] == "test":
            y = load(os.path.join(DATA, "Spoofs_CM", r["path"]))
            if len(y) == len(target) and np.allclose(y, target, atol=1e-4):
                row = r; break
    if row is None:
        print("  could not match", fn); continue
    s0 = load(os.path.join(DATA, "Spoofs", spoof[row["source_spoof"]]["path"]))
    ref = rail[row["channel_ref"]]; x = load(os.path.join(RAIL, ref["clean_path"]))
    on = int(max(0.0, float(ref["onset_s"]) - PAD_S) * SR); off = int(min(len(x) / SR, float(ref["offset_s"]) + PAD_S) * SR)
    sp = x[on:off]; bg = np.concatenate([x[:on], x[off:]])
    g = min(np.sqrt(np.mean(sp ** 2)) / (np.sqrt(np.mean(s0 ** 2)) + 1e-9), 0.98 * np.max(np.abs(sp)) / (np.max(np.abs(s0)) + 1e-9))
    rng = np.random.default_rng(zlib.crc32(row["source_spoof"].encode()))
    _ = rng.integers(0, 1)                         # same draw order as make_channel_matched (reference choice)
    A = s0 * g
    B = A + (loop_noise(bg, len(A), rng) if len(bg) >= int(0.1 * SR) else 0)
    C = xfade(xfade(x[:on], B, XF // 2), x[off:], XF // 2)
    e = int(0.02 * SR)                                              # v4 recipe (as in make_channel_matched.py v4)
    bgv = xfade(x[e:on], x[off:len(x) - e], XF) if on > e and len(x) - e > off else bg
    n = on + len(A) + (len(x) - off); V = np.zeros(n, dtype=np.float32); V[on:on + len(A)] = A
    if len(bgv) >= int(0.1 * SR):
        V = V + loop_noise(bgv, n, np.random.default_rng(1))
    f5 = int(0.005 * SR); w5 = np.linspace(0, 1, f5); V[:f5] *= w5; V[-f5:] *= w5[::-1]
    for name, z in (("A_level_only", A), ("B_level_noise", B), ("C_full", C), ("V4_new", V)):
        save(os.path.join(OUT, f"{tag}_{name}.wav"), z)
    print(f"  {tag}: gain x{g:.1f}  spoof peak {np.max(np.abs(s0)):.2f} -> {np.max(np.abs(A)):.2f}  "
          f"real peak {np.max(np.abs(sp)):.2f}  background {len(bg)/SR:.2f} s")
print(f"saved to {OUT}")
