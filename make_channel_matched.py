# =====================================================================
# KurdFake - make_channel_matched.py   (builds the CHANNEL-MATCHED spoof condition, all splits, both datasets)
#
# Why: channel_test.py showed that the fine-tuned AASIST reached ~0% EER on RAIL by recognising the RAIL
# recording channel (hum, noise floor, level, clipping), not synthesis: once spoofs carried the real channel,
# EER rose from 0.2% to 39.7%. KurdFake therefore ships a channel-matched copy of every spoof:
#   * test/dev : the official HARD condition (a detector must work here, not only on clean spoofs)
#   * adapt    : a training augmentation (used by train_cm.py --aug ch)
# Recipe (same as channel_test.py): take a real recording of the SAME target speaker
#   adapt -> one of that speaker's adapt (training) bona fide clips; dev/test -> one of the speaker's ENROLLMENT
#   clips (never a test trial). Keep its real lead-in and tail, replace its speech region by the spoof scaled to
#   the same speech RMS plus that clip's own background noise looped underneath, then clip at full scale.
# Deterministic (seed = CRC32 of the spoof id). Spoof originals are not touched.
# v2 (23 Nov): background loops and the joins to the real lead-in/tail use 20/10 ms crossfades,
#     because v1 produced small audible clicks at hard splice points (found by listening).
# v4 (23 Nov): no more cut-and-paste into the real clip. The real background (lead-in + tail, joined with a
#     crossfade, 20 ms file edges removed) is looped smoothly under the WHOLE output and the spoof is added on top;
#     v2/v3 still clicked at the background joins (found by step-by-step listening with diag_clicks.py).
# v3 (23 Nov): the spoof is scaled to the real speech RMS but never above the real clip's peak; in v2 peaky
#     synthetic speech was pushed past full scale and hard-clipped, which still gave faint clicks.
#
#   python make_channel_matched.py           (about 15-25 min)
# Writes: Dataset\Spoofs_CM\<system>\<dataset>\<speaker>\<id>.wav
#         Dataset\Spoofs_CM\spoof_manifest_channel_matched.csv   (system names get the suffix '+ch')
# =====================================================================
import os, csv, wave, time, zlib
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np

ROOT = KF_ROOT
DATA = os.path.join(ROOT, "Dataset")
SETS = {"rail": os.path.join(DATA, "RAIL_clean"), "cv": os.path.join(DATA, "CV_ckb_clean")}
OUT = os.path.join(DATA, "Spoofs_CM")
SPOOF_MANIFESTS = [os.path.join(DATA, "Spoofs", f) for f in ["spoof_manifest_knnvc_final.csv", "spoof_manifest_knnvc_D2.csv",
                                                             "spoof_manifest_tts_vekol.csv", "spoof_manifest_f5.csv"]]
SR, PAD_S = 16000, 0.05

def load(path):
    with wave.open(path, "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return x.mean(axis=1).astype(np.float32) / 32768.0

def save(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 32767 / 32768) * 32768).astype(np.int16).tobytes())

refs = {}                                   # (dataset, speaker, pool) -> rows ; pool = 'adapt' | 'enroll'
for ds, base in SETS.items():
    with open(os.path.join(base, "manifest.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r.get("onset_s") or not r.get("offset_s"):
                continue
            if r["split"] == "adapt":
                refs.setdefault((ds, r["speaker"], "adapt"), []).append(r)
            elif r["split"] in ("dev", "test") and r["role"] == "enroll":
                refs.setdefault((ds, r["speaker"], "enroll"), []).append(r)

XF = int(0.02 * SR)                       # 20 ms raised-cosine crossfade (v2: no splice clicks)

def xfade(a, b, f):
    f = min(f, len(a), len(b))
    if f < 2:
        return np.concatenate([a, b])
    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f))
    return np.concatenate([a[:-f], a[-f:] * (1 - w) + b[:f] * w, b[f:]])

def loop_noise(bg, n, rng):
    """background of length n: copies of bg joined with crossfades, then a random crop (no hard joins)."""
    y = bg.copy()
    while len(y) < n + len(bg):
        y = xfade(y, bg, min(XF, len(bg) // 4))
    k = int(rng.integers(0, len(y) - n + 1))
    return y[k:k + n]

cache = {}
def reference(ds, r):
    k = (ds, r["utt_id"])
    if k not in cache:
        if len(cache) > 4000:
            cache.clear()
        x = load(os.path.join(SETS[ds], r["clean_path"]))
        on = int(max(0.0, float(r["onset_s"]) - PAD_S) * SR); off = int(min(len(x) / SR, float(r["offset_s"]) + PAD_S) * SR)
        sp = x[on:off]
        e = int(0.02 * SR)                                    # drop 20 ms at both file edges (record-button transients)
        bgv = xfade(x[e:on], x[off:len(x) - e], XF) if on > e and len(x) - e > off else np.concatenate([x[:on], x[off:]])
        cache[k] = (x, on, off, bgv, float(np.sqrt(np.mean(sp ** 2)) + 1e-9) if len(sp) else 0.05,
                    float(np.max(np.abs(sp))) if len(sp) else 0.5)
    return cache[k]

def transplant(s, ds, r, rng):
    x, on, off, bg, rms_ref, peak_ref = reference(ds, r)
    g = rms_ref / (np.sqrt(np.mean(s ** 2)) + 1e-9)
    g = min(g, 0.98 * peak_ref / (np.max(np.abs(s)) + 1e-9))   # v3: never louder than the real clip's own peak
    s = s * g
    n = on + len(s) + (len(x) - off)                          # same lead-in / tail durations as the real clip
    y = np.zeros(n, dtype=np.float32); y[on:on + len(s)] = s
    if len(bg) >= int(0.1 * SR):
        y = y + loop_noise(bg, n, rng)                        # one continuous real background under everything (v4)
    f = int(0.005 * SR); w = np.linspace(0, 1, f)
    y[:f] *= w; y[-f:] *= w[::-1]                             # 5 ms fade at the file edges
    return np.clip(y, -1.0, 32767 / 32768)

rows_out, skipped, t0 = [], 0, time.time()
spoofs = []
for mp in SPOOF_MANIFESTS:
    with open(mp, encoding="utf-8") as fh:
        spoofs += list(csv.DictReader(fh))
print(f"spoofs to process: {len(spoofs)}")
for i, r in enumerate(spoofs):
    ds = r["dataset"]; pool = "adapt" if r["split"] == "adapt" else "enroll"
    cand = refs.get((ds, r["target_speaker"], pool))
    if not cand:
        skipped += 1; continue
    rng = np.random.default_rng(zlib.crc32(r["utt_id"].encode()))
    ref = cand[int(rng.integers(0, len(cand)))]
    rel = os.path.join(r["system"] + "+ch", ds, r["target_speaker"], r["utt_id"] + "_ch.wav")
    dst = os.path.join(OUT, rel)
    if not os.path.exists(dst):
        save(dst, transplant(load(os.path.join(DATA, "Spoofs", r["path"])), ds, ref, rng))
    o = dict(r); o.update(utt_id=r["utt_id"] + "_ch", system=r["system"] + "+ch", path=rel, channel_ref=ref["utt_id"], source_spoof=r["utt_id"])
    rows_out.append(o)
    if i % 5000 == 0:
        print(f"  {i}/{len(spoofs)}  ({(time.time()-t0)/60:.1f} min)", flush=True)

cols = list(rows_out[0].keys())
for o in rows_out:
    for c in o:
        if c not in cols:
            cols.append(c)
with open(os.path.join(OUT, "spoof_manifest_channel_matched.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows_out)
from collections import Counter
c = Counter((o["dataset"], o["system"], o["split"]) for o in rows_out)
print("\n".join(f"  {k}: {v}" for k, v in sorted(c.items())))
print(f"Done. {len(rows_out)} channel-matched spoofs, {skipped} skipped (no reference clip).  "
      f"Manifest: {os.path.join(OUT, 'spoof_manifest_channel_matched.csv')}   ({(time.time()-t0)/60:.1f} min)")
