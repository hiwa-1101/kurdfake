# =====================================================================
# KurdFake - prepare_cv.py
# Builds the clean, paper-ready Common Voice Central Kurdish (ckb) speaker set.
#
# Usage (Windows, from <KURDFAKE_ROOT>):
#     py -m pip install numpy scipy soundfile matplotlib
#     py prepare_cv.py
#
# CV_ROOT must contain the extracted corpus (the folder that has validated.tsv and clips\).
# The script searches below CV_ROOT for validated.tsv, so the exact sub-folder name does not matter.
# Never modifies or deletes the original files. All output goes to OUT_DIR.
# =====================================================================

import os, csv, json, random, hashlib, collections, datetime, sys
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

# ---------------- configuration ----------------
CV_ROOT  = os.path.join(KF_ROOT, "Dataset", "CommonVoice_ckb")
OUT_DIR  = os.path.join(KF_ROOT, "Dataset", "CV_ckb_clean")
SEED     = 1337
CV_VERSION = "27.0"

MIN_CLIPS_PER_SPK = 20        # speakers need at least this many validated clips
N_SPEAKERS        = 200       # target number of speakers (all eligible if fewer)
MAX_CLIPS_PER_SPK = 50        # cap so a few heavy contributors do not dominate
SPLIT_PATTERN = ["adapt"] * 6 + ["dev"] + ["test"] * 3      # 60/10/30 by speaker
N_ENROLL      = 5             # dev/test: first 5 (random) clips per speaker = enrollment
NONTARGET_PER_TRIAL = 2

SR_OUT       = 16000
VAD_HP_HZ, VAD_MARGIN_DB, VAD_ABS_DB = 100, 10.0, -45.0
FRAME, HOP   = 400, 160
MIN_SPEECH_S = 0.5            # less speech than this -> excluded
MAX_DUR_S    = 15.0           # longer clips -> excluded
CLIP_FRAC    = 0.001
csv.field_size_limit(10_000_000)
# -----------------------------------------------

rng = random.Random(SEED); np.random.seed(SEED)

def find_tsv(root):
    for dp, _, fs in os.walk(root):
        if "validated.tsv" in fs and os.path.isdir(os.path.join(dp, "clips")):
            return dp
    return None

def map_gender(g):
    g = (g or "").strip().lower()
    if g.startswith("male") or g == "m": return "Male"
    if g.startswith("female") or g == "f": return "Female"
    return "unknown"

def to_16k(x, sr):
    if x.ndim > 1: x = x.mean(axis=1)
    if sr != SR_OUT:
        g = gcd(sr, SR_OUT); x = resample_poly(x, SR_OUT // g, sr // g)
    return x.astype(np.float32)

def quality(m):
    X = np.fft.rfft(m); X[np.fft.rfftfreq(len(m), 1 / SR_OUT) < VAD_HP_HZ] = 0
    y = np.fft.irfft(X, n=len(m))
    q = {"dur_s": len(m) / SR_OUT, "peak_dbfs": float(20 * np.log10(np.max(np.abs(m)) + 1e-9)),
         "clip_frac": float(np.mean(np.abs(m) >= 0.999))}
    if len(y) < FRAME:
        q.update(speech_s=0.0, onset_s=None, offset_s=None); return q
    fr = np.lib.stride_tricks.sliding_window_view(y, FRAME)[::HOP]
    e = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-10)
    sp = e > max(np.percentile(e, 10) + VAD_MARGIN_DB, VAD_ABS_DB)
    sp = sp & (np.convolve(sp.astype(int), [1, 1, 1], "same") >= 2)
    q["speech_s"] = float(sp.sum()) * HOP / SR_OUT
    idx = np.where(sp)[0]
    q["onset_s"] = idx[0] * HOP / SR_OUT if len(idx) else None
    q["offset_s"] = (idx[-1] * HOP + FRAME) / SR_OUT if len(idx) else None
    return q

# ---------------- 1. read validated.tsv ----------------
base = find_tsv(CV_ROOT)
if not base:
    sys.exit(f"validated.tsv + clips folder not found under {CV_ROOT}. Extract the Common Voice archive there first.")
print(f"[1] corpus folder: {base}")
with open(os.path.join(base, "validated.tsv"), encoding="utf-8") as fh:
    val = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
by_spk = collections.defaultdict(list)
for r in val:
    by_spk[r["client_id"]].append(r)
n_all_spk = len(by_spk)
print(f"[1] validated clips: {len(val)}  speakers: {n_all_spk}")

# official CV split membership (for reporting only)
official = {}
for name in ["train", "dev", "test"]:
    p = os.path.join(base, f"{name}.tsv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
                official[r["path"]] = name

# ---------------- 2. choose speakers ----------------
def spk_gender(rows):
    c = collections.Counter(map_gender(r.get("gender")) for r in rows)
    known = [(n, g) for g, n in c.items() if g != "unknown"]
    return max(known)[1] if known else "unknown"
def spk_age(rows):
    c = collections.Counter((r.get("age") or "").strip() for r in rows if (r.get("age") or "").strip())
    return c.most_common(1)[0][0] if c else ""
eligible = sorted([s for s, rs in by_spk.items() if len(rs) >= MIN_CLIPS_PER_SPK])
print(f"[2] speakers with >= {MIN_CLIPS_PER_SPK} validated clips: {len(eligible)}")
gender_of = {s: spk_gender(by_spk[s]) for s in eligible}
age_of = {s: spk_age(by_spk[s]) for s in eligible}
if len(eligible) > N_SPEAKERS:
    groups = collections.defaultdict(list)
    for s in eligible: groups[gender_of[s]].append(s)
    for g in groups: rng.shuffle(groups[g])
    chosen, gi = [], {g: 0 for g in groups}
    order = sorted(groups, key=lambda g: (g == "unknown", g))      # balance Male/Female first
    while len(chosen) < N_SPEAKERS:
        progressed = False
        for g in order:
            if gi[g] < len(groups[g]) and len(chosen) < N_SPEAKERS:
                chosen.append(groups[g][gi[g]]); gi[g] += 1; progressed = True
        if not progressed: break
else:
    chosen = eligible[:]
chosen.sort()
spk_id = {s: f"cv{i+1:04d}" for i, s in enumerate(chosen)}
print(f"[2] chosen speakers: {len(chosen)}  gender: {dict(collections.Counter(gender_of[s] for s in chosen))}")

# ---------------- 3. convert + quality ----------------
os.makedirs(OUT_DIR, exist_ok=True)
rows, seen = [], {}
for k, s in enumerate(chosen):
    rs = by_spk[s][:]
    rng.shuffle(rs)
    rs = rs[:MAX_CLIPS_PER_SPK]
    for j, r in enumerate(rs):
        sid = spk_id[s]
        o = {"utt_id": f"{sid}_{j:03d}", "speaker": sid, "gender": gender_of[s], "age": age_of[s],
             "orig_path": r["path"], "official_split": official.get(r["path"], ""), "sentence": r.get("sentence", ""),
             "sentence_id": r.get("sentence_id", ""), "up_votes": r.get("up_votes", ""), "down_votes": r.get("down_votes", ""),
             "exclude": 0, "reason": []}
        try:
            x, sr = sf.read(os.path.join(base, "clips", r["path"]), dtype="float32", always_2d=False)
            m = to_16k(x, sr)
            o["orig_sr"] = sr
            o.update(quality(m))
            o["md5"] = hashlib.md5((np.clip(m, -1, 1) * 32767).astype(np.int16).tobytes()).hexdigest()
            if o["md5"] in seen: o["reason"].append("duplicate_audio"); o["exclude"] = 1
            seen.setdefault(o["md5"], o["utt_id"])
            if o["speech_s"] < MIN_SPEECH_S: o["reason"].append("too_little_speech"); o["exclude"] = 1
            if o["dur_s"] > MAX_DUR_S: o["reason"].append("too_long"); o["exclude"] = 1
            o["flag_clipped"] = int(o["clip_frac"] > CLIP_FRAC)
            o["clean_path"] = os.path.join("wav", sid, o["utt_id"] + ".wav")
            os.makedirs(os.path.join(OUT_DIR, "wav", sid), exist_ok=True)
            sf.write(os.path.join(OUT_DIR, o["clean_path"]), m, SR_OUT, subtype="PCM_16")
        except Exception as ex:
            o["reason"].append("read_error:" + type(ex).__name__); o["exclude"] = 1
        rows.append(o)
    if (k + 1) % 20 == 0 or k + 1 == len(chosen):
        print(f"[3] {k+1}/{len(chosen)} speakers, {len(rows)} clips", flush=True)

# ---------------- 4. speaker-disjoint split ----------------
kept = [r for r in rows if not r["exclude"]]
spk_kept = collections.Counter(r["speaker"] for r in kept)
spks = [spk_id[s] for s in chosen if spk_kept[spk_id[s]] >= N_ENROLL + 5]
for r in kept:
    if r["speaker"] not in spks:
        r["exclude"] = 1; r["reason"].append("speaker_too_few_clips_after_qc")
kept = [r for r in rows if not r["exclude"]]
g_of = {spk_id[s]: gender_of[s] for s in chosen}
order = spks[:]; rng.shuffle(order); order.sort(key=lambda s: g_of[s])
off = rng.randrange(len(SPLIT_PATTERN))
split_of = {s: SPLIT_PATTERN[(i + off) % len(SPLIT_PATTERN)] for i, s in enumerate(order)}
cnt = collections.Counter()
for r in kept:
    r["split"] = split_of[r["speaker"]]
    if r["split"] == "adapt":
        r["role"] = "train"
    else:
        cnt[r["speaker"]] += 1
        r["role"] = "enroll" if cnt[r["speaker"]] <= N_ENROLL else "trial"

# ---------------- 5. trial lists ----------------
trial_stats = {}
for sp in ["dev", "test"]:
    ss = [s for s in spks if split_of[s] == sp]
    pool = collections.defaultdict(list)
    for s in ss: pool[g_of[s]].append(s)
    out = []
    for r in kept:
        if r["split"] != sp or r["role"] != "trial": continue
        out.append((r["speaker"], r["utt_id"], "target"))
        cand = [s for s in (pool[r["gender"]] if r["gender"] != "unknown" else ss) if s != r["speaker"]]
        for s in rng.sample(cand, min(NONTARGET_PER_TRIAL, len(cand))):
            out.append((s, r["utt_id"], "nontarget"))
    with open(os.path.join(OUT_DIR, f"trials_{sp}.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["enroll_speaker", "test_utt_id", "label"]); w.writerows(out)
    trial_stats[sp] = dict(collections.Counter(t[2] for t in out))

# ---------------- 6. manifests ----------------
cols = ["utt_id", "clean_path", "orig_path", "speaker", "gender", "age", "split", "role", "official_split",
        "sentence_id", "sentence", "up_votes", "down_votes", "exclude", "reason", "dur_s", "speech_s",
        "onset_s", "offset_s", "peak_dbfs", "clip_frac", "flag_clipped", "orig_sr", "md5"]
def fmt(r, c):
    v = r.get(c, "")
    if c == "reason": return ";".join(v)
    if isinstance(v, float): return round(v, 4)
    return "" if v is None else v
for name, data in [("manifest_all.csv", rows), ("manifest.csv", kept)]:
    with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in data: w.writerow([fmt(r, c) for c in cols])
with open(os.path.join(OUT_DIR, "speakers.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["speaker", "client_id_sha1_8", "gender", "age", "split", "n_utts", "minutes"])
    inv = {v: k for k, v in spk_id.items()}
    for s in spks:
        rs = [r for r in kept if r["speaker"] == s]
        w.writerow([s, hashlib.sha1(inv[s].encode()).hexdigest()[:8], g_of[s], age_of[inv[s]], split_of[s],
                    len(rs), round(sum(r["dur_s"] for r in rs) / 60, 2)])

# ---------------- 7. summary ----------------
L = [f"KurdFake Common Voice ckb {CV_VERSION} preparation  {datetime.datetime.now():%Y-%m-%d %H:%M}  seed={SEED}", ""]
L.append(f"validated clips: {len(val)}   speakers in validated: {n_all_spk}")
L.append(f"speakers with >= {MIN_CLIPS_PER_SPK} clips: {len(eligible)}   chosen: {len(chosen)}   kept after QC: {len(spks)}")
L.append(f"clips processed: {len(rows)}   kept: {len(kept)}   excluded: {len(rows) - len(kept)}")
for k, v in collections.Counter(x.split(':')[0] for r in rows for x in r["reason"]).most_common():
    L.append(f"    {k}: {v}")
d = sorted(r["dur_s"] for r in kept); sp_ = sorted(r["speech_s"] for r in kept)
if d:
    L.append(f"duration (s): p5={d[int(.05*len(d))]:.2f} median={d[len(d)//2]:.2f} p95={d[int(.95*len(d))]:.2f}   total={sum(d)/3600:.2f} h")
    L.append(f"speech (s):   p5={sp_[int(.05*len(sp_))]:.2f} median={sp_[len(sp_)//2]:.2f}")
L.append(f"original sample rates: {dict(collections.Counter(r.get('orig_sr') for r in kept))}")
L.append(f"clipped (>0.1%): {sum(r['flag_clipped'] for r in kept)}")
L.append(f"unique sentences: {len({r['sentence'] for r in kept})}")
L.append("")
L.append(f"{'split':6s} {'spk':>4s} {'male':>5s} {'female':>6s} {'unk':>4s} {'utts':>6s} {'enroll':>6s} {'trial':>6s} {'hours':>6s}")
for sp in ["adapt", "dev", "test"]:
    ss = [s for s in spks if split_of[s] == sp]; us = [r for r in kept if r["split"] == sp]
    L.append(f"{sp:6s} {len(ss):4d} {sum(g_of[s]=='Male' for s in ss):5d} {sum(g_of[s]=='Female' for s in ss):6d} "
             f"{sum(g_of[s]=='unknown' for s in ss):4d} {len(us):6d} {sum(r['role']=='enroll' for r in us):6d} "
             f"{sum(r['role']=='trial' for r in us):6d} {sum(r['dur_s'] for r in us)/3600:6.2f}")
for sp in ["dev", "test"]:
    L.append(f"trials_{sp}: {trial_stats[sp]}")
sent_split = collections.defaultdict(set)
for r in kept: sent_split[r["sentence"]].add(r["split"])
L.append(f"sentences shared between adapt and test: {sum(1 for v in sent_split.values() if 'adapt' in v and 'test' in v)}")
summary = "\n".join(L)
open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8").write(summary)
json.dump({"cv_version": CV_VERSION, "seed": SEED, "min_clips": MIN_CLIPS_PER_SPK, "n_speakers": N_SPEAKERS,
           "max_clips": MAX_CLIPS_PER_SPK, "split_pattern": SPLIT_PATTERN, "n_enroll": N_ENROLL,
           "min_speech_s": MIN_SPEECH_S, "max_dur_s": MAX_DUR_S},
          open(os.path.join(OUT_DIR, "config_used.json"), "w"), indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].hist(d, bins=40, color="#4C72B0"); ax[0].set_title("Clip duration (s)")
    ax[1].hist(sp_, bins=40, color="#DD8452"); ax[1].set_title("Speech per clip (s)")
    c = collections.Counter(len(v) for v in by_spk.values())
    xs = sorted(c); ax[2].loglog(xs, [c[x] for x in xs], "o", ms=3); ax[2].set_title("Validated clips per speaker (all CV ckb)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "summary_plots.png"), dpi=130)
except Exception as ex:
    print("plots skipped:", ex)
print("\n" + summary + f"\n\nDone. Output folder: {OUT_DIR}")
