# =====================================================================
# KurdFake - prepare_rail.py
# Builds the clean, paper-ready RAIL set from the original files.
#
# Usage (Windows, from <KURDFAKE_ROOT>):
#     py -m pip install numpy matplotlib
#     py prepare_rail.py
#
# Needs next to this script:  speakers_meta.csv  (id, age, gender; no names)
# Never modifies or deletes the original files. All output goes to OUT_DIR.
# =====================================================================

import os, re, csv, wave, json, random, shutil, hashlib, collections, datetime
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np

# ---------------- configuration (edit here if needed) ----------------
SRC_DIR   = os.path.join(KF_ROOT, "Dataset", "RAIL_full")   # contains folders 0..9, all, ...
OUT_DIR   = os.path.join(KF_ROOT, "Dataset", "RAIL_clean")
META_CSV  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speakers_meta.csv")
SEED      = 1337

# Speaker decisions (from the Excel list and the curated "all" folder)
EXCLUDE_IDS = {"000": "template/test recording (Excel: 'templet')",
               "014": "removed in curated 'all' folder (in 'fulti')",
               "045": "removed in curated 'all' folder (in 'fulti')",
               "119": "same person as 002 recorded with a deliberately altered voice"}
MERGE_IDS   = {"041": "006"}            # same person recorded twice -> one speaker
SAME_SPLIT_GROUPS = [("101", "115")]    # uncertain pair: kept separate but forced into the same split

SPLIT_PATTERN = ["adapt"] * 6 + ["dev"] + ["test"] * 3   # 60 / 10 / 30 by speaker
ENROLL_REPS   = {1, 2, 3}               # dev/test: reps 1-3 of each digit = enrollment, rest = trials
NONTARGET_PER_TRIAL = 2                 # same-gender impostor speakers per trial utterance

# Quality thresholds (energy VAD on 100 Hz high-passed signal; tuned on 60 listened clips)
FRAME, HOP    = 400, 160                # 25 ms / 10 ms at 16 kHz
VAD_HP_HZ     = 100                     # removes the 50 Hz mains hum before VAD
VAD_MARGIN_DB = 10.0                    # speech = frame energy > noise floor + 10 dB
VAD_ABS_DB    = -45.0                   # ... and above -45 dBFS
MIN_SPEECH_S  = 0.10                    # less speech than this -> empty (none expected)
MIN_PEAK_DB   = -40.0                   # quieter peak than this -> empty
EDGE_S       = 0.03                     # speech within 30 ms of clip edge -> truncated
CLIP_FRAC    = 0.001                    # >0.1 % full-scale samples -> clipped
EXCLUDE_TRUNCATED = False               # truncated clips are flagged but kept by default
N_REVIEW     = 40                       # flagged clips copied for manual listening
# ---------------------------------------------------------------------

random.seed(SEED); np.random.seed(SEED)
NAME_RE = re.compile(r"^(\d{3})(\d)(\d{1,2})$")

def read_wav(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"unsupported sample width {sw}")
    return sr, ch, np.frombuffer(raw, dtype=np.int16).reshape(-1, ch)

def write_wav(path, x_int16, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(x_int16.astype(np.int16).tobytes())

def to_mono(x):
    """RAIL: speech is on the left channel; the right channel is ~40 dB lower (leakage).
    Averaging would lose 6 dB and add the right-channel noise, so we keep the louder channel."""
    if x.shape[1] == 1:
        return x[:, 0], 0, 1
    rms = np.sqrt((x.astype(np.float64) ** 2).mean(axis=0))
    c = int(np.argmax(rms))
    return x[:, c], c, int(np.array_equal(x[:, 0], x[:, 1]))

def highpass_fft(m, sr, fc):
    X = np.fft.rfft(m)
    X[np.fft.rfftfreq(len(m), 1 / sr) < fc] = 0
    return np.fft.irfft(X, n=len(m))

def quality(m_int16, sr):
    a = m_int16.astype(np.int32)
    m = a.astype(np.float32) / 32768.0
    y = highpass_fft(m, sr, VAD_HP_HZ)                    # RAIL has strong 50 Hz hum
    tot = np.mean(m ** 2) + 1e-12
    q = {"clip_frac": float(np.mean((a >= 32767) | (a <= -32768))),
         "peak_dbfs": float(20 * np.log10(np.max(np.abs(m)) + 1e-9)),
         "rms_dbfs":  float(10 * np.log10(tot)),
         "hum_share_db": float(10 * np.log10((tot - np.mean(y ** 2)) / tot + 1e-12))}   # energy below VAD_HP_HZ
    nf = 1 + (len(y) - FRAME) // HOP if len(y) >= FRAME else 0
    if nf <= 0:
        q.update(speech_s=0.0, onset_s=None, offset_s=None, noise_floor_db=None, touches_start=0, touches_end=0)
        return q
    fr = np.lib.stride_tricks.sliding_window_view(y, FRAME)[::HOP][:nf]
    e = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-10)
    floor = float(np.percentile(e, 10))
    sp = e > max(floor + VAD_MARGIN_DB, VAD_ABS_DB)
    sp = sp & (np.convolve(sp.astype(int), [1, 1, 1], "same") >= 2)
    q["noise_floor_db"] = floor
    q["speech_s"] = float(sp.sum()) * HOP / sr
    if sp.any():
        idx = np.where(sp)[0]
        on, off = idx[0] * HOP / sr, (idx[-1] * HOP + FRAME) / sr
        q.update(onset_s=on, offset_s=off, touches_start=int(on <= EDGE_S),
                 touches_end=int(off >= len(m) / sr - EDGE_S))
    else:
        q.update(onset_s=None, offset_s=None, touches_start=0, touches_end=0)
    return q

def age_band(a):
    a = int(a)
    return "18-21" if a <= 21 else ("22-29" if a <= 29 else "30+")

def r3(v):
    return "" if v is None else round(v, 3)

# ---------------- 1. speaker metadata ----------------
meta = {}
with open(META_CSV, encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        meta[row["id"].zfill(3)] = {"age": row["age"], "gender": row["gender"].strip()}
print(f"[1] metadata: {len(meta)} speakers")

# ---------------- 2. scan + clean the digit folders ----------------
os.makedirs(OUT_DIR, exist_ok=True)
wav_out = os.path.join(OUT_DIR, "wav")
rows, seen_md5 = [], {}
for d in range(10):
    folder = os.path.join(SRC_DIR, str(d))
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(".wav"):
            continue
        src = os.path.join(folder, f)
        stem = f[:-4]
        r = {"orig_path": os.path.relpath(src, SRC_DIR), "orig_id": "", "speaker": "", "digit": "", "rep": "",
             "exclude": 0, "reason": []}
        mt = NAME_RE.match(stem)
        if not mt:
            r["exclude"] = 1; r["reason"].append("unparseable_name"); rows.append(r); continue
        oid, dig, rep = mt.group(1), int(mt.group(2)), int(mt.group(3))
        r.update(orig_id=oid, speaker=MERGE_IDS.get(oid, oid), digit=dig, rep=rep,
                 utt_id=f"rail_{oid}_{dig}_{rep:02d}")
        if dig != d:
            r["reason"].append("digit_folder_mismatch"); r["exclude"] = 1
        if oid in EXCLUDE_IDS:
            r["reason"].append("speaker_excluded"); r["exclude"] = 1
        if r["speaker"] not in meta:
            r["reason"].append("no_metadata"); r["exclude"] = 1
        try:
            raw = open(src, "rb").read()
            md5 = hashlib.md5(raw).hexdigest(); r["md5"] = md5
            sr, ch, x = read_wav(src)
            mono, used_ch, ident = to_mono(x)
            r.update(sr=sr, channels=ch, channel_used=used_ch, channels_identical=ident, n_samples=len(mono))
            if sr != 16000:
                r["reason"].append(f"sr_{sr}"); r["exclude"] = 1
            q = quality(mono, sr); r.update(q)
            r["flag_empty"] = int(q["speech_s"] < MIN_SPEECH_S or q["peak_dbfs"] < MIN_PEAK_DB)
            r["flag_truncated"] = int(not r["flag_empty"] and (q["touches_start"] or q["touches_end"]))
            r["flag_clipped"] = int(q["clip_frac"] > CLIP_FRAC)
            if md5 in seen_md5:
                r["reason"].append(f"duplicate_of:{seen_md5[md5]}"); r["exclude"] = 1
            else:
                seen_md5[md5] = r["utt_id"]
            if r["flag_empty"]:
                r["reason"].append("empty"); r["exclude"] = 1
            if r["flag_truncated"] and EXCLUDE_TRUNCATED:
                r["reason"].append("truncated"); r["exclude"] = 1
            r["clean_path"] = os.path.join("wav", r["speaker"], r["utt_id"] + ".wav")
            write_wav(os.path.join(OUT_DIR, r["clean_path"]), mono, sr)
        except Exception as ex:
            r["reason"].append("read_error:" + type(ex).__name__); r["exclude"] = 1
        rows.append(r)
    print(f"[2] digit {d}: {len(rows)} files processed", flush=True)

# ---------------- 3. cross-check the curated 'all' folder ----------------
all_dir = os.path.join(SRC_DIR, "all")
all_report = {"files": 0, "identical_in_digits": 0, "not_in_digits": [], "folder_to_ids": {}}
if os.path.isdir(all_dir):
    for dp, _, fs in os.walk(all_dir):
        fm = re.search(r"\((\d+)\)", os.path.basename(dp))
        for f in fs:
            if not f.lower().endswith(".wav"):
                continue
            all_report["files"] += 1
            h = hashlib.md5(open(os.path.join(dp, f), "rb").read()).hexdigest()
            if h in seen_md5:
                all_report["identical_in_digits"] += 1
                if fm:
                    sid = seen_md5[h].split("_")[1]
                    all_report["folder_to_ids"].setdefault(fm.group(1), collections.Counter())[sid] += 1
            else:
                all_report["not_in_digits"].append(os.path.relpath(os.path.join(dp, f), SRC_DIR))
print(f"[3] 'all' folder: {all_report['files']} files, {all_report['identical_in_digits']} identical to digit-folder files")

# ---------------- 4. speaker-disjoint split ----------------
kept = [r for r in rows if not r["exclude"]]
spk_ids = sorted({r["speaker"] for r in kept})
group_of = {s: s for s in spk_ids}
for g in SAME_SPLIT_GROUPS:
    present = [s for s in g if s in group_of]
    for s in present:
        group_of[s] = present[0]
units = collections.defaultdict(list)
for s in spk_ids:
    units[group_of[s]].append(s)
rng = random.Random(SEED)
unit_list = list(units)
rng.shuffle(unit_list)
unit_list.sort(key=lambda u: (meta[u]["gender"], age_band(meta[u]["age"])))   # stable: random within stratum
offset = rng.randrange(len(SPLIT_PATTERN))
split_of = {}
for i, u in enumerate(unit_list):
    for s in units[u]:
        split_of[s] = SPLIT_PATTERN[(i + offset) % len(SPLIT_PATTERN)]
for r in rows:
    if r["exclude"]:
        r["split"], r["role"] = "", ""
    else:
        r["split"] = split_of[r["speaker"]]
        r["role"] = "train" if r["split"] == "adapt" else ("enroll" if r["rep"] in ENROLL_REPS else "trial")
    r["gender"] = meta.get(r.get("speaker", ""), {}).get("gender", "")
    r["age"] = meta.get(r.get("speaker", ""), {}).get("age", "")
print("[4] split done")

# ---------------- 5. ASV trial lists (dev, test) ----------------
for sp in ["dev", "test"]:
    spk = sorted({r["speaker"] for r in kept if r["split"] == sp})
    by_g = collections.defaultdict(list)
    for s in spk:
        by_g[meta[s]["gender"]].append(s)
    trials = []
    for r in kept:
        if r["split"] != sp or r["role"] != "trial":
            continue
        trials.append((r["speaker"], r["utt_id"], "target"))
        pool = [s for s in by_g[meta[r["speaker"]]["gender"]] if s != r["speaker"]]
        for s in rng.sample(pool, min(NONTARGET_PER_TRIAL, len(pool))):
            trials.append((s, r["utt_id"], "nontarget"))
    with open(os.path.join(OUT_DIR, f"trials_{sp}.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["enroll_speaker", "test_utt_id", "label"]); w.writerows(trials)
print("[5] trial lists written")

# ---------------- 6. write manifests ----------------
cols = ["utt_id", "clean_path", "orig_path", "orig_id", "speaker", "gender", "age", "digit", "rep",
        "split", "role", "exclude", "reason", "flag_empty", "flag_truncated", "flag_clipped",
        "speech_s", "onset_s", "offset_s", "peak_dbfs", "rms_dbfs", "noise_floor_db", "clip_frac", "hum_share_db",
        "sr", "channels", "channel_used", "channels_identical", "n_samples", "md5"]
def fmt(r, c):
    v = r.get(c, "")
    if c == "reason": return ";".join(v)
    if isinstance(v, float): return round(v, 4)
    return "" if v is None else v
with open(os.path.join(OUT_DIR, "manifest_all.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(cols)
    for r in rows: w.writerow([fmt(r, c) for c in cols])
with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(cols)
    for r in kept: w.writerow([fmt(r, c) for c in cols])
with open(os.path.join(OUT_DIR, "exclusions.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["orig_path", "utt_id", "reason"])
    for r in rows:
        if r["exclude"]: w.writerow([r["orig_path"], r.get("utt_id", ""), ";".join(r["reason"])])
spk_rows = []
for s in spk_ids:
    rs = [r for r in kept if r["speaker"] == s]
    spk_rows.append([s, "+".join(sorted({r["orig_id"] for r in rs})), meta[s]["gender"], meta[s]["age"],
                     age_band(meta[s]["age"]), split_of[s], len(rs), sum(r["flag_truncated"] for r in rs)])
with open(os.path.join(OUT_DIR, "speakers.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["speaker", "orig_ids", "gender", "age", "age_band", "split", "n_utts", "n_truncated"])
    w.writerows(spk_rows)

# ---------------- 7. review samples ----------------
rev = os.path.join(OUT_DIR, "review")
os.makedirs(rev, exist_ok=True)
flagged = [r for r in rows if r.get("clean_path") and (r.get("flag_empty") or r.get("flag_truncated") or r.get("flag_clipped"))]
for r in rng.sample(flagged, min(N_REVIEW, len(flagged))):
    tag = "EMPTY" if r["flag_empty"] else ("CLIP" if r["flag_clipped"] else "TRUNC")
    shutil.copy(os.path.join(OUT_DIR, r["clean_path"]), os.path.join(rev, f"{tag}_{r['utt_id']}.wav"))

# ---------------- 8. summary ----------------
L = [f"KurdFake RAIL preparation  {datetime.datetime.now():%Y-%m-%d %H:%M}  seed={SEED}", ""]
L.append(f"Original files in digit folders : {len(rows)}")
L.append(f"Kept                             : {len(kept)}")
L.append(f"Excluded                         : {len(rows) - len(kept)}")
rc = collections.Counter(x.split(':')[0] for r in rows for x in r["reason"])
for k, v in rc.most_common():
    L.append(f"    {k}: {v}")
L.append(f"Flags among kept: truncated={sum(r['flag_truncated'] for r in kept)}  clipped={sum(r['flag_clipped'] for r in kept)}")
L.append(f"Stereo files with identical channels: {sum(r.get('channels_identical', 0) for r in rows if r.get('channels') == 2)} / {sum(1 for r in rows if r.get('channels') == 2)}")
L.append(f"Channel used: {dict(collections.Counter(r.get('channel_used') for r in rows if 'channel_used' in r))}  (0 = left)")
hs = sorted(r["hum_share_db"] for r in kept)
if hs:
    L.append(f"Share of energy below {VAD_HP_HZ} Hz (dB): median={hs[len(hs)//2]:.1f}  (0 dB = all energy is hum)")
L.append(f"Speech touching clip end: {sum(r['touches_end'] for r in kept)}   touching start: {sum(r['touches_start'] for r in kept)}")
cf = [r["clip_frac"] for r in kept]
L.append(f"Clipped (>0.1% full-scale samples): {sum(c > CLIP_FRAC for c in cf)}   any full-scale sample: {sum(c > 0 for c in cf)}")
sp_ = sorted(r["speech_s"] for r in kept)
if sp_:
    L.append(f"Speech per clip (s): p5={sp_[int(.05*len(sp_))]:.2f} median={sp_[len(sp_)//2]:.2f} p95={sp_[int(.95*len(sp_))]:.2f}")
L.append("")
L.append(f"Speakers kept: {len(spk_ids)}  (merged: {MERGE_IDS}; excluded: {sorted(EXCLUDE_IDS)})")
L.append(f"{'split':6s} {'spk':>4s} {'male':>5s} {'female':>6s} {'utts':>6s} {'enroll':>6s} {'trial':>6s}  age bands")
for sp in ["adapt", "dev", "test"]:
    ss = [x for x in spk_rows if x[5] == sp]
    us = [r for r in kept if r["split"] == sp]
    ab = collections.Counter(x[4] for x in ss)
    L.append(f"{sp:6s} {len(ss):4d} {sum(x[2]=='Male' for x in ss):5d} {sum(x[2]=='Female' for x in ss):6d} "
             f"{len(us):6d} {sum(r['role']=='enroll' for r in us):6d} {sum(r['role']=='trial' for r in us):6d}  {dict(sorted(ab.items()))}")
for sp in ["dev", "test"]:
    with open(os.path.join(OUT_DIR, f"trials_{sp}.csv"), encoding="utf-8") as fh:
        lab = collections.Counter(line.strip().split(",")[-1] for line in list(fh)[1:])
    L.append(f"trials_{sp}: {dict(lab)}")
L.append("")
L.append(f"'all' folder check: {all_report['files']} files, {all_report['identical_in_digits']} byte-identical to digit-folder files, "
         f"{len(all_report['not_in_digits'])} not found in digit folders")
for p in all_report["not_in_digits"][:20]:
    L.append(f"    not in digits: {p}")
id_folders = collections.defaultdict(dict)
for k, v in all_report["folder_to_ids"].items():
    for sid, n in v.items():
        id_folders[sid][k] = n
multi = {sid: f for sid, f in id_folders.items() if len(f) > 1}
mixed = {k: dict(v) for k, v in all_report["folder_to_ids"].items() if len(v) > 1}
L.append(f"'all' folders: {len(all_report['folder_to_ids'])}; speaker ids present: {len(id_folders)}")
L.append(f"    ids found in more than one 'all' folder: {multi}")
L.append(f"    'all' folders mixing several ids: {mixed}")
L.append(f"    digit-folder ids absent from 'all': {sorted({r['orig_id'] for r in rows if r['orig_id']} - set(id_folders))}")
summary = "\n".join(L)
open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8").write(summary)
json.dump({"seed": SEED, "exclude_ids": EXCLUDE_IDS, "merge_ids": MERGE_IDS, "same_split_groups": SAME_SPLIT_GROUPS,
           "split_pattern": SPLIT_PATTERN, "enroll_reps": sorted(ENROLL_REPS),
           "thresholds": {"vad_hp_hz": VAD_HP_HZ, "vad_margin_db": VAD_MARGIN_DB, "vad_abs_db": VAD_ABS_DB, "min_speech_s": MIN_SPEECH_S, "min_peak_db": MIN_PEAK_DB, "edge_s": EDGE_S, "clip_frac": CLIP_FRAC},
           "exclude_truncated": EXCLUDE_TRUNCATED}, open(os.path.join(OUT_DIR, "config_used.json"), "w"), indent=2)

# ---------------- 9. plots (optional) ----------------
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].hist([r["speech_s"] for r in rows if "speech_s" in r], bins=40, color="#4C72B0")
    ax[0].axvline(MIN_SPEECH_S, color="r", ls="--"); ax[0].set_title("Speech duration per clip (s)")
    ax[1].hist([r["onset_s"] for r in kept if r.get("onset_s") is not None], bins=40, alpha=.7, label="onset")
    ax[1].hist([r["offset_s"] for r in kept if r.get("offset_s") is not None], bins=40, alpha=.7, label="offset")
    ax[1].legend(); ax[1].set_title("Speech onset / offset (s)")
    c = collections.Counter(r["speaker"] for r in kept)
    ax[2].bar(range(len(c)), [c[s] for s in sorted(c)], color="#55A868"); ax[2].set_title("Kept clips per speaker")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "summary_plots.png"), dpi=130)
    print("[9] plots saved")
except Exception as ex:
    print("[9] plots skipped:", ex)

print("\n" + summary)
print(f"\nDone. Output folder: {OUT_DIR}")
