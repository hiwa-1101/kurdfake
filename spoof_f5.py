# =====================================================================
# KurdFake - spoof_f5.py   (Prompt 2, held-out TTS family: systems B and BC)
#   Model: Central Kurdish F5-TTS (aranemini/central-kurdish-tts, studio-male, CC BY-NC-ND 4.0)
#          flow-matching DiT + Vocos vocoder -> a family NOT seen in adaptation (no WavLM, no HiFi-GAN)
#   B  = F5-TTS with the authors' own studio prompt (fixed voice), same content as the bona fide data
#   BC = F5-TTS prompted with OTHER recordings of the TARGET speaker (zero-shot cloning attempt);
#        prompt clips are disjoint from the ASV enrollment clips (same rule as D2)
#   Generated for dev and test speakers only (held out from any training / adaptation).
#
# Anaconda Prompt:
#   conda activate gpu_env
#   cd /d <KURDFAKE_ROOT>
#   python spoof_f5.py --smoke     (2 test speakers per dataset)
#   python spoof_f5.py             (all dev/test speakers; resumes if interrupted)
# Writes: Dataset\Spoofs\B\..., Dataset\Spoofs\BC\..., Dataset\Spoofs\spoof_manifest_f5.csv
# =====================================================================
import os, sys, csv, wave, random, argparse, time, collections, zlib, tempfile
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np
import torch
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kf_f5

DATA = os.path.join(KF_ROOT, "Dataset")
SETS = {"rail": os.path.join(DATA, "RAIL_clean"), "cv": os.path.join(DATA, "CV_ckb_clean")}
OUT  = os.path.join(DATA, "Spoofs")
F5_DIR, VOCOS_DIR = kf_f5.F5_DIR, kf_f5.VOCOS_DIR
SEED, SR = 1337, 16000
N_PER_SPK = {"rail": 20, "cv": 10}            # per target speaker and system (2 per digit on RAIL)
RAIL_PROMPT_REP, CV_PROMPT_MIN_S = 4, 6.0     # RAIL: rep 4 of digits 0-9 (never enrollment); CV: >= 6 s of non-enrollment clips
DIGIT_WORDS = ["سفر", "یەک", "دوو", "سێ", "چوار", "پێنج", "شەش", "حەوت", "هەشت", "نۆ"]

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true"); ap.add_argument("--datasets", default="rail,cv")
args = ap.parse_args()
rng = random.Random(SEED)

tts = kf_f5.F5Kurdish(F5_DIR, VOCOS_DIR)
studio_wav = os.path.join(F5_DIR, "prompt-studio-male.wav")
studio_txt = open(os.path.join(F5_DIR, "prompt-studio-male.txt"), encoding="utf-8").read().strip()
tmpdir = tempfile.mkdtemp(prefix="kf_f5_")

def load(base, rel):
    with wave.open(os.path.join(base, rel), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return x.mean(axis=1).astype(np.float32) / 32768.0
def write_wav(path, y, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
def loudnorm(y, tgt=-16.0):
    try:
        import torchaudio
        t = torch.from_numpy(y).float(); return torchaudio.functional.gain(t, tgt - float(torchaudio.functional.loudness(t[None], SR))).numpy()
    except Exception:
        return y * (10 ** (tgt / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-9))
def to16k(y, sr):
    return resample_poly(y, 2, 3).astype(np.float32) if sr == 24000 else y
def seed_of(uid):
    return zlib.crc32(uid.encode("utf-8")) % (2 ** 31)

def rail_prompt(rs, base, spk):
    """Join rep-4 clips of digits 0..9 of the target (speech region only) into one ~4-6 s prompt."""
    clips, words = [], []
    for d in range(10):
        c = [r for r in rs if r["digit"] == str(d) and int(r["rep"]) == RAIL_PROMPT_REP]
        if not c:
            continue
        r = c[0]; x = load(base, r["clean_path"])
        try:
            x = x[max(0, int((float(r["onset_s"]) - 0.02) * SR)): int((float(r["offset_s"]) + 0.02) * SR)]
        except ValueError:
            pass
        clips += [x, np.zeros(int(0.06 * SR), np.float32)]; words.append(DIGIT_WORDS[d])   # short gaps: F5 infers speaking rate from the prompt
    if len(words) < 5:
        return None, None, []
    p = os.path.join(tmpdir, f"prompt_rail_{spk}.wav"); write_wav(p, np.concatenate(clips), SR)
    return p, " ".join(words), [r["utt_id"] for r in rs if int(r["rep"]) == RAIL_PROMPT_REP]

man = os.path.join(OUT, "spoof_manifest_f5.csv"); os.makedirs(OUT, exist_ok=True)
done = set()
if os.path.exists(man):
    with open(man, encoding="utf-8") as fh:
        done = {r["utt_id"] for r in csv.DictReader(fh)}
cols = ["utt_id", "path", "dataset", "system", "split", "target_speaker", "target_gender", "source_utt_id",
        "source_speaker", "digit", "sentence", "text", "prompt_utt_ids", "prompt_text", "dur_s", "label"]
mf = open(man, "a", newline="", encoding="utf-8"); mw = csv.DictWriter(mf, fieldnames=cols)
if not done:
    mw.writeheader()

t0, made = time.time(), 0
for ds in args.datasets.split(","):
    base = SETS[ds]
    with open(os.path.join(base, "manifest.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["speaker"]].append(r)
    spks = [s for s in sorted(by) if by[s][0]["split"] in ("dev", "test")]
    if args.smoke:
        spks = [s for s in spks if by[s][0]["split"] == "test"][:2]
    print(f"\n== {ds}: {len(spks)} dev/test target speakers ==")
    for k, s in enumerate(spks):
        rs = by[s]; split, gender = rs[0]["split"], rs[0]["gender"]
        if ds == "rail":
            p_wav, p_txt, p_ids = rail_prompt(rs, base, s)
            jobs = [(d, None) for d in range(10) for _ in range(N_PER_SPK["rail"] // 10)]
        else:
            trial = sorted([r for r in rs if r["role"] == "trial"], key=lambda r: r["utt_id"])
            # prompt = 1-3 non-enrollment clips of the target joined to >= 6 s (F5 clones better from a longer prompt)
            pr, tot = [], 0.0
            for r in trial:
                if tot >= CV_PROMPT_MIN_S or len(pr) >= 3:
                    break
                if tot + float(r.get("dur_s") or 0) <= 11.0:
                    pr.append(r); tot += float(r.get("dur_s") or 0)
            if pr:
                parts = []
                for r in pr:
                    parts += [load(base, r["clean_path"]), np.zeros(int(0.2 * SR), np.float32)]
                p_wav = os.path.join(tmpdir, f"prompt_cv_{s}.wav"); write_wav(p_wav, np.concatenate(parts), SR)
                p_txt = " ".join(r["sentence"].strip() for r in pr)
            else:
                p_wav = p_txt = None
            p_ids = [r["utt_id"] for r in pr]
            texts = [r for r in trial if r not in pr]; rng.shuffle(texts)
            jobs = [(None, r) for r in texts[:N_PER_SPK["cv"]]]
        for j, (d, src) in enumerate(jobs):
            text = DIGIT_WORDS[d] if ds == "rail" else src["sentence"]
            tag = f"d{d}_{j:02d}" if ds == "rail" else src["utt_id"]
            common = {"dataset": ds, "split": split, "target_speaker": s, "target_gender": gender,
                      "digit": "" if d is None else d, "sentence": "" if src is None else src["sentence"],
                      "text": text, "label": "spoof", "source_speaker": "f5_studio_male",
                      "source_utt_id": "" if src is None else src["utt_id"]}
            for system, pw, pt, pid in [("B", studio_wav, studio_txt, "authors_studio_prompt"),
                                        ("BC", p_wav, p_txt, ";".join(p_ids))]:
                uid = f"spf_{system}_{ds}_{s}_{tag}"
                if uid in done or pw is None:
                    continue
                y = tts.synth(text, pw, pt, seed=seed_of(uid))
                y = loudnorm(to16k(y, tts.sr))
                rel = os.path.join(system, ds, s, uid + ".wav")
                write_wav(os.path.join(OUT, rel), y, SR)
                mw.writerow({**common, "utt_id": uid, "path": rel, "system": system, "prompt_utt_ids": pid,
                             "prompt_text": pt, "dur_s": round(len(y) / SR, 3)})
                made += 1
        mf.flush()
        print(f"  [{ds}] {k+1}/{len(spks)} {s} ({split})  files so far: {made}  {(time.time()-t0)/60:.1f} min", flush=True)
mf.close()
print(f"\nDone. {made} new files.  Manifest: {man}")
