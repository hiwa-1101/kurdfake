# KurdFake - copy a few real/fake pairs into Dataset\listen_pairs for listening
# Usage (Anaconda Prompt, gpu_env):  python make_listen_pairs.py
import csv, os, shutil, random
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README

DATA = os.path.join(KF_ROOT, "Dataset")
OUT = os.path.join(DATA, "listen_pairs")
os.makedirs(OUT, exist_ok=True)
rng = random.Random(7)

clean = {"rail": os.path.join(DATA, "RAIL_clean"), "cv": os.path.join(DATA, "CV_ckb_clean")}
paths = {}
for ds, base in clean.items():
    with open(os.path.join(base, "manifest.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            paths[r["utt_id"]] = (os.path.join(base, r["clean_path"]), r)

with open(os.path.join(DATA, "Spoofs", "spoof_manifest_knnvc.csv"), encoding="utf-8") as fh:
    spoofs = list(csv.DictReader(fh))

n = 0
for ds in ["rail", "cv"]:
    for system in ["G", "D"]:
        cand = [s for s in spoofs if s["dataset"] == ds and s["system"] == system]
        if not cand:
            continue
        s = rng.choice(cand)
        tag = f"{ds}_{system}"
        fake = os.path.join(DATA, "Spoofs", s["path"])
        src_path, src = paths[s["source_utt_id"]]
        shutil.copy(fake, os.path.join(OUT, f"{tag}_2_FAKE_{s['utt_id']}.wav"))
        if system == "G":
            # G: the fake is a resynthesis of exactly this real file
            shutil.copy(src_path, os.path.join(OUT, f"{tag}_1_REAL_source_{s['source_utt_id']}.wav"))
        else:
            # D: content comes from another speaker; voice should match the target speaker
            shutil.copy(src_path, os.path.join(OUT, f"{tag}_1_REAL_source_other_speaker_{s['source_utt_id']}.wav"))
            tgt = [p for p, r in paths.values() if r["speaker"] == s["target_speaker"] and r.get("role") in ("enroll", "train")]
            if tgt:
                shutil.copy(tgt[0], os.path.join(OUT, f"{tag}_3_REAL_target_voice_{s['target_speaker']}.wav"))
        n += 1
        print(f"{tag}: fake {s['utt_id']}  <-  source {s['source_utt_id']}  (target speaker {s['target_speaker']})")
print(f"\n{n} examples copied to {OUT}")
