# KurdFake - listen_channel.py
# Copies a small listening set for the channel-matched condition into results\listen_channel\
# For 3 RAIL test spoofs (systems D, C1D, BC) of 3 different speakers it copies four files side by side:
#   1_REAL_trial      a real test recording of the same speaker, same digit when available
#   2_FAKE_original   the spoof as generated (clean, -16 LUFS)
#   3_FAKE_channel    the same spoof with the real channel transplanted (what the detector now sees)
#   4_REAL_channelref the real enrollment recording whose background/level/clipping was used
# Run AFTER make_channel_matched.py:   python listen_channel.py
import os, csv, shutil, random
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
ROOT = KF_ROOT; DATA = os.path.join(ROOT, "Dataset"); RAIL = os.path.join(DATA, "RAIL_clean")
OUT = os.path.join(ROOT, "results", "listen_channel"); os.makedirs(OUT, exist_ok=True)
rail = {}
with open(os.path.join(RAIL, "manifest.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["split"]:
            rail[r["utt_id"]] = r
orig = {}
for f in ["spoof_manifest_knnvc_final.csv", "spoof_manifest_tts_vekol.csv", "spoof_manifest_f5.csv"]:
    with open(os.path.join(DATA, "Spoofs", f), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            orig[r["utt_id"]] = r
with open(os.path.join(DATA, "Spoofs_CM", "spoof_manifest_channel_matched.csv"), encoding="utf-8") as fh:
    cm = [r for r in csv.DictReader(fh) if r["dataset"] == "rail" and r["split"] == "test"]
rng = random.Random(7); used = set()
for k, system in enumerate(["D+ch", "C1D+ch", "BC+ch"], 1):
    cand = [r for r in cm if r["system"] == system and r["target_speaker"] not in used and r["source_spoof"] in orig]
    r = rng.choice(cand); used.add(r["target_speaker"]); o = orig[r["source_spoof"]]
    spk, digit = r["target_speaker"], o.get("digit", "")
    trials = [t for t in rail.values() if t["speaker"] == spk and t["split"] == "test" and t["role"] == "trial"]
    same = [t for t in trials if t.get("digit") == digit] or trials
    real = rng.choice(same)
    tag = f"{k}_{system.replace('+ch', '')}_spk{spk}"
    pairs = [(os.path.join(RAIL, real["clean_path"]), "1_REAL_trial"),
             (os.path.join(DATA, "Spoofs", o["path"]), "2_FAKE_original"),
             (os.path.join(DATA, "Spoofs_CM", r["path"]), "3_FAKE_channel"),
             (os.path.join(RAIL, rail[r["channel_ref"]]["clean_path"]), "4_REAL_channelref")]
    for src, name in pairs:
        shutil.copy(src, os.path.join(OUT, f"{tag}_{name}.wav"))
    print(f"set {k}: system {system[:-3]}  speaker {spk}  digit {digit or '?'}")
print(f"Copied 12 files to {OUT}")
