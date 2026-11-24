# KurdFake - Step 1a: RAIL inventory (standard library only, read-only)
# Usage (Windows):  python p1_inventory.py "<KURDFAKE_ROOT>\Dataset\RAIL_full"
# Output (next to this script): rail_inventory.csv, rail_summary.txt
# The script never modifies, moves or deletes any audio file.

import os, sys, csv, wave, collections
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(KF_ROOT, "Dataset", "RAIL_full")
out_dir = os.path.dirname(os.path.abspath(__file__))
inv_path = os.path.join(out_dir, "rail_inventory.csv")
sum_path = os.path.join(out_dir, "rail_summary.txt")

rows = []
ext_count = collections.Counter()
ext_bytes = collections.Counter()

for dirpath, _, files in os.walk(root):
    for f in files:
        p = os.path.join(dirpath, f)
        rel = os.path.relpath(p, root)
        ext = os.path.splitext(f)[1].lower()
        size = os.path.getsize(p)
        ext_count[ext] += 1
        ext_bytes[ext] += size
        r = {"rel_path": rel, "folder": os.path.dirname(rel), "name": f, "ext": ext,
             "bytes": size, "sr": "", "channels": "", "bits": "", "frames": "",
             "duration_s": "", "error": ""}
        if ext == ".wav":
            try:
                with wave.open(p, "rb") as w:
                    r["sr"] = w.getframerate()
                    r["channels"] = w.getnchannels()
                    r["bits"] = w.getsampwidth() * 8
                    r["frames"] = w.getnframes()
                    r["duration_s"] = round(w.getnframes() / w.getframerate(), 4)
            except Exception as e:
                r["error"] = type(e).__name__ + ": " + str(e)[:80]
        rows.append(r)

with open(inv_path, "w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["rel_path"])
    wr.writeheader()
    wr.writerows(rows)

wavs = [r for r in rows if r["ext"] == ".wav" and not r["error"]]
dur = sorted(r["duration_s"] for r in wavs)
lines = []
lines.append(f"Root: {root}")
lines.append(f"Total files: {len(rows)}   Total size: {sum(r['bytes'] for r in rows)/1e6:.1f} MB")
lines.append("\nFiles by extension (count, MB):")
for e, c in ext_count.most_common():
    lines.append(f"  {e or '(none)'}: {c}  {ext_bytes[e]/1e6:.1f} MB")
lines.append(f"\nReadable WAV: {len(wavs)}   WAV read errors: {sum(1 for r in rows if r['error'])}")
for key in ["sr", "channels", "bits"]:
    lines.append(f"  {key}: {dict(collections.Counter(r[key] for r in wavs))}")
if dur:
    q = lambda x: dur[min(len(dur) - 1, int(x * len(dur)))]
    lines.append(f"  duration_s: min={dur[0]} p5={q(.05)} median={q(.5)} p95={q(.95)} max={dur[-1]}")
    lines.append(f"  shorter than 0.5 s: {sum(1 for d in dur if d < 0.5)}   longer than 2 s: {sum(1 for d in dur if d > 2)}")
folders = collections.Counter(r["folder"] for r in rows)
lines.append(f"\nFolders containing files: {len(folders)}")
lines.append("First 15 folders (name: file count):")
for k in sorted(folders)[:15]:
    lines.append(f"  {k}: {folders[k]}")
lines.append("\nExample file names (first 30):")
for r in rows[:30]:
    lines.append("  " + r["rel_path"])

with open(sum_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("\n".join(lines))
print(f"\nSaved: {inv_path}\nSaved: {sum_path}")
