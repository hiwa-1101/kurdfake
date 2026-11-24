# KurdFake - Step 1b: RAIL quality check (read-only; never modifies or deletes audio)
# Needs: numpy   ->   py -m pip install numpy
# Usage:  py p1b_quality.py "<KURDFAKE_ROOT>\Dataset\RAIL_full"
# Checks the digit folders (0..9), the per-speaker "all" folder and "fulti".
# Output (next to this script): rail_quality.csv, rail_quality_summary.txt

import os, sys, re, csv, wave, hashlib, collections
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(KF_ROOT, "Dataset", "RAIL_full")
out_dir = os.path.dirname(os.path.abspath(__file__))
FRAME, HOP = 400, 160          # 25 ms / 10 ms at 16 kHz
EDGE_S = 0.03                  # speech within 30 ms of an edge -> "touches edge"

def analyse(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    raw_bytes = open(path, "rb").read()
    md5 = hashlib.md5(raw_bytes).hexdigest()
    x = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch).astype(np.float32)
    r = {"sr": sr, "channels": ch, "md5": md5}
    if ch == 2:
        a, b = x[:, 0], x[:, 1]
        r["ch_identical"] = int(np.array_equal(a, b))
        r["ch_rms_ratio_db"] = round(20*np.log10((np.sqrt(np.mean(a**2))+1e-9)/(np.sqrt(np.mean(b**2))+1e-9)), 2)
        r["ch_corr"] = round(float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0, 4)
    m = x.mean(axis=1) / 32768.0
    r["clip_frac"] = round(float(np.mean(np.abs(x) >= 32767)), 5)
    r["peak_dbfs"] = round(20*np.log10(np.max(np.abs(m)) + 1e-9), 2)
    r["rms_dbfs"] = round(20*np.log10(np.sqrt(np.mean(m**2)) + 1e-9), 2)
    r["dc_offset"] = round(float(m.mean()), 5)
    if len(m) < FRAME:
        r.update(speech_s=0, onset_s="", offset_s="", noise_floor_db="", touches_start=0, touches_end=0)
        return r
    nf = 1 + (len(m) - FRAME) // HOP
    fr = np.stack([m[i*HOP:i*HOP+FRAME] for i in range(nf)])
    e = 10*np.log10(np.mean(fr**2, axis=1) + 1e-10)
    floor = np.percentile(e, 10)
    thr = max(floor + 12.0, -50.0)
    sp = e > thr
    # remove isolated single frames (clicks)
    sp = sp & (np.convolve(sp.astype(int), [1, 1, 1], "same") >= 2)
    r["noise_floor_db"] = round(float(floor), 2)
    r["speech_s"] = round(float(sp.sum()) * HOP / sr, 3)
    if sp.any():
        idx = np.where(sp)[0]
        on, off = idx[0]*HOP/sr, (idx[-1]*HOP + FRAME)/sr
        r["onset_s"], r["offset_s"] = round(on, 3), round(off, 3)
        r["touches_start"] = int(on <= EDGE_S)
        r["touches_end"] = int(off >= len(m)/sr - EDGE_S)
    else:
        r.update(onset_s="", offset_s="", touches_start=0, touches_end=0)
    return r

pat = re.compile(r"^(\d{3})(\d)(\d{1,2})$")
rows = []
targets = [str(d) for d in range(10)] + ["all", "fulti"]
for t in targets:
    base = os.path.join(root, t)
    if not os.path.isdir(base):
        continue
    for dp, _, fs in os.walk(base):
        for f in sorted(fs):
            if not f.lower().endswith(".wav"):
                continue
            p = os.path.join(dp, f)
            stem = f[:-4]
            mt = pat.match(stem)
            view = "digits" if t.isdigit() else t
            fm = re.search(r"\((\d+)\)", os.path.relpath(dp, base)) if view == "all" else None
            r = {"view": view, "all_folder": fm.group(1) if fm else "",
                 "rel_path": os.path.relpath(p, root), "stem": stem,
                 "speaker": mt.group(1) if mt else "", "digit": mt.group(2) if mt else "",
                 "rep": mt.group(3) if mt else "", "folder_digit": t if view == "digits" else "",
                 "parse_ok": int(bool(mt))}
            try:
                r.update(analyse(p)); r["error"] = ""
            except Exception as ex:
                r["error"] = type(ex).__name__ + ": " + str(ex)[:80]
            # flags (heuristic; to be confirmed by listening + Silero-VAD later)
            sp = r.get("speech_s", 0) or 0
            r["flag_empty"] = int(r["error"] == "" and (sp < 0.2 or r.get("peak_dbfs", 0) < -40))
            r["flag_truncated"] = int(not r["flag_empty"] and (r.get("touches_start", 0) or r.get("touches_end", 0)))
            r["flag_clipped"] = int((r.get("clip_frac", 0) or 0) > 0.001)
            rows.append(r)
    print(f"done: {t}  ({len(rows)} files so far)", flush=True)

keys = []
for r in rows:
    for k in r:
        if k not in keys:
            keys.append(k)
qpath = os.path.join(out_dir, "rail_quality.csv")
with open(qpath, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

L = []
for v in ["digits", "all", "fulti"]:
    g = [r for r in rows if r["view"] == v and not r["error"]]
    if not g:
        continue
    L.append(f"== {v}: {len(g)} files ==")
    L.append(f"  empty: {sum(r['flag_empty'] for r in g)}   truncated: {sum(r['flag_truncated'] for r in g)}   clipped: {sum(r['flag_clipped'] for r in g)}")
    if g[0].get("channels") == 2:
        L.append(f"  stereo channels identical: {sum(r.get('ch_identical',0) for r in g)} / {len(g)}")
        cr = sorted(r["ch_corr"] for r in g); L.append(f"  channel corr median: {cr[len(cr)//2]}")
    sp = sorted(r["speech_s"] for r in g)
    L.append(f"  speech_s  p5={sp[int(.05*len(sp))]} median={sp[len(sp)//2]} p95={sp[int(.95*len(sp))]}")
    pk = sorted(r["peak_dbfs"] for r in g)
    L.append(f"  peak_dbfs p5={pk[int(.05*len(pk))]} median={pk[len(pk)//2]}")
    L.append(f"  parse errors: {sum(1 for r in g if not r['parse_ok'])}")
md5 = collections.defaultdict(list)
for r in rows:
    if r.get("md5"): md5[r["md5"]].append(r["rel_path"])
view_of = {r["rel_path"]: r["view"] for r in rows}
dups = [v for v in md5.values() if len(v) > 1]
L.append(f"\nGroups of byte-identical files: {len(dups)}")
combo = collections.Counter(" + ".join(sorted(view_of[p] for p in v)) for v in dups)
for k, c in combo.most_common(12):
    L.append(f"  {c:6d} groups: {k}")
dig_md5 = {r["md5"] for r in rows if r["view"] == "digits" and r.get("md5")}
for v in ["all", "fulti"]:
    g = [r for r in rows if r["view"] == v and r.get("md5")]
    if g:
        L.append(f"  {v}: {sum(r['md5'] in dig_md5 for r in g)} / {len(g)} files have an identical copy in the digit folders")
within = [v for v in dups if sum(view_of[p] == "all" for p in v) > 1]
L.append(f"  duplicates inside 'all' itself (possible padding to 100): {len(within)} groups")
for v in within[:10]:
    L.append("    " + " = ".join(p for p in v if view_of[p] == "all"))
L.append(f"\nWAV errors: {sum(1 for r in rows if r['error'])}")
spath = os.path.join(out_dir, "rail_quality_summary.txt")
open(spath, "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L)); print(f"\nSaved: {qpath}\nSaved: {spath}")
