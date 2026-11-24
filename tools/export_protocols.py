"""Export the public KurdFake protocols (manifests and trial lists) into ./protocols.

Run once on the machine that holds Dataset/:
    set KURDFAKE_ROOT=F:\\hiwa\\kurdfake
    python tools\\export_protocols.py

What it does
  * copies the final manifests and trial lists of RAIL, Common Voice and all spoof systems
  * removes columns that are not needed to reproduce the experiments (age, original file paths)
  * adds an md5 column to every spoof manifest (hash of the audio file), so that anyone who
    receives the audio under the data use agreement can verify it
  * writes protocols/SHA256SUMS.txt for the exported CSV files

No audio and no speaker names are written. Speaker identifiers are the pseudonymous numbers
already used in the manifests (RAIL) or the public Common Voice client identifiers (CV).
"""
import csv, hashlib, os, sys

KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())
DATA = os.path.join(KF_ROOT, "Dataset")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "protocols")

DROP = {"age", "orig_path", "orig_id"}           # not needed for reproduction
# final spoof manifests used in the paper (D+G, D2, C1+C1D, B+BC); earlier drafts are not exported
FINAL_SPOOF_MANIFESTS = ["spoof_manifest_knnvc_final.csv", "spoof_manifest_knnvc_D2.csv",
                         "spoof_manifest_tts_vekol.csv", "spoof_manifest_f5.csv"]
csv.field_size_limit(10_000_000)


def md5(path, block=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def export(src, dst_name, add_md5=False):
    if not os.path.isfile(src):
        print(f"  MISSING  {src}")
        return None
    with open(src, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"  EMPTY    {src}")
        return None
    cols = [c for c in rows[0].keys() if c not in DROP]
    missing = 0
    if add_md5 and "path" in cols:
        if "md5" not in cols:
            cols.append("md5")
        base = os.path.dirname(src)
        for i, r in enumerate(rows, 1):
            if r.get("md5"):
                continue
            p = os.path.join(base, r["path"].replace("\\", os.sep))
            if os.path.isfile(p):
                r["md5"] = md5(p)
            else:
                r["md5"] = ""
                missing += 1
            if i % 5000 == 0:
                print(f"    {os.path.basename(src)}: {i}/{len(rows)} hashed", flush=True)
    dst = os.path.join(OUT, dst_name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    note = f"   ({missing} audio files not found, md5 left empty)" if missing else ""
    print(f"  OK       {dst_name:55s} {len(rows):7d} rows{note}")
    return dst


def main():
    if not os.path.isdir(DATA):
        sys.exit(f"Dataset folder not found: {DATA}\nSet KURDFAKE_ROOT to the folder that contains Dataset/.")
    os.makedirs(OUT, exist_ok=True)
    written = []
    print("Corpora")
    for corpus, folder in [("rail", "RAIL_clean"), ("cv", "CV_ckb_clean")]:
        for name in ["manifest.csv", "trials_dev.csv", "trials_test.csv"]:
            written.append(export(os.path.join(DATA, folder, name), f"{corpus}/{name}"))
    print("Spoofs (raw)")
    for name in FINAL_SPOOF_MANIFESTS:
        written.append(export(os.path.join(DATA, "Spoofs", name), f"spoofs/{name}", add_md5=True))
    print("Spoofs (channel-matched, +ch)")
    written.append(export(os.path.join(DATA, "Spoofs_CM", "spoof_manifest_channel_matched.csv"),
                          "spoofs/spoof_manifest_channel_matched.csv", add_md5=True))
    written = [w for w in written if w]
    n_raw = sum(sum(1 for _ in open(os.path.join(OUT, "spoofs", n), encoding="utf-8")) - 1
                for n in FINAL_SPOOF_MANIFESTS if os.path.isfile(os.path.join(OUT, "spoofs", n)))
    cm = os.path.join(OUT, "spoofs", "spoof_manifest_channel_matched.csv")
    if os.path.isfile(cm):
        n_cm = sum(1 for _ in open(cm, encoding="utf-8")) - 1
        print(f"\nCheck: raw spoofs {n_raw} rows, channel-matched {n_cm} rows -> "
              + ("match" if n_raw == n_cm else "DO NOT MATCH"))
    with open(os.path.join(OUT, "SHA256SUMS.txt"), "w", encoding="utf-8") as fh:
        for p in written:
            fh.write(f"{sha256(p)}  {os.path.relpath(p, OUT).replace(os.sep, '/')}\n")
    print(f"\n{len(written)} files written to {OUT}")
    print("Check the list above: every expected file should say OK. Then upload the protocols folder.")


if __name__ == "__main__":
    main()
