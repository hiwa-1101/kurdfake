# KurdFake - inspect_safetensors.py
# Reads ONLY the header of a .safetensors file (pure Python, no torch, no code execution)
# and lists the tensor names and shapes, so the loader can be written to match exactly.
#   python inspect_safetensors.py
import json, struct, os, collections
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
P = os.path.join(KF_ROOT, "models", "antideepfake_mms300m", "model.safetensors")
with open(P, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n))
meta = hdr.pop("__metadata__", None)
print(f"file size {os.path.getsize(P):,} bytes   tensors {len(hdr)}   metadata {meta}")
tot = 0
for k, v in hdr.items():
    c = 1
    for d in v["shape"]:
        c *= d
    tot += c
print(f"total parameters {tot:,}   dtypes {sorted({v['dtype'] for v in hdr.values()})}")
pre = collections.Counter(".".join(k.split(".")[:3]) for k in hdr)
print("\nname prefixes (first 3 parts):")
for p, c in sorted(pre.items()):
    print(f"  {p:60s} {c}")
print("\nall tensors not inside transformer layers 1..23:")
for k in sorted(hdr):
    parts = k.split(".")
    if any(x.isdigit() and int(x) > 0 for x in parts) and ("layers" in parts):
        continue
    print(f"  {k:75s} {hdr[k]['shape']}")
