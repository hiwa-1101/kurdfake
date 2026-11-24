# =====================================================================
# KurdFake - train_adf.py   (RQ3: E4-base and E4-proposed on MMS-300M-AntiDeepfake)
#   E4-base     : LoRA (r=8) on q/k/v/out of all 24 encoder layers + the original head (mean pooling + linear);
#                 cross-entropy only, no augmentation.
#   E4-proposed : "Speech-Focused Channel-Invariant Adaptation"
#                 = E4-base + Attentive Statistics Pooling head + blind stochastic channel augmentation + CICR:
#                   loss = CE(x) + CE(x~) + lam * (s(x) - s(x~))^2,  s = logit(real) - logit(fake), lam = 0.5 fixed.
#                 Augmentation is BLIND: synthetic 45-65 Hz mains harmonics, random soft/hard clipping, random
#                 high-pass (60-150 Hz), random low-pass (3.4-7 kHz). No real RAIL channel is used in training.
#   LoRA is only an implementation choice (GPU memory, keeping the multilingual knowledge); no claim vs full fine-tune.
#   Same data, splits, held-out systems (B, BC, D2) and report format as train_cm.py; tests raw AND +ch.
#   Model selection: fixed random subset (--dev_n) of dev, bona fide vs training systems, raw and +ch.
#   ASP is initialised so that the untrained head equals the pretrained mean-pooling head (start = zero-shot model).
#
#   python train_adf.py --variant base --smoke        quick check + speed estimate
#   python train_adf.py --variant base                E4-base
#   python train_adf.py --variant proposed            E4-proposed      (add --seed 2026 / --seed 42 later)
# Writes: results\ft\<tag>\  best_lora_head.pth, train_log.csv, scores.csv, results.txt
# Runs ONLY on the university PC (RAIL must not be uploaded to cloud services).
# =====================================================================
import os, sys, csv, json, wave, time, random, argparse, collections, platform, math
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT   = KF_ROOT
DATA   = os.path.join(ROOT, "Dataset")
AASIST = os.path.join(ROOT, "models", "aasist")
RES    = os.path.join(ROOT, "results", "ft")
SETS   = {"rail": os.path.join(DATA, "RAIL_clean"), "cv": os.path.join(DATA, "CV_ckb_clean")}
SPOOF_MANIFESTS = [os.path.join(DATA, "Spoofs", f) for f in ["spoof_manifest_knnvc_final.csv", "spoof_manifest_knnvc_D2.csv",
                                                             "spoof_manifest_tts_vekol.csv", "spoof_manifest_f5.csv"]]
TRAIN_SYSTEMS = {"D", "G", "C1", "C1D"}
CM_MANIFEST = os.path.join(DATA, "Spoofs_CM", "spoof_manifest_channel_matched.csv")   # channel-matched copies, systems 'X+ch'
def base_sys(s):
    return s[:-3] if s.endswith("+ch") else s
SR, TGT_LUFS, HP_HZ = 16000, -16.0, 100

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="proposed", choices=["base", "proposed"])
ap.add_argument("--train", default="both", choices=["both", "rail", "cv"])
ap.add_argument("--aug", default="none", choices=["none", "ch"], help="ch = also train on the real channel-matched adapt spoofs (Spoofs_CM), as E3")
ap.add_argument("--no_asp", action="store_true", help="ablation: proposed loss/augmentation but the original mean-pooling head")
ap.add_argument("--budget", type=float, default=1.0)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--steps", type=int, default=400, help="optimizer steps per epoch (each = --accum batches of --batch)")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--accum", type=int, default=3)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--lam", type=float, default=0.5, help="CICR weight, fixed a priori")
ap.add_argument("--max_s", type=float, default=4.0)
ap.add_argument("--dev_n", type=int, default=3000, help="fixed random dev subset for per-epoch model selection")
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--eval_only", action="store_true")
args = ap.parse_args()
args.cond, args.model = "raw", "ADF"
if args.smoke:
    args.epochs, args.budget, args.steps, args.dev_n = 2, 0.05, 20, 300
tag = f"ADF-MMS300M_{'E4-' + args.variant}{'-noASP' if args.no_asp else ''}{'_aug-ch' if args.aug == 'ch' else ''}_train-{args.train}_b{int(round(args.budget*100))}_s{args.seed}" + ("_smoke" if args.smoke else "")
out_dir = os.path.join(RES, tag); os.makedirs(out_dir, exist_ok=True)
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = True   # speed: cuDNN picks the fastest kernels (seeded, but not bit-exact)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"python {platform.python_version()}  torch {torch.__version__}  device {dev}\nrun: {tag}")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kf_metrics import eer, report

# ---------------- audio ----------------
def load(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    x = x.mean(axis=1).astype(np.float32) / 32768.0
    return control(x) if args.cond == "control" else x

def control(x):                                   # identical to cm_eval.py
    X = np.fft.rfft(x); X[np.fft.rfftfreq(len(x), 1 / SR) < HP_HZ] = 0
    y = np.fft.irfft(X, n=len(x)).astype(np.float32)
    try:
        import torchaudio
        t = torch.from_numpy(y); g = TGT_LUFS - float(torchaudio.functional.loudness(t[None], SR))
        y = torchaudio.functional.gain(t, g).numpy()
    except Exception:
        y = y * (10 ** (TGT_LUFS / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-9))
    return np.clip(y, -1, 1).astype(np.float32)

# ---------------- item lists ----------------
items = []   # dict: uid, path, dataset, split, system, label(1 bona / 0 spoof), speaker
for ds, base in SETS.items():
    with open(os.path.join(base, "manifest.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] in ("adapt", "dev", "test"):
                items.append(dict(uid=r["utt_id"], path=os.path.join(base, r["clean_path"]), dataset=ds, split=r["split"],
                                  system="bonafide", label=1, speaker=r["speaker"]))
for mp in SPOOF_MANIFESTS:
    with open(mp, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            items.append(dict(uid=r["utt_id"], path=os.path.join(DATA, "Spoofs", r["path"]), dataset=r["dataset"], split=r["split"],
                              system=r["system"], label=0, speaker=r["target_speaker"]))
if os.path.exists(CM_MANIFEST):
    with open(CM_MANIFEST, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            items.append(dict(uid=r["utt_id"], path=os.path.join(DATA, "Spoofs_CM", r["path"]), dataset=r["dataset"], split=r["split"],
                              system=r["system"], label=0, speaker=r["target_speaker"]))
else:
    print("WARNING: channel-matched manifest not found - only raw spoofs will be evaluated")

# data budget: a seeded subset of ADAPT speakers per dataset (bona fide and spoofs of the same speakers)
keep_spk = {}
for ds in SETS:
    spk = sorted({i["speaker"] for i in items if i["dataset"] == ds and i["split"] == "adapt" and i["system"] == "bonafide"})
    random.Random(1000 + args.seed).shuffle(spk)
    keep_spk[ds] = set(spk[:max(2, int(round(len(spk) * args.budget)))])
train_ds = list(SETS) if args.train == "both" else [args.train]
train = [i for i in items if i["split"] == "adapt" and i["dataset"] in train_ds and i["speaker"] in keep_spk[i["dataset"]]
         and (i["system"] == "bonafide" or i["system"] in TRAIN_SYSTEMS or (args.aug == "ch" and i["system"].endswith("+ch") and base_sys(i["system"]) in TRAIN_SYSTEMS))]
# model selection on dev: bona fide vs the training systems, BOTH raw and channel-matched (same rule for every run)
devset = [i for i in items if i["split"] == "dev" and i["dataset"] in train_ds and (i["system"] == "bonafide" or base_sys(i["system"]) in TRAIN_SYSTEMS)]
evalset = [i for i in items if i["split"] in ("dev", "test")]
if args.smoke:
    random.Random(args.seed).shuffle(devset); devset = devset[:600]
    evalset = [i for i in evalset if i["split"] == "test"]; random.Random(args.seed).shuffle(evalset); evalset = evalset[:3000]
ADF_CKPT = os.path.join(ROOT, "models", "antideepfake_mms300m", "model.safetensors")
dev_pool = devset
devset = random.Random(1234).sample(devset, min(args.dev_n, len(devset)))      # fixed subset, same for every run
cnt = collections.Counter((i["dataset"], i["system"]) for i in train)
print("train:", dict(sorted(cnt.items())))
print(f"train speakers: " + ", ".join(f"{ds} {len(keep_spk[ds])}" for ds in train_ds) + f"   dev subset: {len(devset)}   eval utterances: {len(evalset)}")

# ---------------- model ----------------
from kf_antideepfake import load_antideepfake, prep

class LoRA(nn.Module):
    def __init__(self, base, r, alpha=16):
        super().__init__()
        self.base = base; self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) * (1.0 / math.sqrt(base.in_features)))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))        # zero: starts exactly as the pretrained layer
    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

class ASP(nn.Module):
    """attentive statistics pooling; zero-initialised so that it starts as plain mean pooling + the pretrained linear."""
    def __init__(self, proj_fc, d=1024, h=128):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, 1))
        nn.init.zeros_(self.att[2].weight); nn.init.zeros_(self.att[2].bias)
        self.fc = nn.Linear(2 * d, 2)
        with torch.no_grad():
            self.fc.weight.zero_(); self.fc.weight[:, :d] = proj_fc.weight; self.fc.bias.copy_(proj_fc.bias)
    def forward(self, h):                                                  # h: (B, T, D)
        w = torch.softmax(self.att(h).squeeze(-1), dim=1).unsqueeze(-1)
        mu = (w * h).sum(1); sd = torch.sqrt(((w * (h - mu.unsqueeze(1)) ** 2).sum(1)).clamp(min=1e-6))
        return self.fc(torch.cat([mu, sd], dim=-1))

class Detector(nn.Module):
    def __init__(self, adf, variant, r):
        super().__init__()
        self.enc = adf.m_ssl.model
        for p in self.enc.parameters():
            p.requires_grad = False
        for layer in self.enc.encoder.layers:
            a = layer.self_attn
            a.q_proj, a.k_proj, a.v_proj, a.out_proj = (LoRA(m, r) for m in (a.q_proj, a.k_proj, a.v_proj, a.out_proj))
        self.asp = (variant == "proposed" and not args.no_asp)
        self.head = ASP(adf.proj_fc) if self.asp else adf.proj_fc
    def forward(self, wav):
        h = self.enc(wav)
        return self.head(h) if self.asp else self.head(h.mean(1))
    def trainable(self):
        return {k: v for k, v in self.state_dict().items() if ".A" in k[-2:] or ".B" in k[-2:] or k.startswith("head.")}

adf, _ = load_antideepfake(ADF_CKPT, "cpu")
model = Detector(adf, args.variant, args.rank).to(dev)
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable parameters: {n_tr:,} (LoRA r={args.rank} + head)")
amp = dict(device_type="cuda", dtype=torch.bfloat16 if (dev == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16, enabled=dev == "cuda")

def score_list(lst):
    """official protocol: whole utterance, per-utterance layer norm; equal-length groups, no padding."""
    model.eval(); out = [None] * len(lst)
    order = sorted(range(len(lst)), key=lambda k: lst[k]["n"] if "n" in lst[k] else 0)
    wavs = {}
    with torch.no_grad():
        for k in range(0, len(lst), 64):
            idx = list(range(k, min(k + 64, len(lst))))
            xs = [load(lst[j]["path"]) for j in idx]
            groups = collections.defaultdict(list)
            for j, x in zip(idx, xs):
                groups[len(x)].append((j, x))
            for L, g in groups.items():
                for b in range(0, len(g), 16):
                    part = g[b:b + 16]
                    with torch.autocast(**amp):
                        lg = model(torch.stack([prep(x) for _, x in part]).to(dev)).float().cpu()
                    for (j, _), s in zip(part, (lg[:, 1] - lg[:, 0]).tolist()):
                        out[j] = s
    return np.array(out)

def dev_eer():
    s = score_list(devset); y = np.array([i["label"] for i in devset])
    return eer(s[y == 1], s[y == 0])[0]

# ---------------- blind channel augmentation (training only) ----------------
def hp_fft(x, fc):
    X = np.fft.rfft(x); X[np.fft.rfftfreq(len(x), 1 / SR) < fc] = 0
    return np.fft.irfft(X, n=len(x)).astype(np.float32)
def lp_fft(x, fc):
    X = np.fft.rfft(x); X[np.fft.rfftfreq(len(x), 1 / SR) > fc] = 0
    return np.fft.irfft(X, n=len(x)).astype(np.float32)
def blind_channel(x, rng):
    y = x.copy(); applied = 0
    if rng.random() < 0.6:                                               # mains hum: 45-65 Hz + 2nd/3rd harmonics
        f0 = rng.uniform(45, 65); t = np.arange(len(y)) / SR
        hum = sum(rng.uniform(0.2, 1.0) / k * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi)) for k in (1, 2, 3))
        snr = rng.uniform(5, 25)
        hum *= np.sqrt(np.mean(y ** 2) + 1e-9) / (np.sqrt(np.mean(hum ** 2)) + 1e-9) * 10 ** (-snr / 20)
        y = y + hum.astype(np.float32); applied += 1
    if rng.random() < 0.3:                                               # random high-pass 60-150 Hz
        y = hp_fft(y, rng.uniform(60, 150)); applied += 1
    if rng.random() < 0.2:                                               # random microphone / phone band limit
        y = lp_fft(y, rng.uniform(3400, 7000)); applied += 1
    if rng.random() < 0.3:                                               # random clipping (hard or soft)
        g = rng.uniform(1.5, 4.0) / (np.max(np.abs(y)) + 1e-9)
        y = np.clip(y * g, -1, 1) if rng.random() < 0.5 else np.tanh(y * g)
        y = y / g; applied += 1
    if applied == 0:
        y = hp_fft(y, rng.uniform(60, 150))
    return y.astype(np.float32)

def crop(x, L, rng):
    if len(x) <= L:
        return x[:L]
    s = int(rng.integers(0, len(x) - L + 1)); return x[s:s + L]

# ---------------- training ----------------
if not args.eval_only:
    t0 = time.time()
    print("loading training audio into memory ...", flush=True)
    for k, i in enumerate(train):
        i["x"] = (load(i["path"]) * 32767).astype(np.int16)
        if k % 10000 == 0:
            print(f"  {k}/{len(train)}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    idx_b = np.array([k for k, i in enumerate(train) if i["label"] == 1]); idx_s = np.array([k for k, i in enumerate(train) if i["label"] == 0])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * args.steps, eta_min=args.lr / 50)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda" and amp["dtype"] == torch.float16))
    rng = np.random.default_rng(args.seed)
    log = [("epoch", "train_loss", "dev_eer", "lr", "minutes")]
    e0 = dev_eer(); best, best_ep, bad = e0, 0, 0
    print(f"epoch 0 (zero-shot)  dev EER {e0:.2%}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    torch.save(model.trainable(), os.path.join(out_dir, "best_lora_head.pth"))
    for ep in range(1, args.epochs + 1):
        model.train(); tot, nb, t_ep = 0.0, 0, time.time()
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            for _ in range(args.accum):
                h = args.batch // 2
                sel = np.concatenate([rng.choice(idx_b, h), rng.choice(idx_s, args.batch - h)])
                xs = [train[j]["x"].astype(np.float32) / 32767 for j in sel]
                L = int(min(min(len(x) for x in xs), args.max_s * SR))
                xs = [crop(x, L, rng) for x in xs]
                y = torch.tensor([train[j]["label"] for j in sel], device=dev)
                xb = torch.stack([prep(x) for x in xs]).to(dev)
                with torch.autocast(**amp):
                    lg = model(xb).float()
                    loss = F.cross_entropy(lg, y)
                    if args.variant == "proposed":
                        xa = torch.stack([prep(blind_channel(x, rng)) for x in xs]).to(dev)
                        la = model(xa).float()
                        s1, s2 = lg[:, 1] - lg[:, 0], la[:, 1] - la[:, 0]
                        loss = loss + F.cross_entropy(la, y) + args.lam * ((s1 - s2) ** 2).mean()
                scaler.scale(loss / args.accum).backward(); tot += float(loss) / args.accum
            scaler.step(opt); scaler.update(); sched.step(); nb += 1
            if nb == 10 and ep == 1:
                print(f"  speed check: {(time.time()-t_ep)/10:.2f} s per step ({args.accum}x{args.batch})  -> about "
                      f"{(time.time()-t_ep)/10*args.steps/60:.0f} min per epoch of {args.steps} steps (+ dev scoring)", flush=True)
            if nb % 100 == 0:
                print(f"  epoch {ep}  step {nb}  loss {tot/nb:.4f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        e = dev_eer()
        log.append((ep, round(tot / nb, 5), round(e, 4), f"{sched.get_last_lr()[0]:.2e}", round((time.time() - t0) / 60, 1)))
        print(f"epoch {ep}  loss {tot/nb:.4f}  dev EER {e:.2%}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        if e < best - 1e-4:
            best, best_ep, bad = e, ep, 0; torch.save(model.trainable(), os.path.join(out_dir, "best_lora_head.pth"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop (no dev improvement for {args.patience} epochs)"); break
    with open(os.path.join(out_dir, "train_log.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(log)
    for i in train:
        i.pop("x", None)
else:
    e0 = best = float("nan"); best_ep = "(re-scored)"

# ---------------- evaluation of the best model ----------------
sd = torch.load(os.path.join(out_dir, "best_lora_head.pth"), map_location=dev, weights_only=True)
missing = model.load_state_dict(sd, strict=False)
print(f"\nbest epoch {best_ep} (dev EER {best:.2%}); scoring {len(evalset)} dev+test utterances ...", flush=True)
s_all = score_list(evalset)
with open(os.path.join(out_dir, "scores.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["utterance_id", "dataset", "split", "system", "label", "score"])
    for i, s in zip(evalset, s_all):
        w.writerow([i["uid"], i["dataset"], i["split"], i["system"], "bonafide" if i["label"] else "spoof", round(float(s), 5)])

S = collections.defaultdict(list)
for i, s in zip(evalset, s_all):
    S[(i["dataset"], i["split"], i["system"])].append(s)
S = {k: np.array(v) for k, v in S.items()}
systems = sorted({k[2] for k in S if k[2] != "bonafide"}, key=lambda x: (x.endswith("+ch"), x))
L = [f"KurdFake fine-tuned CM   run {tag}   {time.strftime('%Y-%m-%d %H:%M')}",
     f"model MMS-300M-AntiDeepfake  E4-{args.variant}{' no-ASP' if args.no_asp else ''}  aug {args.aug}  LoRA r={args.rank}  trained on {args.train}  speakers " + ", ".join(f"{d} {len(keep_spk[d])}" for d in train_ds)
     + (f"  utterances {len(train)}  best epoch {best_ep}  dev EER {best:.2%} (zero-shot {e0:.2%})" if not args.eval_only else "  [re-scored best.pth with --eval_only]"),
     f"training systems: {', '.join(sorted(TRAIN_SYSTEMS))}   held out: B, BC, D2   (+ch = same spoof with the real channel of the target speaker)", ""]
for ds in SETS:
    bt, bd = S.get((ds, "test", "bonafide")), S.get((ds, "dev", "bonafide"))
    if bt is None:
        continue
    L.append(f"== {ds.upper()} test  ({'in-domain' if ds in train_ds else 'CROSS-DATASET: not used in training'})")
    for cond_name, suf in (("raw", False), ("+ch", True)):
        sy = [s for s in systems if s.endswith("+ch") == suf and (ds, "test", s) in S]
        if not sy:
            continue
        L.append(f"  [{cond_name}] EER per system: " + "  ".join(f"{s} {100*eer(bt, S[(ds,'test',s)])[0]:5.1f}%" for s in sy))
        for name, group in (("seen    ", [s for s in sy if base_sys(s) in TRAIN_SYSTEMS]), ("held-out", [s for s in sy if base_sys(s) not in TRAIN_SYSTEMS])):
            st = [S[(ds, "test", s)] for s in group]; sd = [S[(ds, "dev", s)] for s in group if (ds, "dev", s) in S]
            if not st or bd is None or not sd:
                continue
            r = report(bd, np.concatenate(sd), bt, np.concatenate(st))
            L.append(f"  [{cond_name}] {name}: EER {100*r['eer']:5.1f}%  minDCF {r['min_dcf']:.3f}  actDCF {r['act_dcf']:.3f}  Cllr {r['cllr']:.3f}"
                     f"  | dev thr: bona fide rejected {100*r['frr_at_dev_thr']:.1f}%  spoof accepted {100*r['far_at_dev_thr']:.1f}%")
    L.append("")
txt = "\n".join(L)
open(os.path.join(out_dir, "results.txt"), "w", encoding="utf-8").write(txt + "\n")
print("\n" + txt + f"\nSaved to {out_dir}   total {(time.time()-t0)/60:.1f} min")
