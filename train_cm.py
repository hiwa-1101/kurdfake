# =====================================================================
# KurdFake - train_cm.py   (RQ3 / RQ4: can a little Kurdish data adapt an English CM, and does it generalise?)
#   Starts from the official AASIST weights (ASVspoof 2019 LA) and fine-tunes on the Kurdish ADAPT split.
#   Training spoofs  : D, G, C1, C1D            (adapt speakers only)
#   Held out (never seen in training or model selection):
#       D2  = same kNN-VC family, unseen attacker references
#       B, BC = F5-TTS, an unseen family (flow matching + Vocos)
#   Model selection  : best pooled EER on the DEV split, bona fide vs the four training systems only.
#   Test             : every system on RAIL test and CV test, thresholds from dev.
#
# Runs (one at a time; each prints its own tag):
#   python train_cm.py                                   E1  main: RAIL+CV, raw, all adapt speakers
#   python train_cm.py --cond control                    E2  baseline: 100 Hz high-pass on every file (removes the 50 Hz hum shortcut)
#   python train_cm.py --aug ch                          E3  adds the channel-matched adapt spoofs (Spoofs_CM) to training
#   python train_cm.py --train rail                      E3a cross-dataset: train RAIL, test on CV
#   python train_cm.py --train cv                        E3b cross-dataset: train CV, test on RAIL
#   python train_cm.py --budget 0.1                      E4  data budget: 10% of adapt speakers
#   python train_cm.py --budget 0.3                      E4  data budget: 30% of adapt speakers
#   (optional, only if E1 is unstable: --seed 2, --seed 3)
#   --smoke : 2 short epochs on 5% of the data, to check that everything runs (about 5 minutes)
#   Training budget: 12 epochs x 400 class-balanced batches of 24 (= 115k examples, about 3 passes over the full adapt set),
#   early stop after 3 epochs without dev improvement. Same budget for every run, so runs are comparable.
#
# Writes: results\ft\<tag>\  best.pth, train_log.csv, scores.csv (dev+test), results.txt
# =====================================================================
import os, sys, csv, json, wave, time, random, argparse, collections, platform
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np
import torch
import torch.nn as nn

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
SR, NB_SAMP, TGT_LUFS, HP_HZ = 16000, 64600, -16.0, 100

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="AASIST", choices=["AASIST", "AASIST-L"])
ap.add_argument("--cond", default="raw", choices=["raw", "control"])
ap.add_argument("--aug", default="none", choices=["none", "ch"], help="ch = also train on the channel-matched adapt spoofs (E3)")
ap.add_argument("--train", default="both", choices=["both", "rail", "cv"])
ap.add_argument("--budget", type=float, default=1.0, help="fraction of adapt speakers used (per dataset)")
ap.add_argument("--epochs", type=int, default=12, help="an epoch = --steps batches (class-balanced sampling)")
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--eval_only", action="store_true", help="no training: re-score the existing best.pth of this run (same tag) on the current dev/test sets incl. +ch")
args = ap.parse_args()
if args.smoke:
    args.epochs, args.budget, args.steps = 2, 0.05, 40
tag = f"{args.model}_{args.cond}{'_aug-ch' if args.aug == 'ch' else ''}_train-{args.train}_b{int(round(args.budget*100))}_s{args.seed}" + ("_smoke" if args.smoke else "")
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

def pad_eval(x):                                  # aasist/data_utils.pad
    return x[:NB_SAMP] if len(x) >= NB_SAMP else np.tile(x, int(NB_SAMP / len(x)) + 1)[:NB_SAMP]

def pad_train(x):                                 # aasist/data_utils.pad_random
    if len(x) >= NB_SAMP:
        s = np.random.randint(0, len(x) - NB_SAMP + 1); return x[s:s + NB_SAMP]
    return np.tile(x, int(NB_SAMP / len(x)) + 1)[:NB_SAMP]

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
cnt = collections.Counter((i["dataset"], i["system"]) for i in train)
print("train:", dict(sorted(cnt.items())))
print(f"train speakers: " + ", ".join(f"{ds} {len(keep_spk[ds])}" for ds in train_ds) + f"   dev utterances: {len(devset)}   eval utterances: {len(evalset)}")

t0 = time.time()
if not args.eval_only:
    print("loading training audio into memory ...", flush=True)
    for k, i in enumerate(train + devset):
        if "x" not in i:
            i["x"] = (load(i["path"]) * 32767).astype(np.int16)
        if k % 5000 == 0:
            print(f"  {k}/{len(train) + len(devset)}  ({(time.time()-t0)/60:.1f} min)", flush=True)

# ---------------- model ----------------
import importlib.util
_spec = importlib.util.spec_from_file_location("aasist_model", os.path.join(AASIST, "models", "AASIST.py"))
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
cfg = json.load(open(os.path.join(AASIST, "config", f"{args.model}.conf")))
model = _mod.Model(cfg["model_config"])
model.load_state_dict(torch.load(os.path.join(AASIST, "models", "weights", f"{args.model}.pth"), map_location="cpu", weights_only=True))
model = model.to(dev)
idx_b = np.array([k for k, i in enumerate(train) if i["label"] == 1]); idx_s = np.array([k for k, i in enumerate(train) if i["label"] == 0])
crit = nn.CrossEntropyLoss()          # classes are balanced by sampling: each batch is half bona fide, half spoof
opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
steps = args.epochs * args.steps
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=args.lr / 50)

def score(lst, from_memory):
    model.eval(); out = []
    with torch.no_grad():
        for b in range(0, len(lst), 32):
            chunk = lst[b:b + 32]
            xs = [pad_eval(c["x"].astype(np.float32) / 32767 if from_memory else load(c["path"])) for c in chunk]
            _, o = model(torch.from_numpy(np.stack(xs)).to(dev))
            out += o[:, 1].float().cpu().tolist()
    return np.array(out)

def dev_eer():
    s = score(devset, True); y = np.array([i["label"] for i in devset])
    return eer(s[y == 1], s[y == 0])[0]

# ---------------- training ----------------
if args.eval_only:
    e0 = best = float("nan"); best_ep = "(re-scored, see train_log.csv)"
else:
    log = [("epoch", "train_loss", "dev_eer", "lr", "minutes")]
    e0 = dev_eer(); log.append((0, "", round(e0, 4), "", round((time.time() - t0) / 60, 1)))
    print(f"epoch 0 (zero-shot)  dev EER {e0:.2%}", flush=True)
    best, best_ep, bad = e0, 0, 0
    torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
    for ep in range(1, args.epochs + 1):
        model.train(); tot, nb = 0.0, 0; t_ep = time.time()
        for b in range(args.steps):
            h = args.batch // 2
            sel = np.concatenate([np.random.choice(idx_b, h), np.random.choice(idx_s, args.batch - h)])
            chunk = [train[j] for j in sel]
            x = torch.from_numpy(np.stack([pad_train(c["x"].astype(np.float32) / 32767) for c in chunk])).to(dev)
            y = torch.tensor([c["label"] for c in chunk], device=dev)
            _, o = model(x); loss = crit(o, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss); nb += 1
            if nb == 20 and ep == 1:
                torch.cuda.synchronize() if dev == "cuda" else None
                print(f"  speed check: {(time.time()-t_ep)/20:.2f} s per batch of {args.batch}  -> about {(time.time()-t_ep)/20*args.steps/60:.0f} min per epoch of {args.steps} batches", flush=True)
            if nb % 200 == 0:
                print(f"  epoch {ep}  batch {nb}  loss {tot/nb:.4f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        e = dev_eer()
        log.append((ep, round(tot / nb, 5), round(e, 4), f"{sched.get_last_lr()[0]:.2e}", round((time.time() - t0) / 60, 1)))
        print(f"epoch {ep}  loss {tot/nb:.4f}  dev EER {e:.2%}  ({(time.time()-t0)/60:.1f} min)", flush=True)
        if e < best - 1e-4:
            best, best_ep, bad = e, ep, 0; torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop (no dev improvement for {args.patience} epochs)"); break
    with open(os.path.join(out_dir, "train_log.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(log)

# ---------------- evaluation of the best model ----------------
for i in train + devset:
    i.pop("x", None)
if args.eval_only and not os.path.exists(os.path.join(out_dir, "best.pth")):
    raise SystemExit(f"no best.pth in {out_dir}")
model.load_state_dict(torch.load(os.path.join(out_dir, "best.pth"), map_location=dev, weights_only=True))
print(f"\nbest epoch {best_ep} (dev EER {best:.2%}); scoring {len(evalset)} dev+test utterances ...", flush=True)
s_all = score(evalset, False)
with open(os.path.join(out_dir, "scores_v4eval.csv" if args.eval_only else "scores.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["utterance_id", "dataset", "split", "system", "label", "score"])
    for i, s in zip(evalset, s_all):
        w.writerow([i["uid"], i["dataset"], i["split"], i["system"], "bonafide" if i["label"] else "spoof", round(float(s), 5)])

S = collections.defaultdict(list)
for i, s in zip(evalset, s_all):
    S[(i["dataset"], i["split"], i["system"])].append(s)
S = {k: np.array(v) for k, v in S.items()}
systems = sorted({k[2] for k in S if k[2] != "bonafide"}, key=lambda x: (x.endswith("+ch"), x))
L = [f"KurdFake fine-tuned CM   run {tag}   {time.strftime('%Y-%m-%d %H:%M')}",
     f"model {args.model}  condition {args.cond}  aug {args.aug}  trained on {args.train}  speakers " + ", ".join(f"{d} {len(keep_spk[d])}" for d in train_ds)
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
open(os.path.join(out_dir, "results_v4eval.txt" if args.eval_only else "results.txt"), "w", encoding="utf-8").write(txt + "\n")
print("\n" + txt + f"\nSaved to {out_dir}   total {(time.time()-t0)/60:.1f} min")
