# =====================================================================
# KurdFake - significance.py   (paired speaker-level bootstrap: E4-ch vs E2, 3 seeds each)
# For RAIL and CV test, raw and +ch spoofs, held-out systems (B, BC, D2) and all systems:
#   point estimates per seed: EER, minDCF, actDCF (ASVspoof 5 cost: Cmiss=1, Cfa=10, pi_spoof=0.05;
#   actDCF = Bayes threshold on LLRs calibrated on dev, as in kf_metrics.report)
#   paired bootstrap over TEST SPEAKERS (1,000 resamples; the same resample is applied to all 6 score sets);
#   per resample each model's metric is averaged over its 3 seeds; delta = E4-ch - E2 (negative = E4-ch better);
#   95% percentile CI of delta and two-sided bootstrap p-value.
# Reads results\ft\<run>\scores.csv and the manifests (for speaker ids). No GPU.
#   python significance.py      (about 3-6 minutes)  -> results\significance.txt
# =====================================================================
import os, csv, sys, time, collections
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np
ROOT = KF_ROOT; DATA = os.path.join(ROOT, "Dataset"); FT = os.path.join(ROOT, "results", "ft")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kf_metrics import fit_calibration
C_MISS, C_FA, P_SPOOF = 1.0, 10.0, 0.05
NORM = min(C_MISS * (1 - P_SPOOF), C_FA * P_SPOOF); T_BAYES = np.log((C_FA * P_SPOOF) / (C_MISS * (1 - P_SPOOF)))
SEEDS = (1337, 2026, 42); NB = 1000
MODELS = {"E2": [f"AASIST_control_train-both_b100_s{s}" for s in SEEDS],
          "E4-ch": [f"ADF-MMS300M_E4-base_aug-ch_train-both_b100_s{s}" for s in SEEDS]}
HELD = {"B", "BC", "D2"}

# ---------- speakers ----------
spk = {}
for ds, base in (("rail", "RAIL_clean"), ("cv", "CV_ckb_clean")):
    with open(os.path.join(DATA, base, "manifest.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            spk[r["utt_id"]] = r["speaker"]
for mp in [os.path.join(DATA, "Spoofs", f) for f in ["spoof_manifest_knnvc_final.csv", "spoof_manifest_knnvc_D2.csv",
           "spoof_manifest_tts_vekol.csv", "spoof_manifest_f5.csv"]] + [os.path.join(DATA, "Spoofs_CM", "spoof_manifest_channel_matched.csv")]:
    with open(mp, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            spk[r["utt_id"]] = r["target_speaker"]

# ---------- scores ----------
def read(run):
    out = {}
    with open(os.path.join(FT, run, "scores.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["utterance_id"]] = (r["dataset"], r["split"], r["system"], float(r["score"]))
    return out
runs = {m: [read(r) for r in rs] for m, rs in MODELS.items()}
ref = runs["E2"][0]
ids = sorted(ref)

# ---------- fast metrics ----------
def rates(b, s, t):
    bs, ss = np.sort(b), np.sort(s)
    pm = np.searchsorted(bs, t, "left") / len(bs); pf = 1 - np.searchsorted(ss, t, "left") / len(ss)
    return pm, pf
def eer_mindcf(b, s):
    t = np.append(np.unique(np.concatenate([b, s])), np.inf)
    pm, pf = rates(b, s, t)
    k = int(np.argmin(np.abs(pm - pf)))
    return (pm[k] + pf[k]) / 2, float(np.min((C_MISS * (1 - P_SPOOF) * pm + C_FA * P_SPOOF * pf) / NORM))
def act_dcf(lb, ls):
    pm = np.mean(lb < T_BAYES); pf = np.mean(ls >= T_BAYES)
    return (C_MISS * (1 - P_SPOOF) * pm + C_FA * P_SPOOF * pf) / NORM

L = [f"KurdFake paired speaker-level bootstrap, E4-ch vs E2 (3 seeds each, {NB} resamples)  {time.strftime('%Y-%m-%d %H:%M')}",
     "delta = E4-ch minus E2 (negative = E4-ch better). Values in % for EER; minDCF / actDCF normalised (ASVspoof 5 cost).", ""]
t0 = time.time()
for ds in ("rail", "cv"):
    for sset in ("raw", "+ch"):
        for group in ("held-out", "all"):
            def keep(u, split):
                d, sp, sy, _ = ref[u]
                if d != ds or sp != split:
                    return None
                if sy == "bonafide":
                    return "b"
                base = sy[:-3] if sy.endswith("+ch") else sy
                if sy.endswith("+ch") != (sset == "+ch"):
                    return None
                if group == "held-out" and base not in HELD:
                    return None
                return "s"
            sel = {sp: [(u, keep(u, sp)) for u in ids] for sp in ("dev", "test")}
            sel = {sp: [(u, k) for u, k in v if k] for sp, v in sel.items()}
            tst = sel["test"]; lab = np.array([k == "b" for _, k in tst]); sp_of = np.array([spk.get(u, "?") for u, _ in tst])
            speakers = np.unique(sp_of); idx_by = {s: np.where(sp_of == s)[0] for s in speakers}
            M = {}   # model -> list over seeds of (test scores, llr test)
            for m, rs in runs.items():
                M[m] = []
                for r in rs:
                    sd = np.array([r[u][3] for u, _ in sel["dev"]]); ld = np.array([k == "b" for _, k in sel["dev"]])
                    cal = fit_calibration(sd[ld], sd[~ld])
                    st = np.array([r[u][3] for u, _ in tst]); M[m].append((st, cal(st)))
            def metrics(ix):
                out = {}
                for m, lst in M.items():
                    v = []
                    for st, lt in lst:
                        b, s = st[ix][lab[ix]], st[ix][~lab[ix]]
                        e, md = eer_mindcf(b, s); ad = act_dcf(lt[ix][lab[ix]], lt[ix][~lab[ix]])
                        v.append((100 * e, md, ad))
                    out[m] = np.array(v)          # seeds x 3
                return out
            full = metrics(np.arange(len(tst)))
            rng = np.random.default_rng(2026); deltas = []
            for _ in range(NB):
                pick = rng.choice(speakers, len(speakers), replace=True)
                ix = np.concatenate([idx_by[s] for s in pick])
                mm = metrics(ix); deltas.append(mm["E4-ch"].mean(0) - mm["E2"].mean(0))
            deltas = np.array(deltas)
            L.append(f"== {ds.upper()} test  {sset:4s}  {group:8s}  (speakers {len(speakers)}, bona fide {lab.sum()}, spoofs {(~lab).sum()})")
            for j, name in enumerate(("EER %", "minDCF", "actDCF")):
                e2, e4 = full["E2"][:, j], full["E4-ch"][:, j]
                d = deltas[:, j]; lo, hi = np.percentile(d, [2.5, 97.5])
                p = min(1.0, 2 * min((d >= 0).mean(), (d <= 0).mean()))
                L.append(f"   {name:7s} E2 {e2.mean():7.3f} +- {e2.std(ddof=1):.3f}   E4-ch {e4.mean():7.3f} +- {e4.std(ddof=1):.3f}"
                         f"   delta {e4.mean()-e2.mean():+8.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p {'<0.001' if p < 0.001 else f'{p:.3f}'}")
            print(L[-4]); print(L[-3]); print(f"   ({(time.time()-t0)/60:.1f} min)", flush=True)
L.append("")
L.append("p-value: two-sided bootstrap (share of resamples on the other side of zero, x2); p<0.001 = no resample crossed zero.")
txt = "\n".join(L)
open(os.path.join(ROOT, "results", "significance.txt"), "w", encoding="utf-8").write(txt + "\n")
print("\n" + txt)
