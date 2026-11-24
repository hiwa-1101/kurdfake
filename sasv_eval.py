# =====================================================================
# KurdFake - sasv_eval.py   (system-level metric: ASV + CM together, ASVspoof 5 a-DCF)
#   No new audio processing: reads results\asv_scores.csv and results\cm_scores.csv.
#   Question answered: does putting a zero-shot CM in front of ECAPA reduce the real cost
#   of the whole verification system, or not?
#
#   a-DCF (Shim et al., Odyssey 2024; ASVspoof 5 SASV track):
#     a-DCF = C_miss*pi_tar*P_miss + C_fa_non*pi_non*P_fa_non + C_fa_spf*pi_spf*P_fa_spf,
#     normalised by the cost of the best "accept all / reject all" system.
#     Parameters = ASVspoof 5: C_miss=1, C_fa_non=10, C_fa_spf=10, pi_tar=0.9, pi_non=0.05, pi_spf=0.05
#   Systems compared (per dataset, test split; thresholds fixed on dev = 'actual', or swept = 'min'):
#     ASV only            : accept if ASV >= t_asv
#     ASV + CM (cascade)  : accept if ASV >= t_asv AND CM >= t_cm   (for every CM model x condition)
#
# Anaconda Prompt:   conda activate gpu_env  ->  cd /d <KURDFAKE_ROOT>  ->  python sasv_eval.py
# Writes: results\sasv_results.txt
# =====================================================================
import os, csv, collections, datetime
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np

RES = os.path.join(KF_ROOT, "results")
C_MISS, C_FA_NON, C_FA_SPF = 1.0, 10.0, 10.0
PI_TAR, PI_NON, PI_SPF = 0.90, 0.05, 0.05
NORM = min(C_MISS * PI_TAR, C_FA_NON * PI_NON + C_FA_SPF * PI_SPF)
GRID = 200                                     # threshold grid per axis for the 'min' search

def adcf(pm, pfn, pfs):
    return (C_MISS * PI_TAR * pm + C_FA_NON * PI_NON * pfn + C_FA_SPF * PI_SPF * pfs) / NORM

def rates(tar, non, spf, acc_tar, acc_non, acc_spf):
    return 1 - acc_tar.mean(), acc_non.mean(), acc_spf.mean()

def eer_thr(pos, neg):
    s = np.sort(np.concatenate([pos, neg])); best, bt = 9, s[0]
    for t in np.quantile(s, np.linspace(0, 1, 2001)):
        d = abs((pos < t).mean() - (neg >= t).mean())
        if d < best: best, bt = d, t
    return bt

# ---------------- load ----------------
cm = {}
with open(os.path.join(RES, "cm_scores.csv"), encoding="utf-8") as fh:
    rd = csv.DictReader(fh); cm_cols = [c for c in rd.fieldnames if c.startswith("AASIST")]
    for r in rd:
        cm[r["utterance_id"]] = {c: float(r[c]) for c in cm_cols}
trials = collections.defaultdict(list)          # (dataset, split) -> rows
miss_cm = 0
with open(os.path.join(RES, "asv_scores.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["utterance_id"] not in cm:
            miss_cm += 1; continue
        lab = r["label"].strip().lower()
        r["kind"] = "spf" if lab == "spoof" else ("tar" if lab in ("target", "tar", "1", "true") else "non")
        trials[(r["dataset"], r["split"])].append(r)

def arrays(rows, col):
    a = {k: np.array([float(r["score"]) for r in rows if r["kind"] == k]) for k in ("tar", "non", "spf")}
    c = {k: np.array([cm[r["utterance_id"]][col] for r in rows if r["kind"] == k]) if col else None for k in ("tar", "non", "spf")}
    sysn = np.array([r["system"] for r in rows if r["kind"] == "spf"])
    return a, c, sysn

out = [f"KurdFake SASV evaluation (a-DCF, ASVspoof 5 parameters)  {datetime.datetime.now():%Y-%m-%d %H:%M}",
       f"C_miss={C_MISS} C_fa_non={C_FA_NON} C_fa_spf={C_FA_SPF}  pi_tar={PI_TAR} pi_non={PI_NON} pi_spf={PI_SPF}",
       "a-DCF = 1.0 means no better than a system that accepts or rejects everyone; lower is better.",
       "act = thresholds fixed on dev (what a deployed system would do); min = best possible threshold on test (oracle).", ""]
if miss_cm:
    out.append(f"NOTE: {miss_cm} ASV rows had no CM score and were skipped.\n")

for ds in ("rail", "cv"):
    dev, test = trials[(ds, "dev")], trials[(ds, "test")]
    if not test:
        continue
    out.append(f"===== {ds.upper()} test   (target={sum(r['kind']=='tar' for r in test)}  nontarget={sum(r['kind']=='non' for r in test)}  spoof={sum(r['kind']=='spf' for r in test)})")
    ad, _, _ = arrays(dev, None); at, _, sysn = arrays(test, None)
    t_asv = eer_thr(ad["tar"], ad["non"])          # ASV threshold: EER on bona fide dev trials (as in RQ1)
    grid_a = np.quantile(np.concatenate([at["tar"], at["non"]]), np.linspace(0, 1, GRID))
    # ASV only
    def asv_only(t):
        return rates(None, None, None, at["tar"] >= t, at["non"] >= t, at["spf"] >= t)
    pm, pfn, pfs = asv_only(t_asv); act = adcf(pm, pfn, pfs)
    mn = min(adcf(*asv_only(t)) for t in grid_a)
    out.append(f"  ASV only              act a-DCF {act:.3f}  (miss {pm:.1%}  FA-non {pfn:.1%}  FA-spoof {pfs:.1%})   min a-DCF {mn:.3f}")
    rows_sys = []
    for col in cm_cols:
        cdv = arrays(dev, col)[1]; _, ct, _ = arrays(test, col)
        bona_dev_cm = np.concatenate([cdv["tar"], cdv["non"]])
        t_cm = eer_thr(bona_dev_cm, cdv["spf"])     # CM threshold: CM EER on dev
        acc = lambda ta, tc: (at["tar"] >= ta) & (ct["tar"] >= tc), lambda ta, tc: (at["non"] >= ta) & (ct["non"] >= tc), lambda ta, tc: (at["spf"] >= ta) & (ct["spf"] >= tc)
        def cas(ta, tc):
            return 1 - acc[0](ta, tc).mean(), acc[1](ta, tc).mean(), acc[2](ta, tc).mean()
        pm, pfn, pfs = cas(t_asv, t_cm); act = adcf(pm, pfn, pfs)
        grid_c = np.quantile(np.concatenate([ct["tar"], ct["non"], ct["spf"]]), np.linspace(0, 1, GRID // 4))
        mn = min(adcf(*cas(ta, tc)) for ta in grid_a[::4] for tc in np.append(grid_c, -np.inf))
        out.append(f"  ASV + {col:<16} act a-DCF {act:.3f}  (miss {pm:.1%}  FA-non {pfn:.1%}  FA-spoof {pfs:.1%})   min a-DCF {mn:.3f}")
        sp_acc = acc[2](t_asv, t_cm)
        rows_sys.append((col, {s: sp_acc[sysn == s].mean() for s in sorted(set(sysn))}))
    s_all = sorted(set(sysn))
    asv_acc = at["spf"] >= t_asv
    out.append("  spoofs accepted per system (dev thresholds):   " + "  ".join(f"{s:>4}" for s in s_all))
    out.append(f"    {'ASV only':<22}" + "".ljust(24) + "  ".join(f"{asv_acc[sysn==s].mean():4.0%}" for s in s_all))
    for col, d in rows_sys:
        out.append(f"    ASV + {col:<16}" + "".ljust(24) + "  ".join(f"{d[s]:4.0%}" for s in s_all))
    out.append("")

txt = "\n".join(out); print(txt)
with open(os.path.join(RES, "sasv_results.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt + "\n")
print(f"Saved to {os.path.join(RES, 'sasv_results.txt')}")
