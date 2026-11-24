# =====================================================================
# KurdFake - sasv_final.py   (final system-level metric: ECAPA ASV + adapted CM, a-DCF of ASVspoof 5)
# Reads results\asv_scores_with_ch.csv (from asv_eval.py, includes the +ch spoofs) and the CM scores of the
# locked comparison runs (3 seeds each). No audio is processed.
#   Spoof sets are evaluated separately: raw spoofs, and channel-matched (+ch) spoofs.
#   Thresholds: ASV = EER on dev bona fide trials; CM = EER on dev (bona fide vs all dev spoofs of that set).
#   act = dev thresholds (deployed system); min = oracle thresholds on test.
#   python sasv_final.py     -> results\sasv_final.txt
# =====================================================================
import os, csv, collections, datetime
KF_ROOT = os.environ.get("KURDFAKE_ROOT", os.getcwd())   # data/models/results root; see README
import numpy as np
RES = os.path.join(KF_ROOT, "results"); FT = os.path.join(RES, "ft")
C_MISS, C_FA_NON, C_FA_SPF = 1.0, 10.0, 10.0
PI_TAR, PI_NON, PI_SPF = 0.90, 0.05, 0.05
NORM = min(C_MISS * PI_TAR, C_FA_NON * PI_NON + C_FA_SPF * PI_SPF)
RUNS = {"ASV + E2 (AASIST, 100 Hz HPF)": [f"AASIST_control_train-both_b100_s{s}" for s in (1337, 2026, 42)],
        "ASV + E4-ch (MMS-300M + real channel)": [f"ADF-MMS300M_E4-base_aug-ch_train-both_b100_s{s}" for s in (1337, 2026, 42)]}
GRID = 120

def adcf(pm, pfn, pfs):
    return (C_MISS * PI_TAR * pm + C_FA_NON * PI_NON * pfn + C_FA_SPF * PI_SPF * pfs) / NORM

def eer_thr(pos, neg):
    s = np.unique(np.quantile(np.concatenate([pos, neg]), np.linspace(0, 1, 2001)))
    d = [abs((pos < t).mean() - (neg >= t).mean()) for t in s]
    return s[int(np.argmin(d))]

asv = collections.defaultdict(list)
with open(os.path.join(RES, "asv_scores_with_ch.csv"), encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        lab = r["label"].strip().lower()
        r["kind"] = "spf" if lab == "spoof" else ("tar" if lab in ("target", "tar", "1", "true") else "non")
        asv[(r["dataset"], r["split"])].append(r)

def cm_scores(run):
    d = {}
    with open(os.path.join(FT, run, "scores.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d[r["utterance_id"]] = float(r["score"])
    return d

def arr(rows, cm, kind, sset):
    sel = [r for r in rows if r["kind"] == kind and (kind != "spf" or (r["system"].endswith("+ch") == (sset == "ch")))]
    if cm is not None:
        sel = [r for r in sel if r["utterance_id"] in cm]
    a = np.array([float(r["score"]) for r in sel])
    c = np.array([cm[r["utterance_id"]] for r in sel]) if cm is not None else None
    return a, c

L = [f"KurdFake final SASV (a-DCF, ASVspoof 5 parameters)  {datetime.datetime.now():%Y-%m-%d %H:%M}",
     "lower is better; 1.0 = no better than accepting or rejecting everyone. Mean over 3 seeds (min..max).", ""]
cms = {name: [cm_scores(r) for r in runs] for name, runs in RUNS.items()}
for ds in ("rail", "cv"):
    dev, test = asv[(ds, "dev")], asv[(ds, "test")]
    for sset in ("raw", "ch"):
        L.append(f"== {ds.upper()} test, {'raw spoofs' if sset == 'raw' else 'channel-matched (+ch) spoofs'}")
        at = {k: arr(test, None, k, sset)[0] for k in ("tar", "non", "spf")}
        ad = {k: arr(dev, None, k, sset)[0] for k in ("tar", "non")}
        t_asv = eer_thr(ad["tar"], ad["non"])
        pm, pfn, pfs = 1 - (at["tar"] >= t_asv).mean(), (at["non"] >= t_asv).mean(), (at["spf"] >= t_asv).mean()
        grid_a = np.quantile(np.concatenate([at["tar"], at["non"]]), np.linspace(0, 1, GRID))
        mn = min(adcf(1 - (at["tar"] >= t).mean(), (at["non"] >= t).mean(), (at["spf"] >= t).mean()) for t in grid_a)
        L.append(f"  {'ASV only':42s} act {adcf(pm, pfn, pfs):.3f}   min {mn:.3f}   (FA-spoof {pfs:.1%})")
        for name, seeds in cms.items():
            acts, mins, fas = [], [], []
            for cm in seeds:
                both = {k: arr(test, cm, k, sset) for k in ("tar", "non", "spf")}
                at2 = {k: v[0] for k, v in both.items()}; ct = {k: v[1] for k, v in both.items()}
                cd = {k: arr(dev, cm, k, sset)[1] for k in ("tar", "non", "spf")}
                t_cm = eer_thr(np.concatenate([cd["tar"], cd["non"]]), cd["spf"])
                acc = lambda ta, tc, k: (at2[k] >= ta) & (ct[k] >= tc)
                f = lambda ta, tc: (1 - acc(ta, tc, "tar").mean(), acc(ta, tc, "non").mean(), acc(ta, tc, "spf").mean())
                p = f(t_asv, t_cm); acts.append(adcf(*p)); fas.append(p[2])
                grid_c = np.append(np.quantile(np.concatenate([ct["tar"], ct["non"], ct["spf"]]), np.linspace(0, 1, GRID // 3)), -np.inf)
                mins.append(min(adcf(*f(ta, tc)) for ta in grid_a[::3] for tc in grid_c))
            L.append(f"  {name:42s} act {np.mean(acts):.3f} ({min(acts):.3f}..{max(acts):.3f})   min {np.mean(mins):.3f}   (FA-spoof {np.mean(fas):.1%})")
        L.append("")
txt = "\n".join(L); print(txt)
open(os.path.join(RES, "sasv_final.txt"), "w", encoding="utf-8").write(txt + "\n")
