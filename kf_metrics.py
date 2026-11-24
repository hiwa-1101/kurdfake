# KurdFake - shared evaluation metrics (CM side). Python 3.8 compatible, numpy only.
# Convention: higher score = more bona fide. "bona" = bona fide scores, "spoof" = spoof scores.
import numpy as np

def eer(bona, spoof):
    """Equal error rate and the threshold where it occurs."""
    bona, spoof = np.asarray(bona, float), np.asarray(spoof, float)
    s = np.concatenate([bona, spoof]); y = np.concatenate([np.ones(len(bona)), np.zeros(len(spoof))])
    o = np.argsort(s, kind="mergesort"); s, y = s[o], y[o]
    frr = np.cumsum(y) / y.sum()                       # bona fide at/below threshold are rejected
    far = 1 - np.cumsum(1 - y) / (1 - y).sum()         # spoofs above threshold are accepted
    i = np.argmin(np.abs(frr - far))
    return float((frr[i] + far[i]) / 2), float(s[i])

def rates_at(bona, spoof, thr):
    """Miss (bona fide rejected) and false-alarm (spoof accepted) rates at a fixed threshold."""
    return float((np.asarray(bona) < thr).mean()), float((np.asarray(spoof) >= thr).mean())

# ASVspoof 5 CM cost parameters (Wang et al., ASVspoof 5 evaluation plan)
C_MISS, C_FA, P_SPOOF = 1.0, 10.0, 0.05

def dcf_at(bona, spoof, thr, c_miss=C_MISS, c_fa=C_FA, p_spoof=P_SPOOF):
    pm, pf = rates_at(bona, spoof, thr)
    dcf = c_miss * (1 - p_spoof) * pm + c_fa * p_spoof * pf
    return float(dcf / min(c_miss * (1 - p_spoof), c_fa * p_spoof))

def min_dcf(bona, spoof, **kw):
    """Normalised minimum DCF over all thresholds; returns (minDCF, threshold)."""
    cand = np.unique(np.concatenate([bona, spoof, [np.inf]]))
    vals = [dcf_at(bona, spoof, t, **kw) for t in cand]
    i = int(np.argmin(vals)); return float(vals[i]), float(cand[i])

def cllr(bona_llr, spoof_llr):
    """Log-likelihood-ratio cost (bits). Inputs must be LLRs (natural log), e.g. after calibration."""
    b, s = np.asarray(bona_llr, float), np.asarray(spoof_llr, float)
    return float(0.5 * (np.mean(np.logaddexp(0, -b)) + np.mean(np.logaddexp(0, s))) / np.log(2))

def fit_calibration(bona, spoof, iters=500, lr=0.1):
    """Linear logistic calibration a*s+b -> LLR, fitted on dev scores (prior-weighted 0.5/0.5)."""
    s = np.concatenate([bona, spoof]).astype(float); y = np.concatenate([np.ones(len(bona)), np.zeros(len(spoof))])
    w = np.where(y == 1, 0.5 / len(bona), 0.5 / len(spoof))
    mu, sd = s.mean(), s.std() + 1e-9; z = (s - mu) / sd
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * z + b))); g = (p - y) * w
        a -= lr * 10 * np.sum(g * z); b -= lr * 10 * np.sum(g)
    return lambda x: a * (np.asarray(x, float) - mu) / sd + b

def report(bona_dev, spoof_dev, bona_test, spoof_test):
    """Full ASVspoof-5-style report: EER, minDCF, actDCF and FRR/FAR at the dev threshold, Cllr / minCllr proxy."""
    e, _ = eer(bona_test, spoof_test)
    md, _ = min_dcf(bona_test, spoof_test)
    _, thr_dev = eer(bona_dev, spoof_dev)
    frr, far = rates_at(bona_test, spoof_test, thr_dev)
    cal = fit_calibration(np.asarray(bona_dev), np.asarray(spoof_dev))
    c = cllr(cal(bona_test), cal(spoof_test))
    # actual DCF: Bayes threshold on calibrated LLRs
    t_bayes = np.log((C_FA * P_SPOOF) / (C_MISS * (1 - P_SPOOF)))
    lb, ls = cal(bona_test), cal(spoof_test)
    adcf = dcf_at(lb, ls, t_bayes)
    return {"eer": e, "min_dcf": md, "act_dcf": adcf, "cllr": c, "frr_at_dev_thr": frr, "far_at_dev_thr": far}
