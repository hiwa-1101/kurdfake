@echo off
REM KurdFake - locked rule: 3 seeds (1337, 2026, 42) for the selected model (E4-ch) and for E2.
REM E4-ch first (about 30-40 min each), then E2 (about 4.5 h each). Results: results\ft\..._s1337, _s2026, _s42
cd /d %KURDFAKE_ROOT%
for %%S in (1337 2026 42) do python train_adf.py --variant base --aug ch --seed %%S
for %%S in (1337 2026 42) do python train_cm.py --cond control --seed %%S
echo ALL SEED RUNS FINISHED
