# KurdFake

Code for **KurdFake**, a spoofing (deepfake) benchmark for Kurdish (Sorani) speaker verification, and for the
experiments of the accompanying paper:

> *KurdFake: Spoofing Attacks on Kurdish Speaker Verification and the Hidden Channel Shortcut* (under review).

The repository contains code, protocols and pseudonymised manifests only. **No audio is included.**
Common Voice audio must be obtained from Mozilla Common Voice; RAIL audio and all spoofed audio are
available to academic researchers through a controlled repository at the University of Raparin under a
data use agreement (see *Data access* below).

## What the code does

| Stage | Scripts | Output |
|---|---|---|
| 1. Corpus preparation | `prepare_rail.py`, `prepare_cv.py` | cleaned 16 kHz audio, manifests, trial lists |
| 2. Spoof generation (7 systems) | `spoof_knnvc.py` (D, D2, G), `spoof_tts_vekol.py` (C1, C1D), `spoof_f5.py` + `kf_f5.py` (B, BC) | `Dataset/Spoofs/` and spoof manifests |
| 3. Channel-matched condition (+ch) | `make_channel_matched.py` | `Dataset/Spoofs_CM/` |
| 4. ASV vulnerability (RQ1) | `asv_eval.py` | SPF-FAR of ECAPA-TDNN per system |
| 5. Off-the-shelf detectors (RQ2) | `reproduce_asvspoof2019.py`, `cm_eval.py`, `adf_eval.py` + `kf_antideepfake.py`, `sasv_eval.py` | zero-shot EER, a-DCF |
| 6. Channel shortcut analysis | `channel_test.py`, `channel_ablation.py` | transplant and ablation results |
| 7. Adaptation (RQ3) | `train_cm.py` (E1-E3, AASIST), `train_adf.py` (E4, MMS-300M-AntiDeepfake), `select_model.py` | fine-tuned detectors, dev-based model selection |
| 8. Robustness (RQ3-RQ4) | `run_seeds.bat`, `significance.py`, `sasv_final.py` | 3-seed results, paired bootstrap, final a-DCF |

Shared metrics (EER, minDCF, actDCF, Cllr; ASVspoof 5 cost parameters) are in `kf_metrics.py`.
`tools/` holds the data-inspection and listening utilities used during development.

## Directory layout and paths

All scripts read and write under one root folder, given by the environment variable `KURDFAKE_ROOT`
(default: the current working directory):

```
KURDFAKE_ROOT/
  Dataset/   RAIL_full/ (raw RAIL), RAIL_clean/, CommonVoice_ckb/, CV_ckb_clean/, Spoofs/, Spoofs_CM/, LA.zip
  models/    aasist/, knnvc/, ecapa/, vekol/, f5_ckb/, vocos/, antideepfake_mms300m/
  results/
```

```
set KURDFAKE_ROOT=D:\kurdfake          (Windows)
export KURDFAKE_ROOT=/data/kurdfake     (Linux/macOS)
```

The experiments were run on Windows 10 with Python 3.8, PyTorch 2.4.1 (CUDA 12.1) and one GPU.
Manifests written on Windows store relative paths with backslashes; on Linux, convert them with
`path.replace("\\", "/")` before use.

## External models (not included; obtain from their authors and respect their licences)

| Folder | Model | Source / licence |
|---|---|---|
| `models/aasist` | AASIST, AASIST-L (code and weights) | clovaai/aasist, MIT |
| `models/knnvc` | kNN-VC: WavLM-Large + prematched HiFi-GAN | bshall/knn-vc, MIT |
| `models/ecapa` | ECAPA-TDNN speaker embeddings | speechbrain/spkrec-ecapa-voxceleb, Apache-2.0 |
| `models/vekol` | Vekol TTS for Central Kurdish (Piper/VITS ONNX) | RevgeAI/vekol-tts-ckb-edge, CC BY-NC 4.0 |
| `models/f5_ckb`, `models/vocos` | F5-TTS Central Kurdish (studio-male) and Vocos | aranemini/central-kurdish-tts, CC BY-NC-ND 4.0; charactr/vocos-mel-24khz |
| `models/antideepfake_mms300m` | MMS-300M-AntiDeepfake | nii-yamagishilab/mms-300m-anti-deepfake, CC BY-NC-SA 4.0 |

`kf_antideepfake.py` re-implements the wav2vec 2.0 architecture in plain PyTorch so that the official
`model.safetensors` loads without fairseq; every tensor is matched by name and shape, and the output was
checked against the Hugging Face `transformers` implementation.

## Reproducing the main results

```
python prepare_rail.py && python prepare_cv.py
python spoof_knnvc.py && python spoof_knnvc.py --disjoint
python spoof_tts_vekol.py && python spoof_f5.py
python make_channel_matched.py
python asv_eval.py
python cm_eval.py && python adf_eval.py
python train_cm.py                      # E1
python train_cm.py --cond control       # E2
python train_cm.py --aug ch             # E3
python train_adf.py --variant base      # E4-base
python train_adf.py --variant proposed  # E4-proposed
python train_adf.py --variant base --aug ch   # E4-ch (post-hoc follow-up)
python select_model.py
run_seeds.bat                            # 3 seeds for E2 and E4-ch
python train_adf.py --variant base --aug ch --train cv      # cross-corpus
python train_adf.py --variant base --aug ch --budget 0.1    # data budget
python significance.py && python sasv_final.py
python channel_test.py && python channel_ablation.py
```

Every script has a header describing its inputs, outputs and options; most accept `--smoke` for a quick check.

## Data access and ethics

RAIL was recorded at the University of Raparin in 2019 with the informed (written or oral) consent of every
participant for use in scientific research. Because the consent did not cover public release, no real or
cloned RAIL audio is published. Cloned Common Voice audio is shared under the same conditions, since Common
Voice speakers did not consent to voice cloning. Academic researchers can request access to the audio from
the University of Raparin under a data use agreement that prohibits redistribution and re-identification.
Speaker-adapted voice-cloning models are not released.

## Licence

Code: MIT (see `LICENSE`). Synthetic audio and third-party models keep the licences of their sources
(non-commercial for Vekol, F5-TTS and MMS-300M-AntiDeepfake).

## Citation

To be added after publication.
