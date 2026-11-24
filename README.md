# DeCoTTA: Deconfounded Continual Test-Time Adaptation for Medical Image Segmentation

Official PyTorch implementation of **DeCoTTA** (*Deconfounded Continual Test-Time Adaptation*) — a unified, backpropagation-free, single-sample (`batch_size=1`) continual test-time adaptation framework for 2D and 3D medical image segmentation.

## Key Capabilities

- **Backpropagation-Free & Bitwise Parameter-Preserving**: Zero gradient updates, zero trainable parameters, and zero modification to pretrained weights or BatchNorm buffers (`334–490 MiB` peak GPU VRAM).
- **Empty-Slice Safe on 3D Volumes (`OPG`)**: Organ Presence Gating with a decoder-frozen deep encoder rescue probe prevents the `91%–100%` empty-slice false-positive collapse suffered by standard BN/entropy TTA methods on 3D MRI slice streams.
- **Prevalence-Deconfounded Deep Calibration (`PMA` + `DGS`)**: Decouples anatomical organ-size / field-of-view variation from true domain shift via leave-one-out `M`-NN source prevalence matching, anatomical support gating, and dual residual shift shrinkage.
- **Universal Single Configuration**: Identical code path and one fixed 20-field `PDDualConfig()` across MRI and RGB fundus, Softmax and overlapping multi-label Sigmoid heads, and three segmentation backbones (`ResUNet-34`, `VPTTA ResUnet-34`, `2D U-Net`).
- **Privacy-Preserving Source Moment Summary ($\mathcal{S}_{\text{src}}$)**: One-time source calibration exports a compact channel-moment summary (`.pt`, `3.7–42.0 MB` across all 9 source domains) containing **zero raw patient images or masks**.

---

## Paper-to-Code Terminology Map

Core Python classes and JSON/CSV preset keys are kept frozen for exact bitwise reproducibility across benchmark scripts and archived evaluation tables:

| Paper Term | Symbol / Acronym | Code Class / Preset Key |
| :--- | :--- | :--- |
| **Deconfounded Continual Test-Time Adaptation** | `DeCoTTA` | `PrevalenceDeconfoundedDualTTA` / preset `pd_dual_tta` (alias `v2_full`) |
| **Source Moment Summary** | $\mathcal{S}_{\text{src}}$ | `DeepSourceBank` (`.pt` summary via `save_source_summary` / `from_source_summary`) |
| **Shallow Background Calibration** | `SBC` | `ShallowBackgroundStreamTTA` (ablation preset `shallow_only`) |
| **Organ Presence Gating** | `OPG` | `tau_pres` + `rescue_mode="enc_inst"` (ablation preset `no_rescue`) |
| **Prevalence Matched Anchoring** | `PMA` | `use_prev_match=True` (ablation preset `no_prev_match`) |
| **Dual Gated Shrinkage** | `DGS` | `use_supp_gate` + `use_shift_gate` (ablation presets `no_supp_only`, `no_shift_only`, `no_supp_shift_gate`) |

---

## Repository Structure

```text
.
├── tta/                          # Core DeCoTTA library
│   ├── config.py                 # Universal PDDualConfig & single-factor ablation presets
│   ├── masks.py                  # Padding stripping, BN partition discovery & erosion replay
│   ├── shallow_frontend.py       # Stage 1 (SBC): Background-anchored shallow calibration
│   ├── deep_bank.py              # Stage 3a (PMA): LOO prevalence-indexed source moment bank (.pt I/O)
│   ├── pd_dual_tta.py            # Unified 3-stage DeCoTTA operator (SBC + OPG + PMA + DGS)
│   └── baselines.py              # Reference bs=1 online TTA baselines (SourceOnly/PTBN/TENT/SAR/CoTTA)
├── models/                       # Standalone segmentation backbones (ResUNet, VPTTAResUNet, SicTTAUNet, UNet)
├── data/                         # Dataset preprocessing & stream loaders (Prostate, RIGA+, M&Ms, OPTIC)
├── evaluate/                     # 3D/2D Dice, exact Euclidean HD95/ASSD, paired stats & CLI runner
├── train/                        # Supervised source-domain pretraining
├── scripts/                      # Full reproduction, source-bank export & robustness/latency scripts
├── docs/                         # Mathematical architecture & benchmark protocol specifications
└── tests/                        # Deterministic CPU unit test suite
```

---

## Installation (Conda)

All paper experiments were verified on **Python 3.12** with **PyTorch 2.7.1 (CUDA 12.6)**.

### 1. Create and Activate the Conda Environment

```bash
conda env create -f environment.yml
conda activate decotta
```

If you need a specific CUDA build of PyTorch matching your local driver (for example, CUDA 12.6 as used in our reference runtime):

```bash
pip install --force-reinstall torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

*(Optional)* Install plotting utilities used for figure generation and diagnostic scripts:

```bash
pip install -r requirements-figures.txt
```

### 2. Quick CPU Verification

Run the deterministic CPU unit test suite (~4 s, requires no datasets or pretrained checkpoints):

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

---

## Pretrained Checkpoints & Source Moment Summaries

Frozen source-domain checkpoints (`decotta_checkpoints_v1.tar.gz`, `658 MB` compressed / `711 MB` extracted), privacy-preserving `.pt` source moment summaries (`decotta_source_summaries_v1.tar.gz`, `177 MB` compressed / `193 MB` extracted, zero patient pixels), and audited paper result tables (`decotta_paper_results_v1.tar.gz`, `9.5 MB`) are hosted on [Google Drive](https://drive.google.com/drive/folders/1FoRXb2gTIjM_01mEafTb6qxVyq5whZWB?usp=sharing).

### 1. Download & Extract via `gdown`

```bash
pip install gdown

# 1) Frozen source-domain checkpoints (658 MB)
gdown --id 1rqae30HOU3Hv0BednoHysWcMp_Z6lEGT -O decotta_checkpoints_v1.tar.gz
echo "9ad0dd66e35c76a387b202abf7d6a8edd84035c4a5af38d316f977f0f17d1102  decotta_checkpoints_v1.tar.gz" | sha256sum -c -
tar -xzf decotta_checkpoints_v1.tar.gz

# 2) (Optional) Pre-exported privacy-preserving .pt source moment summaries (177 MB)
gdown --id 1G2pIUq-S7f-fxMuRHGI_atF6BVGrDmIE -O decotta_source_summaries_v1.tar.gz
echo "009aaf496d568124830b3d91ac4704b68e46a8ae60124205b1df5c5291ac315a  decotta_source_summaries_v1.tar.gz" | sha256sum -c -
tar -xzf decotta_source_summaries_v1.tar.gz

# 3) (Optional) Audited paper tables and per-case CSV archives (9.5 MB)
gdown --id 1szlshmES57iSg7-oCV1s0krpBb-zIkpc -O decotta_paper_results_v1.tar.gz
echo "54ddd446ccae35039ab60035b1aa263d9d91aea90a0926e26b333d3271600661  decotta_paper_results_v1.tar.gz" | sha256sum -c -
tar -xzf decotta_paper_results_v1.tar.gz
```

### 2. Expected Checkpoint Layout

After extraction, `checkpoints/` should match the layout below (or point `PD_TTA_CKPT_DIR` / `--ckpt_dir` to a custom directory):

```text
checkpoints/
├── source_prostate_runmc_seed42.pth        # Prostate (RUNMC, VPTTAResUNet, 87 MB)
├── source_rigaplus_resunet_seed42.pth      # RIGA+ (Magrabia, ResUNet-34, 94 MB)
├── source_mms_resunet_seed42.pth           # M&Ms (Vendor B, ResUNet-34, 94 MB)
├── source_mms2d_unet_A.pth                 # M&Ms Cross-Backbone (Vendor A, SicTTAUNet, 7 MB)
└── optic/                                  # OPTIC 5-center models (VPTTAResUNet, MIT, 432 MB total)
    ├── Drishti_GS/last-Res_Unet.pth
    ├── ORIGA/last-Res_Unet.pth
    ├── REFUGE/last-Res_Unet.pth
    ├── REFUGE_Valid/last-Res_Unet.pth
    └── RIM_ONE_r3/last-Res_Unet.pth
```

> **Note on Datasets**: Due to original clinical dataset licenses, raw and preprocessed medical images are not redistributed on Google Drive. Follow [docs/BENCHMARKS_AND_PROTOCOL.md](docs/BENCHMARKS_AND_PROTOCOL.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) to obtain each benchmark from its official source and preprocess it locally.

---

## Reproducing the 5-Benchmark Results & Ablations

Prepare datasets and checkpoints following [docs/BENCHMARKS_AND_PROTOCOL.md](docs/BENCHMARKS_AND_PROTOCOL.md), then run:

```bash
# Evaluate main DeCoTTA on a single suite or all 5 suites:
OMP_NUM_THREADS=1 python -m evaluate.run_benchmark \
  --suite all --presets pd_dual_tta --out_dir results/main_5benchmarks --device cuda:0

# Reproduce main DeCoTTA + all 8 single-factor decoupled ablations:
bash scripts/reproduce_main_and_ablations.sh all

# Export a standalone privacy-preserving .pt source moment summary:
OMP_NUM_THREADS=1 python scripts/export_source_bank.py \
  --suite prostate --output source_summaries/prostate_runmc_summary.pt --device cuda:0

# Run continual un-reset streams, 5-seed order shuffling & isolated GPU latency/VRAM:
OMP_NUM_THREADS=1 python scripts/run_robustness_and_efficiency.py \
  --mode both --out_json results/robustness_and_latency.json --device cuda:0
```

### Summary across 5 Benchmarks (10 Methods, Identical Default `PDDualConfig()`)

| Method | Backprop-Free | Prostate 3D Dice (Empty FPR) | RIGA+ Dice | M&Ms (`ResUNet`) 3D Dice | OPTIC (20 Streams) Dice | M&Ms (`2D UNet`) 3D Dice | **5-Suite Mean Dice / HD95** |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Source Only` | Yes | 73.17 (20.5%) | 83.70 | 59.79 | 68.47 | 76.07 | 72.24 / 30.79 |
| `PTBN` | Yes | 54.63 (100.0%) | 82.33 | 61.54 | 72.69 | 46.82 | 63.60 / 88.96 |
| `TENT` | No | 57.75 (100.0%) | 82.54 | 61.52 | 69.39 | 53.17 | 64.87 / 79.96 |
| `SAR` | No | 54.66 (100.0%) | 82.33 | 61.54 | 72.59 | 46.86 | 63.60 / 88.82 |
| `CoTTA` | No | 54.63 (100.0%) | 82.33 | 61.81 | 72.73 | 46.82 | 63.66 / 88.65 |
| `GraTa` | No | 54.07 (100.0%) | 82.39 | 61.71 | **74.35** | 48.47 | 64.20 / 87.67 |
| `VPTTA` | No | 77.26 (32.3%) | 82.47 | 63.03 | 73.23 | 77.21 | 74.64 / 36.51 |
| `SicTTA` | Yes | 73.87 (34.4%) | 74.51 | 59.37 | 71.85 | **78.68** | 71.66 / 41.52 |
| `DIGA` | Yes | 76.38 (25.2%) | 83.25 | 65.48 | 71.39 | 76.29 | 74.56 / 30.15 |
| **`DeCoTTA` (Ours)** | **Yes** | **77.46 (17.6%)** | **85.04** | **68.36** | 72.44 | 78.04 | **76.27 / 28.95** |

---

## License

Released under the [MIT License](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dataset and baseline attributions.
