# Benchmarks & Zero-Leakage Evaluation Protocol

All five benchmark suites are evaluated at `batch_size=1`, single view (`1-view`), `num_workers=0`, using **100% identical `PDDualConfig()` hyperparameters** and **zero per-dataset conditional branches**.

## 1. Suite Specifications & Expected File Layout

Place preprocessed datasets under `data/processed/` (or `data/raw/`) and frozen source checkpoints under `checkpoints/` (override via `--processed_root`, `--raw_root`, `--ckpt_dir`, or env vars `PD_TTA_PROCESSED_DIR`, `PD_TTA_RAW_DIR`, `PD_TTA_CKPT_DIR`):

| Suite | Modality & Task | Source Domain | Target Domains (Slices / Images) | Backbone & Output Mode | Checkpoint Path |
| --- | --- | --- | --- | --- | --- |
| `prostate` | 3D Prostate T2 MRI (binary, `56.3%` empty slices) | `RUNMC` (578 slices) | `BMC, I2CVB, UCL, BIDMC, HK` (`3,309` slices / `86` vols) | `VPTTAResUNet` (`1->2`, Softmax) | `checkpoints/source_prostate_runmc_seed42.pth` |
| `rigaplus` | 2D Retinal Fundus (Disc & Cup) | `Magrabia` (`train`) | `BinRushed, MESSIDOR` (`131` test images) | `ResUNet` (`3->3`, Softmax) | `checkpoints/source_rigaplus_resunet_seed42.pth` |
| `mms` | 3D Cardiac Cine MRI (`LV, MYO, RV`) | Vendor `B` (`train_labeled`) | Vendors `A, C, D` (`3,240` slices / `390` vols) | `ResUNet` (`1->4`, Softmax) | `checkpoints/source_mms_resunet_seed42.pth` |
| `mms2d_paper` | 3D Cardiac Cine MRI (Cross-Backbone) | Vendor `A` (`train.csv`) | Vendors `B, C, D` (`4,091` slices / `500` vols) | `SicTTAUNet` (`1->4`, Softmax) | `checkpoints/source_mms2d_unet_A.pth` |
| `optic` | 2D Retinal Fundus (`5x4 = 20` streams) | Each of `5` centers (`train`) | Remaining `4` centers (`8,440` total evals) | `VPTTAResUNet` (`3->2`, Sigmoid) | `checkpoints/optic/<domain>/last-Res_Unet.pth` |

## 2. Protocol Invariants

1. **Source-Only Calibration**: `PrevalenceDeconfoundedDualTTA.fit_from_source` accesses strictly the source training split (or loads a pre-exported `.pt` summary table via `from_source_summary`, containing zero raw images).
2. **Bitwise Frozen Backbone**: `model_state_sha256(model)` is asserted identical before and after calibration and after every target stream.
3. **Stream Isolation vs. Continual Mode**: Standard multi-domain evaluation calls `adapter.reset_stream()` at the start of each target hospital stream. Continual multi-hospital evaluation (`scripts/run_robustness_and_efficiency.py --mode robustness`) concatenates all target hospitals without calling `reset_stream()`, relying on Stage 1's 2-step jump detector for automatic domain-switch resets.
4. **Metrics**:
   - **3D MRI (`prostate`, `mms`, `mms2d_paper`)**: Patient/volume-level 3D Dice (`%`), 3D Euclidean `HD95` and `ASSD` (`voxels`), and Empty-Slice / Absent-RV False-Positive Rate (`pred_px > 0` on ground-truth empty organ slices).
   - **2D Fundus (`rigaplus`, `optic`)**: Domain-equal Whole-Disc & Optic-Cup Mean Dice (`%`) and 2D Euclidean `HD95` and `ASSD` (`pixels`).

## 3. Dataset Acquisition & Preprocessing

Due to original dataset licensing and clinical data-use agreements, raw and preprocessed medical images are **not** redistributed with this repository. Obtain the public datasets from their official sources (see [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)) and run the corresponding preprocessing scripts:

- **Prostate (Multi-Site T2 MRI)**: `python data/prepare_prostate.py --help`
- **RIGA+ (Multi-Center Retinal Fundus)**: `python data/prepare_rigaplus.py --help`
- **M&Ms (Cardiac Cine MRI)**: `python data/prepare_mms.py --help`
- **M&Ms 2D Paper (`mms2d_paper`) & OPTIC**: place under `data/raw/mms2d_paper/` and `data/raw/optic/` following the standard directory structures of SicTTA and VPTTA respectively.

For source-free deployment without storing or transferring source patient images, export or download the pre-exported `.pt` source moment summaries ($\mathcal{S}_{\text{src}}$) via `scripts/export_source_bank.py`:

```bash
OMP_NUM_THREADS=1 python scripts/export_source_bank.py \
  --suite prostate --output source_summaries/prostate_runmc_summary.pt --device cuda:0
```
