# Third-Party Notices

This repository builds upon public medical image segmentation benchmarks,
standard segmentation backbones, and published Test-Time Adaptation (TTA)
methods for fair experimental comparison.

## Datasets
- **Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation (M&Ms)**:
  Campello et al., *IEEE Transactions on Medical Imaging*, 2021.
- **RIGA+ Multi-Center Retinal Optic Disc and Cup Benchmark**:
  Almazroa et al., *SPIE Medical Imaging*, 2018; Hu et al., *MICCAI*, 2022.
- **Multi-Site Prostate MRI Segmentation**:
  Liu et al., *IEEE Transactions on Medical Imaging*, 2020 (NCI-ISBI 2013, I2CVB, PROMISE12).
- **OPTIC Multi-Center Fundus Benchmark**:
  RIM-ONE-r3, REFUGE, ORIGA, Drishti-GS (packaged following Chen et al., *CVPR*, 2024).

## Segmentation Backbones & Comparative Baselines
- **VPTTA ResUnet-34** (`models/vptta_resunet.py`): adapted from the official
  [VPTTA](https://github.com/Chen-Ziyang/VPTTA) repository (Chen et al., *CVPR*, 2024, MIT License)
  to load official multi-center checkpoints natively.
- **SicTTA 2D U-Net** (`models/sictta_unet.py`): adapted from the official
  [SicTTA](https://github.com/HiLab-git/SicTTA) repository (*Medical Image Analysis*, 2026)
  for CPU/GPU cross-backbone evaluation without external dependencies.
- **Reference Online TTA Baselines** (`tta/baselines.py`): clean `batch_size=1` re-implementations
  following:
  - **PTBN** (Nado et al., 2020): [google/uncertainty-baselines](https://github.com/google/uncertainty-baselines)
  - **TENT** (Wang et al., *ICLR*, 2021): [DequanWang/tent](https://github.com/DequanWang/tent)
  - **CoTTA** (Wang et al., *CVPR*, 2022): [qinenergy/cotta](https://github.com/qinenergy/cotta)
  - **SAR** (Niu et al., *ICLR*, 2023): [mr-eggplant/SAR](https://github.com/mr-eggplant/SAR)
- **Additional Comparative Baselines Evaluated in the Paper** (run via their official repositories):
  - **DIGA** (Wang et al., *MICCAI*, 2023): [Waybaba/DIGA](https://github.com/Waybaba/DIGA)
  - **GraTa** (Chen et al., *ICLR*, 2025): [Chen-Ziyang/GraTa](https://github.com/Chen-Ziyang/GraTa)
  - **VPTTA** (Chen et al., *CVPR*, 2024): [Chen-Ziyang/VPTTA](https://github.com/Chen-Ziyang/VPTTA)
  - **SicTTA** (*Medical Image Analysis*, 2026): [HiLab-git/SicTTA](https://github.com/HiLab-git/SicTTA)
