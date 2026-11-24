# DeCoTTA: Architecture & Mathematical Specification

**DeCoTTA** (*Deconfounded Continual Test-Time Adaptation*, implemented as `PrevalenceDeconfoundedDualTTA` in `tta/pd_dual_tta.py`) performs single-sample (`batch_size=1`), backpropagation-free online Test-Time Adaptation (TTA) on frozen medical segmentation networks (`ResUNet-34`, `VPTTA ResUnet-34`, `2D U-Net`) without modifying any model parameter or BatchNorm buffer.

## Module Decomposition (Paper-to-Code Map)

| Module | Paper Component | Role |
| --- | --- | --- |
| `tta/config.py` | Universal Config | Immutable `PDDualConfig` (20 universal defaults shared across all benchmarks) and canonical single-factor ablation presets (`pd_dual_tta`, `shallow_only`, `no_rescue`, `no_prev_match`, `no_supp_only`, `no_shift_only`, `no_supp_shift_gate`, `deep_inst_only`, `weight_only_sf`). |
| `tta/masks.py` | Body/Background & Prevalence Masks | Border-connected non-anatomical padding stripping (`detect_body_mask`), forward-hook BN partition discovery (`discover_bn_partition`), zero-bleed `SpatialErosionReplay`, and foreground prevalence vector extraction (`extract_prevalence_vector`). |
| `tta/shallow_frontend.py` | **Stage 1: Shallow Background Calibration (`SBC`)** | Background-anchored shallow feature calibration (`stride <= 4`), Empirical-Bayes stream ledger with source invariance-radius shrinkage (`1.5 * max(q_med_s, 0.25)`), `q95_s` outlier fuse, and 2-step hospital-switch auto-reset. |
| `tta/deep_bank.py` | **Offline Prep & Stage 3a: Source Moment Summary ($\mathcal{S}_{\text{src}}$) & `PMA`** | Leave-one-out (LOO) calibrated `M`-nearest-neighbor prevalence-indexed source deep moment bank (`DeepSourceBank`), `.pt` source-summary serialization (zero patient pixels), and GPU residual stream state (`DeepResidualStreamState`). |
| `tta/pd_dual_tta.py` | **Unified Orchestrator (`DeCoTTA`: `SBC` + `OPG` + `PMA` + `DGS`)** | Composes Stage 1 Pass-0 (`SBC`), Stage 2 Organ Presence Gating (`OPG`) + decoder-frozen encoder rescue, and Stage 3 Prevalence Matched Anchoring (`PMA`) + Dual Gated Shrinkage (`DGS`). |

## Three-Stage Inference Pipeline

1. **Stage 1 — Background-Anchored Shallow Calibration (`stride <= 4`)**
   - Strips border-connected padding $\Omega_{\text{pad}}$ (MRI flat borders and RGB fundus dark camera rims) and excludes dilated coarse foreground inside $\Omega_{\text{body}}$ (with automatic fallback to clean $\Omega_{\text{body}}$ when coarse foreground exceeds `35%`).
   - Estimates per-layer background $(\mu, \log\sigma)$ shifts via an Empirical-Bayes stream ledger with 2-step jump auto-reset and applies feature rescaling strictly inside $\Omega_{\text{body}}$ during Pass-0, recording exact per-layer affine ops.

2. **Stage 2 — Anatomical Presence Gate & Conservative Dim-Foreground Rescue**
   - Computes the Pass-0 foreground area vector $\hat{\pi}_t \in [0, 1]^K$ and total foreground fraction $r_{\text{fg}} = \sum_k \hat{\pi}_{t,k}$ inside $\Omega_{\text{body}}$.
   - When $r_{\text{fg}} < \tau_{\text{pres}}$ (`0.002`), runs a conservative **encoder-only** moment-blended probe (keeping all decoder BNs strictly in `eval()` mode so empty slices cannot hallucinate). If rescued foreground remains below $\tau_{\text{rescue}}$ (`0.005`), Stage 3 is short-circuited and Pass-0 logits are returned unchanged.

3. **Stage 3 — Prevalence-Deconfounded Deep Feature Transport (`stride > 4` & Decoder)**
   - Maps $\hat{\psi}_t = \sqrt{\hat{\pi}_t}$ into the standardized source foreground prevalence space and queries the $M$ nearest source foreground neighbors via a Gaussian kernel to obtain prevalence-matched reference moments $(\hat{\mu}_{s,l}(\hat{\pi}_t), \hat{\sigma}^2_{s,l}(\hat{\pi}_t))$.
   - Computes two smooth shrinkage gates calibrated strictly from LOO source statistics:
     - **Anatomical Support Gate** $w_{\text{supp}}(t) = \exp\big(-\gamma_{\text{supp}}\max(0, (\bar{d}_M(t)/d_{s,0.90})^2 - 1)\big)$ protecting out-of-support camera/field-of-view shifts;
     - **Residual Shift Gate** $w_{\text{shift}}(t, l)$ shrinking sub-noise deviations below the source LOO conditional variance $(\tau_{\mu,l}^2, \tau_{\eta,l}^2)$.
   - Applies exact non-mutating pre-hook affine transport before frozen `eval()` deep BatchNorm layers. Under `deep_mode="inst"`, the blended target variance uses the exact TTN two-component mixture variance $v_{\text{mix}} = (1 - \lambda) v_s + \lambda v_{\text{cur}} + \lambda(1 - \lambda)(\Delta\mu)^2$, keeping the between-component cross-term rather than linearly interpolating variances.

## Compatibility Decisions

- **Legacy preset aliases (`tta/config.py` — `PRESET_ALIASES`)**: Legacy preset identifiers (`v2_full`, `v1_shallow_only`, `abl_*`, `v2_no_*`) are retained as aliases for `CANONICAL_PRESETS` strictly to maintain backward compatibility with archived JSON and CSV benchmark table keys.
