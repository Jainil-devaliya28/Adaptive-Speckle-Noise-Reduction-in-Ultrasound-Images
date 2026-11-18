# Adaptive Speckle Noise Reduction in Ultrasound Images

Course project, Biomedical Ultrasound. Python implementation of 10 classical
and state-of-the-art despeckling filters plus a proposed adaptive method,
**AS-NLM (Adaptive Speckle-statistics Non-Local Means)**, evaluated on
physics-based simulations and on the public BUSI breast ultrasound dataset.

## Layout

```
speckle/
  simulate.py   physics-based B-mode simulator (scatterers * PSF -> envelope -> log compression)
                with a speckle-free ground truth, plus synthetic Loupas/Gamma speckle
  filters.py    Median, Lee, Kuan, Frost, Perona-Malik, SRAD, DPAD, TV, Bilateral,
                OBNLM, and the proposed AS-NLM
  metrics.py    PSNR, SSIM, Pratt FOM, EPI, SNR/ENL, CNR, gCNR, ROI helpers
experiments/
  common.py            parameter grids, tuned parameters, BUSI loader
  01_tune.py           fair grid search of every filter on a held-out tuning set
  02_simulation.py     main benchmark on 12 held-out phantoms
  03_robustness.py     fixed parameters under changed frequency, dynamic range, SNR
  04_ablation.py       contribution of each AS-NLM stage
  05_busi_real.py      real clinical images: CNR, gCNR, SNR, boundary contrast, Dice
  06_busi_synthetic.py BUSI + added synthetic speckle at 3 noise levels
  07_figures.py        all report figures
  08_posthoc_window.py post-hoc AS-NLM-c (consistent noise window) analysis
results/               CSV tables, tuned_params.json, figures/
```

## Setup

```bash
pip install numpy scipy scikit-image matplotlib pandas
```

BUSI dataset (Al-Dhabyani et al., Data in Brief 2020): download `Dataset_BUSI_with_GT`
and point `BUSI_DIR` at the folder containing `benign/`, `malignant/`, `normal/`:

```bash
export BUSI_DIR=/path/to/Dataset_BUSI_with_GT
```

## Run

```bash
cd experiments
python 01_tune.py          # ~6 min on 2 cores
python 02_simulation.py    # ~3 min
python 03_robustness.py
python 04_ablation.py
python 05_busi_real.py     # N_PER_CLASS=50 by default
python 06_busi_synthetic.py
python 07_figures.py
python 08_posthoc_window.py   # post-hoc variant, reported separately
```

## Use a filter on your own image

```python
from skimage import io
from speckle.filters import asnlm
img = io.imread("scan.png", as_gray=True).astype(float)   # values in [0, 1]
out = asnlm(img)                                            # proposed method
out, maps = asnlm(img, return_maps=True)                    # + structure map, beta(x), sigma^2
```

## The proposed method in one paragraph

Fully developed speckle obeys a known statistical law: in the displayed image,
local variance is proportional to local mean (Loupas model). AS-NLM tests that
law at every pixel. Where the statistics say "only speckle here", it averages
aggressively (strong non-local means, then linear diffusion); where they say
"real structure here", it averages gently and diffuses only along boundaries.
A second test on the first-pass estimate catches low-contrast boundaries that
the raw image hid. Every threshold is relative to the estimated speckle level,
so the same parameters work across probes, gains and dynamic ranges.

## Headline results

| | AS-NLM | Best baseline |
| --- | --- | --- |
| Simulation PSNR / SSIM / EPI (12 phantoms) | 27.14 dB / 0.840 / 0.525 | 26.54 dB (Median) / 0.834 (TV) / 0.512 (TV) |
| Simulation Pratt FOM | 0.874 | 0.879 (Perona-Malik) |
| Worst-case SSIM over 11 acquisition changes | 0.743 | 0.716 (DPAD) |
| BUSI gCNR (100 images) | 0.696 (3rd) | 0.714 (SRAD) |

See the report for the full analysis, including where the method falls short on clinical images.
