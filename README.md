# ABC: Analytical Bias Correction for Drifting Models

Minimal demo code to reproduce the CIFAR-10 results in **"Subsampling Bias in Drifting Models and Its Analytical Correction"**.

## Layout

```
.release/
├── README.md
├── requirements.txt
├── model.py             # DriftDiT (DiT backbone with adaLN-Zero)
├── drifting.py          # drifting field + ABC correction
├── feature_encoder.py   # frozen DINOv2 multi-scale encoder
├── compute_fid.py       # FID + Inception score
├── utils.py             # EMA, scheduler, sample queue, checkpoint I/O
├── reproduce_cifar10.py # training entrypoint
├── sample.py            # sampling / FID evaluation from a checkpoint
└── weights/
    ├── README.md
    ├── seed42_n8_abc.pt    # ABC-corrected, n=8, seed 42 (4000 epochs)
    └── seed42_n8_noabc.pt  # Standard (uncorrected), n=8, seed 42
```

## Setup

```bash
pip install -r requirements.txt
```

DINOv2 weights are downloaded on first run via `torch.hub` (no manual fetch needed).

> **Note on DINOv2 / `torch.hub`.** The encoder pulls the latest `facebookresearch/dinov2`
> repo from `torch.hub`. We tested with the repo state as of late 2025 (DINOv2 ViT-S/14,
> hub API `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", ...)`).
> If upstream renames or removes the entry point in the future, you may need to either
> (a) install a known-good commit manually and load it from a local checkpoint, or
> (b) replace `feature_encoder.py` with any frozen ViT-S/14-style encoder. The drifting
> loss only relies on multi-scale GAP features and is encoder-agnostic.

## Reproduce

The flag that toggles ABC is `--abc`.

```bash
# Standard (uncorrected) baseline at n=8, 4000 epochs
python reproduce_cifar10.py \
    --gpu 0 --seed 42 --n_pos 8 --n_neg 8 --epochs 4000 \
    --output_dir runs/n8_noabc_s42

# ABC-corrected at n=8, same budget
python reproduce_cifar10.py \
    --gpu 0 --seed 42 --n_pos 8 --n_neg 8 --epochs 4000 \
    --abc --output_dir runs/n8_abc_s42
```

The training log, checkpoints, and intermediate FID / generated samples are written under `--output_dir`.

We follow the protocol in the paper: total positive samples seen is held fixed across $n$ by scaling epochs inversely with $n$ (e.g., $n{=}8$ runs for 4000 epochs, $n{=}64$ for 500). To replicate any other $n$ in the paper, change `--n_pos`, `--n_neg`, and `--epochs` together (we keep `--n_neg = --n_pos = n` in all paper experiments).

## Sample from a checkpoint

```bash
python sample.py \
    --ckpt weights/seed42_n8_abc.pt \
    --num_samples 50000 \
    --output_dir samples/n8_abc_s42
```

`sample.py` uses a fixed classifier-free-guidance strength $\gamma=1.0$ by default.

## Pretrained weights

Two seed-42 checkpoints (n=8, 4000 epochs each, lowest-FID snapshot under fixed $\gamma=1.0$ evaluation) are bundled in `weights/`:

| File | Method | FID ($\gamma=1.0$) | Epoch |
|------|--------|----------|------------|
| `seed42_n8_noabc.pt` | Standard   | 7.28 | 3799 |
| `seed42_n8_abc.pt`   | ABC (ours) | 5.79 | 3399 |

Numbers are seed-42 only.  Multi-seed (42, 43, 44) results in the paper appear in Table 1.

## What ABC does

The drifting model needs the kernel-weighted centroid

$$T^* = \mathbb{E}[w \mathbf{y}] / \mathbb{E}[w]$$

over the full reference distribution.  In a minibatch of $n$ samples we have

$$T_n = \sum_i \alpha_i \mathbf{y}_i, \quad \alpha_i = w_i / \sum_j w_j,$$

which has a pointwise $O(1/n)$ bias.  ABC subtracts the in-batch plug-in of the leading bias term, yielding

$$T_n^{\text{ABC}} = (1 - \sum_i \alpha_i^2) T_n + \sum_i \alpha_i^2 \mathbf{y}_i,$$

which has $O(1/n^2)$ residual bias and provably no first-order variance inflation.

## Citation

Anonymous submission.  Citation will be added upon acceptance.
