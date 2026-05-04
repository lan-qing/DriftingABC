# Pretrained weights — seed 42, $n=8$, 4000 epochs each

> **Note for anonymous review.** Anonymous GitHub / 4open.science may only show
> Git LFS pointer files for the `.pt` checkpoints rather than the checkpoint
> contents themselves. The actual weights exceed the anonymous repository file
> size limit and should be obtained from the GitHub LFS repository or provided
> separately by the authors.

Two checkpoints, both DriftDiT-Small (~26.6M params), trained on CIFAR-10 from scratch with seed 42 and identical hyperparameters except the centroid estimator.  Each file is the lowest-FID snapshot recorded during training.

| File | Method | FID | Epoch |
|------|--------|----------|------------|
| `seed42_n8_noabc.pt` | Standard (uncorrected $T_n$) | 7.28 | 3799 |
| `seed42_n8_abc.pt`   | ABC-corrected $T_n^{\text{ABC}}$ | 5.79 | 3399 |

## Loading

Each `.pt` is a PyTorch dict containing `model`, `ema`, `config`, `best_fid`, `best_alpha`, and `epoch`.  For inference, prefer the EMA snapshot:

```python
import torch
from model import DriftDiT_models

ckpt = torch.load("weights/seed42_n8_abc.pt", map_location="cuda")
model = DriftDiT_models["DriftDiT-Small"](num_classes=10).cuda()
model.load_state_dict(ckpt["ema"])
model.eval()
```

`sample.py` does this automatically; just pass `--ckpt weights/seed42_n8_abc.pt`.
