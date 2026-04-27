# Pretrained weights — seed 42, $n=8$, 4000 epochs each

Two checkpoints, both DriftDiT-Small (~26.6M params), trained on CIFAR-10 from scratch with seed 42 and identical hyperparameters except the centroid estimator.  Each file is the lowest-FID snapshot recorded during training under fixed $\gamma=1.0$ evaluation.

| File | Method | FID ($\gamma=1.0$) | Epoch |
|------|--------|----------|------------|
| `seed42_n8_noabc.pt` | Standard (uncorrected $T_n$) | 7.25 | 3799 |
| `seed42_n8_abc.pt`   | ABC-corrected $T_n^{\text{ABC}}$ | 5.83 | 3399 |

## Loading

Each `.pt` is a PyTorch dict containing `model`, `ema`, `optimizer`, `scheduler`, `epoch`, `step`, `config`, `best_fid`, and `fid_results`.  For inference, prefer the EMA snapshot:

```python
import torch
from model import DriftDiT_models

ckpt = torch.load("weights/seed42_n8_abc.pt", map_location="cuda")
model = DriftDiT_models["DriftDiT-Small"](num_classes=10).cuda()
model.load_state_dict(ckpt["ema"])
model.eval()
```

`sample.py` does this automatically; just pass `--ckpt weights/seed42_n8_abc.pt`.

## Provenance

Trained on a single NVIDIA A100 80GB over ~46 hours per checkpoint.  Multi-seed numbers in the paper (Table 1) average over seeds 42, 43, 44.
