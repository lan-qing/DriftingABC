"""
Faithful reproduction of the Drifting Model on CIFAR-10.

Follows the original drifting model paper as closely as possible:
- DriftDiT-Small generator (DiT-style ViT with adaLN-Zero)
- DINOv2 frozen feature encoder (4 separate scales)
- Multi-temperature drift field [0.02, 0.05, 0.2]
- Per-dim standardization + L2 normalization
- EMA (decay 0.999)
- CFG training with gamma ~ Uniform[1, 4]
- AdamW optimizer with linear warmup

The flag ``--abc`` toggles Analytical Bias Correction (default off).

Usage:
    python reproduce_cifar10.py --gpu 0          # single GPU
    python reproduce_cifar10.py --gpu 0 --gpu2 1 # 2 GPUs (model on gpu0, DINOv2 on gpu1)
"""

import argparse
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DriftDiT_models
from drifting import compute_V
from feature_encoder import DINOv2Encoder
from compute_fid import (
    InceptionFeatureExtractor,
    get_real_stats,
    generate_and_get_stats,
    compute_fid,
    compute_inception_score,
)
from utils import (
    EMA,
    WarmupLRScheduler,
    SampleQueue,
    save_checkpoint,
    save_image_grid,
    count_parameters,
    set_seed,
)


# ─── Feature extraction & normalization ───────────────────────────────────────

def extract_multiscale_features(x, encoder, device):
    """Extract per-scale GAP features from DINOv2."""
    feat_maps = encoder(x)
    return [F.adaptive_avg_pool2d(fm, 1).flatten(1) for fm in feat_maps]


def normalize_pair(feat_gen, feat_pos):
    """Per-dim standardization + L2 normalize."""
    all_feat = torch.cat([feat_gen.detach(), feat_pos], dim=0)
    mean = all_feat.mean(dim=0, keepdim=True)
    std = all_feat.std(dim=0, keepdim=True).clamp(min=1e-6)
    fg = F.normalize((feat_gen - mean) / std, p=2, dim=1)
    fp = F.normalize((feat_pos - mean) / std, p=2, dim=1)
    return fg, fp


# ─── Drift loss ───────────────────────────────────────────────────────────────

def compute_class_drift_loss(
    feat_gen_scales, feat_pos_scales, labels_gen, labels_pos,
    temperatures, abc=False,
):
    """
    Multi-scale, multi-temperature, class-conditional drift loss.

    For each class and each DINOv2 scale:
      1. Normalize features (per-dim std + L2)
      2. Compute multi-temperature V field
      3. MSE(features, features + V)

    Returns total loss (averaged over classes and scales).
    """
    device = feat_gen_scales[0].device
    num_scales = len(feat_gen_scales)
    num_classes = labels_gen.max().item() + 1

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_drift = 0.0
    count = 0

    for s in range(num_scales):
        for c in range(num_classes):
            mask_gen = labels_gen == c
            mask_pos = labels_pos == c
            if not mask_gen.any() or not mask_pos.any():
                continue

            fg_c = feat_gen_scales[s][mask_gen]
            fp_c = feat_pos_scales[s][mask_pos]
            fg_norm, fp_norm = normalize_pair(fg_c, fp_c)

            # Multi-temperature V
            V = torch.zeros_like(fg_norm)
            for tau in temperatures:
                V_tau = compute_V(
                    fg_norm, fp_norm, fg_norm,  # negatives = generated
                    temperature=tau, mask_self=True, abc=abc,
                )
                v_rms = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
                V = V + V_tau / (v_rms + 1e-8)

            target = (fg_norm + V).detach()
            loss_cs = F.mse_loss(fg_norm, target)
            total_loss = total_loss + loss_cs
            total_drift += (V ** 2).mean().item() ** 0.5
            count += 1

    if count > 0:
        total_loss = total_loss / count
        total_drift /= count

    return total_loss, {"loss": total_loss.item(), "drift_norm": total_drift}


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    device2 = torch.device(f"cuda:{args.gpu2}") if args.gpu2 >= 0 else device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args).copy()
    config["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Data ──
    data_root = args.data_root
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    train_dataset = datasets.CIFAR10(data_root, train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # ── Model ──
    model = DriftDiT_models["DriftDiT-Small"](
        img_size=32, in_channels=3, num_classes=10,
        label_dropout=args.label_dropout,
    ).to(device)
    n_params = count_parameters(model)
    print(f"Model: DriftDiT-Small ({n_params:,} params)")

    ema = EMA(model, decay=args.ema_decay)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    scheduler = WarmupLRScheduler(optimizer, args.warmup_steps, args.lr)

    # ── Feature encoder (frozen DINOv2) ──
    feature_encoder = DINOv2Encoder(input_size=98).to(device2)
    feature_encoder.eval()
    print(f"DINOv2 encoder on {device2} (multi-scale, 4 layers)")

    # ── Sample queue ──
    queue = SampleQueue(
        num_classes=10,
        queue_size=max(args.queue_size, args.n_pos * 2),
        sample_shape=(3, 32, 32),
    )

    # ── Resume ──
    start_epoch = 0
    global_step = 0
    best_fid = float("inf")
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            # Fix: move optimizer state tensors to correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)
        best_fid = ckpt.get("best_fid", float("inf"))
        # Restore RNG state if available
        if "rng_torch" in ckpt:
            torch.set_rng_state(ckpt["rng_torch"])
        if "rng_cuda" in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        if "rng_numpy" in ckpt:
            np.random.set_state(ckpt["rng_numpy"])
        # Restore queue if available
        if "queue" in ckpt:
            queue.queues = ckpt["queue"]["queues"]
            queue.counts = ckpt["queue"]["counts"]
            queue.indices = ckpt["queue"]["indices"]
            print(f"Restored queue: {sum(queue.counts.values())} samples")
        print(f"Resumed at epoch {start_epoch}, step {global_step}, best FID={best_fid:.2f}")

    # ── Fill queue (only if not restored from checkpoint) ──
    queue_restored = args.resume and "queue" in ckpt
    if not queue_restored:
        print("Filling sample queue...")
        for batch in train_loader:
            queue.add(batch[0], batch[1])
            if queue.is_ready(args.n_pos):
                break
    else:
        print("Queue restored from checkpoint, skipping fill.")

    # ── Training ──
    temperatures = [float(t) for t in args.temperatures]
    print(f"\n{'='*60}")
    print(f"Training: {args.epochs} epochs, lr={args.lr}, "
          f"n_pos={args.n_pos}, n_neg={args.n_neg}")
    print(f"Temperatures: {temperatures}")
    print(f"CFG alpha: [{args.alpha_min}, {args.alpha_max}]")
    print(f"EMA decay: {args.ema_decay}")
    print(f"Grad accum: {args.grad_accum_steps}")
    print(f"ABC: {args.abc}")
    print(f"{'='*60}\n")

    log_file = open(output_dir / "train.log", "a")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_drift = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            x_real, labels_real = batch[0].to(device), batch[1].to(device)
            queue.add(x_real.cpu(), labels_real.cpu())

            if not queue.is_ready(args.n_pos):
                continue

            # ── Generate samples ──
            model.train()
            batch_size = 10 * args.n_neg  # all 10 classes
            labels_list = []
            for c in range(10):
                labels_list.append(torch.full((args.n_neg,), c, device=device, dtype=torch.long))
            labels_gen = torch.cat(labels_list)

            alpha = torch.empty(batch_size, device=device).uniform_(
                args.alpha_min, args.alpha_max
            )
            noise = torch.randn(batch_size, 3, 32, 32, device=device)
            x_gen = model(noise, labels_gen, alpha)

            # ── Sample positives ──
            x_pos_list, labels_pos_list = [], []
            for c in range(10):
                x_c = queue.sample(c, args.n_pos, device)
                x_pos_list.append(x_c)
                labels_pos_list.append(torch.full((args.n_pos,), c, device=device, dtype=torch.long))
            x_pos = torch.cat(x_pos_list)
            labels_pos = torch.cat(labels_pos_list)

            # ── Extract features ──
            # Move images to encoder device
            x_gen_enc = x_gen.to(device2)
            x_pos_enc = x_pos.to(device2)

            feat_gen_scales = extract_multiscale_features(x_gen_enc, feature_encoder, device2)
            # Move gen features back to model device for loss
            feat_gen_scales = [f.to(device) for f in feat_gen_scales]

            with torch.no_grad():
                feat_pos_scales = extract_multiscale_features(x_pos_enc, feature_encoder, device2)
                feat_pos_scales = [f.to(device) for f in feat_pos_scales]

            # ── Compute loss ──
            loss, info = compute_class_drift_loss(
                feat_gen_scales, feat_pos_scales,
                labels_gen, labels_pos,
                temperatures=temperatures,
                abc=args.abc,
            )

            # ── Backward ──
            if n_batches % args.grad_accum_steps == 0:
                optimizer.zero_grad()

            (loss / args.grad_accum_steps).backward()

            if (n_batches + 1) % args.grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                optimizer.step()
                ema.update(model)
                scheduler.step()

            epoch_loss += info["loss"]
            epoch_drift += info["drift_norm"]
            n_batches += 1
            global_step += 1

            # ── Logging ──
            if global_step % args.log_interval == 0:
                lr = scheduler.get_lr()
                msg = (f"E{epoch+1}/{args.epochs} S{global_step} | "
                       f"loss={info['loss']:.4f} drift={info['drift_norm']:.4f} "
                       f"lr={lr:.6f}")
                print(msg)
                log_file.write(msg + "\n")
                log_file.flush()

            # ── Periodic samples ──
            if global_step % 500 == 0:
                path = output_dir / f"samples_step{global_step}.png"
                _generate_grid(ema.shadow, device, str(path))

        # ── End of epoch ──
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_drift = epoch_drift / max(n_batches, 1)
        msg = (f"\nEpoch {epoch+1} done in {epoch_time:.0f}s | "
               f"avg_loss={avg_loss:.4f} avg_drift={avg_drift:.4f}")
        print(msg)
        log_file.write(msg + "\n")

        # ── Save checkpoint ──
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_ep{epoch+1}.pt"
            save_checkpoint(
                str(ckpt_path), model, ema, optimizer, scheduler,
                epoch, global_step, config, queue=queue,
            )
            # Add best_fid to checkpoint
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            ckpt["best_fid"] = best_fid
            torch.save(ckpt, str(ckpt_path))
            print(f"Checkpoint saved: {ckpt_path}")

        # ── Sample grid ──
        if (epoch + 1) % args.sample_interval == 0:
            path = output_dir / f"samples_ep{epoch+1}.png"
            _generate_grid(ema.shadow, device, str(path))

        # ── FID evaluation ──
        if (epoch + 1) % args.fid_interval == 0 and (epoch + 1) >= args.fid_start_epoch:
            print(f"\nComputing FID at epoch {epoch+1}...")
            fid_results = evaluate_fid(
                ema.shadow, device,
                num_samples=args.fid_samples,
                alphas=[args.fid_alpha],
                cache_path=args.fid_cache,
                data_root=args.data_root,
            )
            r = fid_results[args.fid_alpha]
            fid_val = r["fid"]
            is_val = r["is_mean"]

            msg = (
                f"FID: {fid_val:.2f} (alpha={args.fid_alpha}) "
                f"IS: {is_val:.2f}±{r['is_std']:.2f}"
            )
            print(msg)
            log_file.write(msg + "\n")

            # Save best
            if fid_val < best_fid:
                best_fid = fid_val
                best_path = output_dir / "checkpoint_best.pt"
                save_checkpoint(
                    str(best_path), model, ema, optimizer, scheduler,
                    epoch, global_step, config, queue=queue,
                )
                ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
                ckpt["best_fid"] = best_fid
                ckpt["fid_alpha"] = args.fid_alpha
                ckpt["fid_results"] = fid_results
                torch.save(ckpt, str(best_path))
                print(f"★ New best FID: {best_fid:.2f} (saved)")

            log_file.flush()

    # ── Final save ──
    final_path = output_dir / "checkpoint_final.pt"
    save_checkpoint(
        str(final_path), model, ema, optimizer, scheduler,
        args.epochs - 1, global_step, config, queue=queue,
    )
    ckpt = torch.load(str(final_path), map_location="cpu", weights_only=False)
    ckpt["best_fid"] = best_fid
    torch.save(ckpt, str(final_path))
    print(f"\nTraining complete! Final: {final_path}, Best FID: {best_fid:.2f}")
    log_file.close()


# ─── Helpers ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate_grid(model, device, save_path, alpha=1.0, n_per_class=8):
    model.eval()
    samples = []
    for c in range(10):
        z = torch.randn(n_per_class, 3, 32, 32, device=device)
        labels = torch.full((n_per_class,), c, device=device, dtype=torch.long)
        alpha_tensor = torch.full((n_per_class,), alpha, device=device)
        x = model(z, labels, alpha_tensor)
        samples.append(x.clamp(-1, 1))
    save_image_grid(torch.cat(samples), save_path, nrow=n_per_class)


def evaluate_fid(model, device, num_samples=50000, alphas=None,
                 cache_path="./fid_cache_cifar10.npz", data_root="./data"):
    """Evaluate FID at one or more fixed alpha values."""
    if alphas is None:
        alphas = [1.0]

    inception = InceptionFeatureExtractor().to(device)
    mu_real, sigma_real = get_real_stats(device, cache_path=cache_path, data_root=data_root)

    results = {}
    for alpha in alphas:
        mu_gen, sigma_gen, logits = generate_and_get_stats(
            model, device, inception,
            num_samples=num_samples, batch_size=256,
            alpha=alpha, num_classes=10,
        )
        fid = compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)
        is_mean, is_std = compute_inception_score(logits)
        results[alpha] = {"fid": fid, "is_mean": is_mean, "is_std": is_std}

    del inception
    torch.cuda.empty_cache()
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Drifting Model CIFAR-10 Reproduction")

    # Model
    parser.add_argument("--label_dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_accum_steps", type=int, default=1)

    # Batch
    parser.add_argument("--n_pos", type=int, default=64,
                        help="Positive (real) samples per class per step")
    parser.add_argument("--n_neg", type=int, default=64,
                        help="Generated samples per class per step")
    parser.add_argument("--queue_size", type=int, default=256)

    # Drift
    parser.add_argument("--temperatures", nargs="+", type=float,
                        default=[0.02, 0.05, 0.2])
    parser.add_argument("--abc", action="store_true", default=False,
                        help="Enable Analytical Bias Correction")

    # CFG (note: ``alpha`` here is paper notation ``gamma``;
    # flag names are kept for backward compatibility)
    parser.add_argument("--alpha_min", type=float, default=1.0,
                        help="Lower bound of CFG strength gamma during training.")
    parser.add_argument("--alpha_max", type=float, default=4.0,
                        help="Upper bound of CFG strength gamma during training.")

    # Eval
    parser.add_argument("--fid_interval", type=int, default=25)
    parser.add_argument("--fid_start_epoch", type=int, default=25)
    parser.add_argument("--fid_samples", type=int, default=50000)
    parser.add_argument("--fid_alpha", type=float, default=1.0,
                        help="Fixed CFG strength gamma used when computing FID.")

    # Data + cache
    parser.add_argument("--data_root", type=str, default="./data",
                        help="CIFAR-10 download/cache directory")
    parser.add_argument("--fid_cache", type=str, default="./fid_cache_cifar10.npz",
                        help="Cached real-image FID statistics")

    # Logging
    parser.add_argument("--output_dir", type=str, default="./outputs_reproduce")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=25)
    parser.add_argument("--sample_interval", type=int, default=10)

    # System
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu2", type=int, default=-1,
                        help="Second GPU for DINOv2 encoder (-1 = same as --gpu)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
