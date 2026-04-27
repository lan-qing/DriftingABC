"""
Sample from a DriftDiT checkpoint and report FID at a fixed CFG strength.
Saves a class-conditional grid of generated images for inspection.
"""

import argparse
from pathlib import Path

import torch

from model import DriftDiT_models
from utils import save_image_grid, set_seed
from compute_fid import (
    InceptionFeatureExtractor,
    get_real_stats,
    generate_and_get_stats,
    compute_fid,
    compute_inception_score,
)


@torch.no_grad()
def class_grid(model, device, gamma=1.0, n_per_class=8, num_classes=10):
    """Generate ``n_per_class`` images for each class using CFG strength ``gamma``."""
    images = []
    for c in range(num_classes):
        z = torch.randn(n_per_class, 3, 32, 32, device=device)
        labels = torch.full((n_per_class,), c, device=device, dtype=torch.long)
        alpha = torch.full((n_per_class,), gamma, device=device)
        x = model(z, labels, alpha)
        images.append(x.clamp(-1, 1))
    return torch.cat(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to a DriftDiT checkpoint (.pt).")
    parser.add_argument("--output_dir", type=str, default="./samples")
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of samples used for FID.")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Fixed CFG strength used for sampling and FID.")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--fid_cache", type=str,
                        default="./fid_cache_cifar10.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = DriftDiT_models["DriftDiT-Small"](
        img_size=32, in_channels=3, num_classes=10,
    ).to(device)
    if "ema" in ckpt:
        print("Loading EMA weights")
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    # Class-conditional grid for visual inspection.
    grid = class_grid(model, device, gamma=args.gamma)
    grid_path = output_dir / f"class_grid_gamma{args.gamma:g}.png"
    save_image_grid(grid, str(grid_path), nrow=8)
    print(f"Saved class grid to {grid_path}")

    # FID at fixed CFG strength.
    inception = InceptionFeatureExtractor().to(device)
    mu_real, sigma_real = get_real_stats(device, cache_path=args.fid_cache,
                                         data_root=args.data_root)

    print(f"\nEvaluating gamma={args.gamma}, {args.num_samples} samples")
    mu_gen, sigma_gen, logits = generate_and_get_stats(
        model, device, inception,
        num_samples=args.num_samples, batch_size=256,
        alpha=args.gamma, num_classes=10,
    )
    fid = compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)
    is_mean, is_std = compute_inception_score(logits)
    print(f"gamma={args.gamma}: FID={fid:.3f}  IS={is_mean:.2f}±{is_std:.2f}")
    with open(output_dir / "fid_results.txt", "w") as f:
        f.write(f"gamma={args.gamma}\tFID={fid:.4f}\tIS={is_mean:.4f}\u00b1{is_std:.4f}\n")


if __name__ == "__main__":
    main()
