"""
Compute FID for drifting model checkpoints.
Uses InceptionV3 features + scipy for FID calculation.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torchvision.models import inception_v3, Inception_V3_Weights

from model import DriftDiT_models
from utils import set_seed


class InceptionFeatureExtractor(nn.Module):
    """Extract 2048-dim features and 1000-class logits from InceptionV3."""

    def __init__(self):
        super().__init__()
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        # Feature extractor (pool3 → 2048-dim)
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, nn.MaxPool2d(3, 2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = inception.fc  # (2048 → 1000)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        # x: (B, 3, 299, 299) in [0, 1]
        feat = self.blocks(x).flatten(1)  # (B, 2048)
        logits = self.fc(feat)            # (B, 1000)
        return feat, logits


def get_real_stats(device, cache_path="fid_cache_cifar10.npz", dataset_name="cifar10",
                    data_root="./data"):
    """Compute or load cached real image statistics."""
    cache = Path(cache_path)
    if cache.exists():
        data = np.load(cache)
        return data["mu"], data["sigma"]

    print("Computing real image statistics (cached for future runs)...")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if dataset_name == "cifar100":
        dataset = datasets.CIFAR100(data_root, train=True, download=True, transform=transform)
    else:
        dataset = datasets.CIFAR10(data_root, train=True, download=True, transform=transform)

    inception = InceptionFeatureExtractor().to(device)
    features = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    for batch_idx, (images, _) in enumerate(loader):
        images = F.interpolate(images.to(device), size=299, mode="bilinear", align_corners=False)
        feat, _ = inception(images)
        features.append(feat.cpu().numpy())
        if (batch_idx + 1) % 50 == 0:
            print(f"  Real: {(batch_idx+1)*128}/{len(dataset)}")

    features = np.concatenate(features, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)

    np.savez(cache, mu=mu, sigma=sigma)
    print(f"Cached to {cache}")
    return mu, sigma


def generate_and_get_stats(model, device, inception, num_samples=50000,
                           batch_size=256, alpha=1.0, num_classes=10):
    """Generate samples and compute inception statistics (features + logits)."""
    model.eval()
    features = []
    all_logits = []
    num_generated = 0

    while num_generated < num_samples:
        bs = min(batch_size, num_samples - num_generated)
        z = torch.randn(bs, 3, 32, 32, device=device)
        labels = torch.randint(0, num_classes, (bs,), device=device)

        with torch.no_grad():
            alpha_tensor = torch.full((bs,), alpha, device=device)
            x = model(z, labels, alpha_tensor)
            x = x.clamp(-1, 1)
            # Convert to [0, 1] for inception
            x = (x + 1) / 2
            x = F.interpolate(x, size=299, mode="bilinear", align_corners=False)
            feat, logits = inception(x)

        features.append(feat.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        num_generated += bs
        if num_generated % 5000 == 0 or num_generated >= num_samples:
            print(f"  Generated: {num_generated}/{num_samples}")

    features = np.concatenate(features, axis=0)[:num_samples]
    all_logits = np.concatenate(all_logits, axis=0)[:num_samples]
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma, all_logits


def compute_inception_score(logits, num_splits=10):
    """Compute Inception Score (IS) from logits.

    IS = exp(E_x[KL(p(y|x) || p(y))])
    Higher is better. Measures quality and diversity.
    """
    from scipy.special import softmax
    probs = softmax(logits, axis=1)
    scores = []
    split_size = len(probs) // num_splits
    for i in range(num_splits):
        part = probs[i * split_size:(i + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return float(np.mean(scores)), float(np.std(scores))


def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute FID between two Gaussian distributions."""
    from scipy import linalg

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Fixed CFG strength used for FID.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model_name = config.get("model", "DriftDiT-Small")

    num_classes = config.get("num_classes", 10)
    if args.dataset == "cifar100":
        num_classes = 100

    model_fn = DriftDiT_models[model_name]
    model = model_fn(img_size=32, in_channels=3, num_classes=num_classes).to(device)

    if "ema" in ckpt:
        print("Loading EMA weights")
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    # Inception
    inception = InceptionFeatureExtractor().to(device)

    # Real stats
    cache_name = f"fid_cache_{args.dataset}.npz"
    mu_real, sigma_real = get_real_stats(device, cache_path=cache_name, dataset_name=args.dataset)

    # Fixed-alpha evaluation
    print(f"\nEvaluating alpha={args.alpha}")
    print(f"Generating {args.num_samples} samples\n")

    mu_gen, sigma_gen, logits_gen = generate_and_get_stats(
        model, device, inception,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        alpha=args.alpha,
        num_classes=num_classes,
    )
    fid = compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)
    is_mean, is_std = compute_inception_score(logits_gen)

    print("=" * 50)
    print("Result")
    print("=" * 50)
    print(f"alpha={args.alpha:.1f}: FID={fid:.2f}, IS={is_mean:.2f}±{is_std:.2f}")


if __name__ == "__main__":
    main()
