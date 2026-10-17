#!/usr/bin/env python3
"""Synthetic-input throughput and memory benchmark for the Study 2 backbones."""

from __future__ import annotations

import argparse
import json
import time

import torch

from cytofair.backbones import ARCHITECTURES, build_classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for architecture in ARCHITECTURES:
        torch.cuda.reset_peak_memory_stats()
        model, spec = build_classifier(architecture, 5)
        model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        images = torch.randn(args.batch_size, 3, 224, 224, device=device)
        targets = torch.randint(0, 5, (args.batch_size,), device=device)
        for step in range(args.steps + 5):
            if step == 5:
                torch.cuda.synchronize()
                started = time.time()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images), targets)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.time() - started
        print(json.dumps({
            "architecture": architecture,
            "feature_dim": spec.feature_dim,
            "train_images_per_second": round(args.steps * args.batch_size / elapsed, 1),
            "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        }), flush=True)
        del model, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
