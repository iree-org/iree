#!/usr/bin/env python3
"""Generate non-zero numpy inputs for matmul benchmarks."""
import numpy as np
import os

OUT = os.path.dirname(__file__)

shapes = {
    "matmul_2048x2048x2048": {"lhs": (2048, 2048), "rhs": (2048, 2048)},
    "matmul_2048x1024x4096": {"lhs": (2048, 4096), "rhs": (4096, 1024)},
    "matmul_4096x4096x4096": {"lhs": (4096, 4096), "rhs": (4096, 4096)},
}

rng = np.random.default_rng(42)
for name, s in shapes.items():
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    lhs = rng.standard_normal(s["lhs"]).astype(np.float16)
    rhs = rng.standard_normal(s["rhs"]).astype(np.float16)
    np.save(os.path.join(d, "lhs.npy"), lhs)
    np.save(os.path.join(d, "rhs.npy"), rhs)
    print(f"{name}: lhs={lhs.shape} rhs={rhs.shape} (non-zero, seed=42)")
