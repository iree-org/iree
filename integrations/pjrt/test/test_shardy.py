# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test that JAX works with the Shardy partitioner enabled.

JAX 0.8.2+ uses Shardy by default (jax_use_shardy_partitioner=True).
This test verifies that IREE can properly handle MLIR bytecode containing
the sdy (Shardy) dialect by:
1. Registering the sdy dialect for bytecode deserialization
2. Stripping sdy ops/attributes during input conversion (single-device)
"""

import sys
import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")

# Check if Shardy is available and enabled
# JAX 0.8.2+ has Shardy enabled by default
# Older versions may not have the config option at all
shardy_available = hasattr(jax.config, 'jax_use_shardy_partitioner')
if shardy_available:
    shardy_enabled = jax.config.jax_use_shardy_partitioner
    print(f"Shardy partitioner enabled: {shardy_enabled}")
else:
    shardy_enabled = False
    print("Shardy partitioner not available in this JAX version")
    print("Skipping Shardy-specific tests (dialect registration still verified)")

# Even if Shardy is not enabled, we still test that basic compilation works.
# The Shardy dialect registration is needed for JAX 0.8.2+ bytecode.

# Simple matrix multiplication - exercises core JAX compilation path
print("Testing matrix multiplication...")
a = jnp.ones((4, 4))
b = jnp.eye(4)
c = jnp.dot(a, b)
print(f"Matrix multiply result shape: {c.shape}")
print(c)

# Test with JIT compilation
@jax.jit
def matmul_jit(x, y):
    return jnp.dot(x, y)

print("\nTesting JIT-compiled matmul...")
d = matmul_jit(a, b)
print(f"JIT result shape: {d.shape}")
print(d)

# Test with vmap (vectorized mapping)
@jax.jit
def batched_add(x, y):
    return x + y

print("\nTesting vmap...")
xs = jnp.arange(12).reshape(3, 4)
ys = jnp.ones((3, 4))
result = jax.vmap(batched_add)(xs, ys)
print(f"vmap result shape: {result.shape}")
print(result)

# Test with grad (automatic differentiation)
def simple_loss(x):
    return jnp.sum(x ** 2)

print("\nTesting grad...")
x = jnp.array([1.0, 2.0, 3.0])
grad_fn = jax.grad(simple_loss)
grad_result = grad_fn(x)
print(f"Gradient of sum(x^2) at {x}: {grad_result}")

if shardy_enabled:
    print("\nAll Shardy integration tests passed!")
else:
    print("\nBasic JAX compilation tests passed (Shardy not active in this JAX version).")
