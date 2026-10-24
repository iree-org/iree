# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Scaled dot-product attention through torchax as a StableHLO composite.

`install(env)` overrides torch's scaled_dot_product_attention on a torchax
environment. It emits a `stablehlo.composite` named
`torch.scaled_dot_product_attention`, version 1. IREE lowers it to
`iree_linalg_ext.attention` without materialising the (L, S) scores, except for
grouped-query attention over run-time or indivisible head counts, causal
attention with a mask,
masks shaped other than (L, S) or (B..., L, S) with each batch dimension equal
or 1, a run-time head size without a scale, and element types wider than 32
bits. Those, and other consumers, take
the decomposition, which is torchax's reference arithmetic.
Dropout keeps torchax's own lowering.

Exports through `iree_torchax` apply the override in their own
environment; `install(env)` applies it to an eager environment.

Known limitation: query-dependent masks do not suppress nonfinite values in V
before the final contraction. Uniform key exclusions do suppress them. An
unmasked row of -inf scores also retains the contraction's floating-point
behavior.
"""

import inspect
import math

import jax
import jax.numpy as jnp
import torch
import torchax
from torchax.ops import jtorch

from iree_torchax import _torchax_registry
from iree_torchax._torchax_composite import composite

__all__ = ["install"]


def _sdpa_impl(query, key, value, *mask, is_causal, enable_gqa, scale=None):
    length, source = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    bias = jnp.zeros((length, source), query.dtype)
    if is_causal:
        keep = jnp.tril(jnp.ones((length, source), jnp.bool_))
        bias = jnp.where(keep, bias, -jnp.inf)
    if mask:
        if mask[0].dtype == jnp.bool_:
            bias = jnp.where(mask[0], bias, -jnp.inf)
        else:
            bias = bias + mask[0]
    if enable_gqa:
        groups = query.shape[-3] // key.shape[-3]
        key = jnp.repeat(key, groups, axis=-3)
        value = jnp.repeat(value, groups, axis=-3)
    weight = query @ jnp.swapaxes(key, -2, -1) * scale_factor + bias
    # As torch's _safe_softmax: a row whose scores are all -inf attends nothing.
    empty = jnp.all(weight == -jnp.inf, axis=-1, keepdims=True)
    return jnp.where(empty, 0, jax.nn.softmax(weight, axis=-1)) @ value


_sdpa = composite(_sdpa_impl, name="torch.scaled_dot_product_attention", version=1)


def _jax(value):
    return value.jax() if isinstance(value, torchax.tensor.Tensor) else value


def _scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    *,
    env,
):
    if dropout_p:
        return jtorch.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            env=env,
        )
    operands = [_jax(query), _jax(key), _jax(value)]
    if attn_mask is not None:
        operands.append(_jax(attn_mask))
    attrs: dict[str, object] = {
        "is_causal": bool(is_causal),
        "enable_gqa": bool(enable_gqa),
    }
    if scale is not None:
        attrs["scale"] = float(scale)
    return torchax.tensor.Tensor(_sdpa(*operands, **attrs), env)


def _export_scaled_dot_product_attention(*args, env, **kwargs):
    """An exported function has no runtime RNG contract to draw dropout from."""
    bound = inspect.signature(_scaled_dot_product_attention).bind(
        *args, env=env, **kwargs
    )
    if bound.arguments.get("dropout_p", 0.0):
        raise NotImplementedError(
            "scaled_dot_product_attention with dropout_p > 0 cannot be "
            "exported: stochastic export is unsupported without a runtime RNG "
            "contract"
        )
    return _scaled_dot_product_attention(*args, env=env, **kwargs)


def entries():
    """The registrations of this family, for `_torchax_registry`, as (key,
    handler, is_function, export_handler)."""
    return [
        (
            torch.nn.functional.scaled_dot_product_attention,
            _scaled_dot_product_attention,
            True,
            _export_scaled_dot_product_attention,
        ),
        (
            torch.ops.aten.scaled_dot_product_attention.default,
            _scaled_dot_product_attention,
            False,
            _export_scaled_dot_product_attention,
        ),
    ]


def install(env=None):
    """Overrides scaled_dot_product_attention on a torchax environment."""
    _torchax_registry.install(env or torchax.default_env(), provider="iree.attention")
