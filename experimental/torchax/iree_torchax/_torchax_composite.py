# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""StableHLO composites from JAX functions, also under symbolic shapes.

`composite(decomposition, name=..., version=...)` returns a function over JAX
values that emits `stablehlo.composite` with `decomposition` as its body.
Positional arguments are operands; keyword arguments are static composite
attributes, also passed to `decomposition`.
"""

import functools

__all__ = ["composite"]


def composite(decomposition, *, name, version=1):
    """As jax.lax.composite, valid under symbolic shapes too."""
    if not callable(decomposition):
        raise TypeError(f"decomposition must be callable, got {decomposition!r}")
    if not isinstance(name, str):
        raise TypeError(f"name must be a string, got {name!r}")
    if not name:
        raise ValueError("name must be nonempty")
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError(f"version must be an int, got {version!r}")
    if not 0 <= version < 2**31:
        raise ValueError(f"version {version} is outside 0..2^31-1")

    @functools.cache
    def emit():
        from ._torchax_composite_lowering import composite as lower_composite

        return lower_composite(decomposition, name=name, version=version)

    @functools.wraps(decomposition)
    def bind(*args, **kwargs):
        return emit()(*args, **kwargs)

    return bind
