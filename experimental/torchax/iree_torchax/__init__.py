# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Experimental stateless Torch export and composite registration.

Importing iree_torchax loads neither Torchax nor JAX. Accessing an exporter
loads them. Each export captures registrations and uses a private environment;
install(env) explicitly changes the chosen eager environment. JAX dtype and
matmul settings are restored when export exits, including exceptional exits.

Torchax renames Torch's private-use backend on import. That process-wide change
cannot be undone by a context manager. Export in a child process when the parent
must retain its original Torch environment; exchange the StableHLO text and
read-only constants through files. Runtime execution needs neither Torchax nor
JAX. See the package README for an example.

Plugins use composite, register_composite and register_decomposition. Repeating
the same registration is harmless; conflicting registrations need replace=True.
A composite and a decomposition cannot replace one another. User-defined
Torchax handlers take precedence, and each export captures a stable snapshot.
"""

import importlib

from iree_torchax import _torchax_registry
from iree_torchax._torchax_composite import composite
from iree_torchax._torchax_registry import (
    register_composite,
    register_decomposition,
)

# Their modules import Torchax and JAX, so they load on first access.
_LAZY = {
    "JaxProgram": "iree_torchax.torchax_export",
    "exported_program_to_jax": "iree_torchax.torchax_export",
    "exported_program_to_stablehlo": "iree_torchax.torchax_export",
}

__all__ = [
    "JaxProgram",
    "composite",
    "exported_program_to_jax",
    "exported_program_to_stablehlo",
    "install",
    "register_composite",
    "register_decomposition",
]


def install(env, *, replace=False):
    """Installs every registered composite on `env`; a user's override there
    stays unless `replace` is set."""
    _torchax_registry.install(env, replace=replace)


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_LAZY[name]), name)
    globals()[name] = value
    return value
