# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""IREE's Torchax binding.

    import iree_torchax

Plugins register composites and decompositions per operation. Importing this
module loads neither Torchax nor JAX and changes no Torch or Torchax state;
the exporters and the names that need Torchax load it on first access. Each
export installs the composites on the environment it builds for itself, and
`install(env)` installs them on an eager environment.

The import traces, exports and compiles nothing. Torchax 0.0.14 renames Torch's
private-use backend when it loads, after which a strict `torch.export` in the
same process fails. The first access to an exporter or `JaxProgram` loads Torchax and JAX,
and their state stays for the life of
the process; `install(env)` changes `env`.

A process that must keep Torch as it was runs the export in a process of its
own and exchanges files with it:

    # export.py, run as its own process.
    import numpy as np
    import torch
    import iree_torchax

    exported = torch.export.export(model, example_inputs)
    program, artifact = iree_torchax.exported_program_to_stablehlo(exported)
    with open("model.mlir", "w") as f:
        f.write(iree_torchax.externalize_parameters(
            str(artifact.mlir_module()), program.constant_names))
    iree_torchax.save_parameters(program, "model.irpa")
    np.savez("state.npz", *(np.asarray(value) for value in program.mutable))

The parent runs `export.py`, compiles `model.mlir` with
`input_type="stablehlo"`, and loads the module with `model.irpa` as the
`model` parameter scope, as `iree-run-module --parameters=model=model.irpa`
does; `state.npz` holds the initial state it passes to the first call. The
parent never imports Torchax. The child's Torch changes stay in the child; they
are not undone. The in-process functions remain for callers that want live
objects such as `JaxProgram`, with the limit above.

Plugins extend the binding after importing it:

    scale = iree_torchax.composite(lambda x: x * 2, name="example.scale")

    def scale_handler(x, *, env):
        return env.j2t_iso(scale(x.jax()))

    iree_torchax.register_composite(
        torch.ops.example.scale.default, scale_handler, provider="example.ops"
    )

`register_composite` keeps an operation whole through export and emits its
composite; `register_decomposition` expands an operation with a Torch callable
before export. Registering the same thing twice is harmless; a different
registration of an occupied operation raises `ValueError` unless
`replace=True`, and a composite never replaces a decomposition or the reverse.
A handler registered in Torchax's global op registry as user-defined wins over
IREE's in an export; an override on one environment, the default one included,
stays in that environment. `install(env)` keeps a user's override in `env`
unless `replace=True`. Each export uses the registrations made before it
started.
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
    "externalize_parameters": "iree_torchax.torchax_export",
    "save_parameters": "iree_torchax.torchax_export",
}

__all__ = [
    "JaxProgram",
    "composite",
    "exported_program_to_jax",
    "exported_program_to_stablehlo",
    "externalize_parameters",
    "install",
    "register_composite",
    "register_decomposition",
    "save_parameters",
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
