# Experimental Torchax integration

Exports stateless Torch programs through Torchax and StableHLO for IREE.
Read-only parameters, buffers and lifted constants remain explicit inputs.
The exporter preserves symbolic input shapes and attention composites. Mutation,
effect tokens and stochastic exports are unsupported. The API and dependency
versions are experimental.

This is a separate package, imported as `iree_torchax`. It is not included in the
compiler wheel. Compiler lowerings remain in the StableHLO input plugin.

## Installation

Use Python 3.12 or newer and IREE compiler/runtime packages from the same revision.
Install the pinned dependencies and the package from the repository root:

```sh
python -m pip install -r experimental/torchax/requirements-test.txt \
  --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/
python -m pip install --no-deps -e experimental/torchax
```

The pins include development versions available from upstream repositories or
locally built wheels. When using IREE build-tree bindings, set:

```sh
export PYTHONPATH="$PWD/build/compiler/bindings/python:$PWD/build/runtime/bindings/python${PYTHONPATH:+:$PYTHONPATH}"
```

## Export

```python
import torch
import iree_torchax

exported = torch.export.export(model, example_inputs)
program, artifact = iree_torchax.exported_program_to_stablehlo(exported)
module_text = str(artifact.mlir_module())
```

Compile `module_text` with IREE's StableHLO input plugin. The entry takes
`*program.constants` followed by the user inputs, and returns the user outputs.
`program.constant_names` follows graph-signature order; tied tensors share an
entry. `program.mutable_names` and `program.mutable` are empty. For direct JAX
use, `program.func(program.constants, (), inputs)` returns `((), outputs)`;
callers tracing this function choose their JAX settings.

Importing the facade loads neither Torchax nor JAX. Accessing an exporter loads
them. Each export captures the registration catalog, creates a private Torchax
environment and scopes its JAX dtype and matmul settings. Those settings are
restored on success or failure. User-defined Torchax handlers take precedence.
`install(env)` explicitly installs handlers in an eager environment chosen by
the caller.

Torchax itself renames Torch's private-use backend on import. This process-wide
change cannot be undone by a context manager. To preserve the parent's Torch
environment, export in a child process and exchange files. For example, the child
can save the result of the export above:

```python
from pathlib import Path
import numpy as np

Path("model.mlir").write_text(module_text)
np.savez("constants.npz", *(np.asarray(value) for value in program.constants))
```

The parent compiles the MLIR and passes constants in archive order before user
inputs. Runtime execution needs neither Torchax nor JAX.

## Registration

Plugins can keep operations whole as versioned StableHLO composites:

```python
scale = iree_torchax.composite(lambda x: x * 2, name="example.scale")

def scale_handler(x, *, env):
    return env.j2t_iso(scale(x.jax()))

iree_torchax.register_composite(
    torch.ops.example.scale.default, scale_handler, provider="example.ops"
)
```

An optional `export_handler` specializes a composite for export. Alternatively,
`register_decomposition(op, decomposition, provider=...)` expands an operation
with a Torch callable. Repeating an identical registration is harmless; replacing
a different registration requires `replace=True`. Composites and decompositions
cannot replace one another. A prepared export retains its captured registrations.
Compiler plugins consume emitted composites through the StableHLO target registry.

The producer keeps symbolic dimension bookkeeping inside the decomposition when
those dimensions can be recovered from operand shapes. This preserves the tensor
operand signature used by compiler handlers. If a decomposition needs a platform
selector or dimensions that cannot be recovered, the original JAX signature is
preserved and handlers can decline in favor of decomposition. Multi-platform
export remains supported. This lazy lowering adapter uses private APIs from the
pinned JAX version.

The pinned JAX version cannot execute some symbolic composite artifacts through
`artifact.call` because its shape refinement does not update decomposition
signatures. This also affects upstream `jax.lax.composite`. Compile these
artifacts with IREE; direct concrete JAX calls remain available.

## Tests

With dependencies and the package installed:

```sh
python experimental/torchax/run_tests.py
```

The runner requires compiler and runtime bindings and starts each test module in
its own process. Tests cover stateless export, graph-signature ordering, lazy
imports, scoped settings, registration snapshots and attention. They need no model
downloads. Compiler regressions remain in the compiler's MLIR suites.

The `CI - Experimental Torchax` workflow builds IREE and exercises the separately
installed wheel. Graduation requires a reproducible pretrained SmolLM2 run,
maintained dependency compatibility and a settled environment contract.
