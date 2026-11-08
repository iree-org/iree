# Experimental Torchax integration

Exports Torch programs through Torchax and StableHLO for compilation with IREE.
The integration preserves symbolic shapes, mutable buffers, registered composites
and named parameters. Its Python API and dependency versions are experimental.

This is a separate Python package, imported as `iree_torchax`. The IREE compiler
wheel does not include it. Compiler lowerings and runtime support remain in their
normal IREE locations.

## Installation

Use Python 3.12 or newer and compiler/runtime Python packages built from the same
IREE revision. The tests currently require development versions of Torchax and
JAX; `requirements-test.txt` records the tested versions. Install those versions
from their upstream package repositories or locally built wheels, then install
this package from the repository root:

```sh
python -m pip install -r experimental/torchax/requirements-test.txt \
  --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/
python -m pip install --no-deps -e experimental/torchax
```

When using an IREE build tree instead of installed compiler/runtime wheels, make
its Python bindings available:

```sh
export PYTHONPATH="$PWD/build/compiler/bindings/python:$PWD/build/runtime/bindings/python${PYTHONPATH:+:$PYTHONPATH}"
```

## Export

```python
import torch
import iree_torchax

exported = torch.export.export(model, example_inputs)
program, artifact = iree_torchax.exported_program_to_stablehlo(exported)
```

`exported_program_to_stablehlo` reads no weight data: it traces the shapes and
types of the constants and returns the program's own tensors, so a model on the
meta device exports too. `save_parameters(program, path)` writes the constants
to the IRPA archive straight from their storage. `exported_program_to_jax`
returns JAX arrays aliasing the weights; a constant JAX cannot alias, one not
contiguous and 64-byte aligned, is an error unless the caller passes
`copy_constants=True`.

Importing `iree_torchax` loads neither Torchax nor JAX. Each export installs its
handlers in its own environment and scopes its JAX precision settings. Plugins
use `register_composite`, `register_decomposition` and `composite` on this facade.

Torchax itself changes Torch's private-use backend on first use. That change
cannot be undone by a context manager. If the caller must retain its original
Torch environment, run export in a child process and exchange MLIR, parameter
archives and initial state through files. The package docstring includes that
example. Runtime execution does not require Torchax or JAX.

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

From the repository root, with the package installed:

```sh
python experimental/torchax/run_tests.py
```

The runner requires all dependencies, including IREE runtime bindings, and runs
each test file in a fresh process. Individual tests can also be selected, for
example:

```sh
python experimental/torchax/tests/torchax_export_test.py TorchaxExportTest.test_symbolic_cache_prefix_updates
```

Keep compiler regressions as MLIR tests in their existing compiler test suites;
they must not depend on installing this package. No model downloads are needed
for the Python suite.

For a Bazel worktree, use the same Python 3.12 environment and install the
separate Torchax wheel with `--no-deps` after installing the test requirements.
The runner builds the native compiler/runtime extensions and tools, assembles a
temporary Python package from this checkout's source/generated bindings, and
runs the suite against those bindings:

```sh
uv build --wheel --out-dir /tmp/iree-torchax-wheels experimental/torchax
python -m pip install --no-deps /tmp/iree-torchax-wheels/iree_torchax-*.whl
python experimental/torchax/run_bazel.py
```

It selects Bazel's Python 3.12 configuration without changing the default Python
version. An explicit script and its arguments run in the same environment:
`python experimental/torchax/run_bazel.py path/to/script.py [arguments]`.
Use `--no-build` only while the Python 3.12 binding targets are already current.

The `CI - Experimental Torchax` workflow builds IREE at the checked-out revision
and runs this suite against the separately installed wheel. It can be invoked
manually or called by another workflow. The JAX index follows the upstream
[nightly installation instructions](https://docs.jax.dev/en/latest/installation.html#jax-nightly-installation).

State export donates mutable buffers by default: pass every returned state value
into the next invocation and discard the donated inputs. Use
`exported_program_to_stablehlo(exported, donate_state=False)` when earlier state
values must remain available. This preserves functional values across invocations
and allows the compiler to reuse storage only where its ownership rules permit it.

## Pretrained inference check

The f32 SmolLM2-135M example pins the pretrained checkpoint revision and uses
its Transformers decoder layers with registered tensor cache buffers. Explicit
position inputs supply causal coordinates and symbolic prefix lengths. Export
runs in a child process, preserving the reference process's Torch environment.

After installing the separate Torchax wheel, install the model dependencies
and run through the Bazel bindings:

```sh
python -m pip install -r experimental/torchax/requirements-smollm2.txt \
  --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry/simple/
python experimental/torchax/run_bazel.py \
  experimental/torchax/examples/smollm2_cpu.py --output-dir /tmp/smollm2-cpu
```

This downloads the pinned weights/tokenizer, exports symbolic prefill and decode,
and writes an external IRPA parameter archive. It runs one prefill and two decode
invocations, passing all 60 returned cache tensors forward. Logits and every cache
tensor are compared with Torch at `rtol=atol=1e-4`; the cached reference is also
checked against ordinary full-prefix inference. Predictions must agree.

Both registered attention specialization and portable decomposition run. The
fallback check deliberately changes the attention composite to unsupported
version 2 while retaining its decomposition. Input IR checks confirm the two
lowering paths. `results.json` records dependency/model versions, archive hash,
predictions, and maximum numerical errors; the directory also retains MLIR and
compiled modules. `--model-dir` accepts an already-downloaded snapshot of the
pinned revision; `--reuse-export` reuses its existing exported artifacts.

The default profile uses batch one, capacity 16, generic LLVM CPU and local-task.
Select a native backend with `--target-backend` and, when needed, a runtime
device URI with `--device`. Repeat `--compiler-arg=...` for target options.
For example, `--target-backend=vmvx` uses local-task, and
`--target-backend=vulkan-spirv --device=vulkan` uses Vulkan. Use a separate
output directory for each target. For this f32 model, VMVX also requires
`--compiler-arg=--iree-vmvx-reassociate-fp-reductions` to meet the numerical
comparison policy. This opt-in setting groups eligible f32 sums into contiguous
chunks of 64 before combining their partial sums. It changes association and
rounded results, including possible overflow and underflow, and does not
promise improved accuracy for every input. Extra partial-sum storage and loops
may increase execution time. The original init participates only
in the final sum. Other types, nondivisible or dynamic reduction extents, and
unsupported scalar bodies keep their original lowering. Reassociation is
otherwise disabled by default; the setting is retained in executable target
configuration when compilation is resumed from a checkpoint.

For the f32 Metal comparison, use the supplied projection configuration with a
Metal-enabled runtime and compiler tools from this checkout. A CMake runtime
build must enable `IREE_HAL_DRIVER_METAL` and `IREE_BUILD_PYTHON_BINDINGS`; the
Bazel staging runner above currently provides a runtime without the Metal driver.

```sh
python experimental/torchax/examples/smollm2_cpu.py \
  --output-dir /tmp/smollm2-metal --target-backend=metal-spirv --device=metal \
  --retain-state \
  --compiler-arg=--iree-metal-compile-to-metallib=false \
  --compiler-arg=--iree-codegen-transform-dialect-library=experimental/torchax/examples/smollm2_spirv_f32.mlir
```

This explicit profile uses the existing SPIR-V vectorisation pipeline with
serial FMA accumulation for the model's transposed-weight f32 projections.
It covers reduction widths 576 and 1536 with column counts divisible by 32,
including symbolic row counts. It preserves existing user configurations and
leaves other operations, including batched attention contractions, unchanged.
The accumulation order agrees with the pinned Torch reference in the recorded
validation environment. Parallel accumulation can be closer to a float64 oracle
while differing more from that f32 reference; this profile makes no general
accuracy or speed claim and does not change compiler defaults. Other hosts and
Torch builds still require validation. The profile and its configuration
checkpoint are checked by `tests/smollm2_tuning_test.py` in the separate Torchax
suite. `--iree-codegen-tuning-spec-path` alone does not apply a profile through
the current SPIR-V pipeline; use the transform-library option shown above.

The recorded Apple M5 validation passed both attention modes, prefill and two
decodes, every cache and predicted token at the unchanged `1e-4` comparison
policy. It covered retained and donated state and three prompt lengths.
For the five-token prefill, 21 alternating measurements after three warmups gave
median times of 27.1 ms without the profile and 23.0 ms with it, including logit
readback but excluding compilation and parameter loading. These measurements
describe this workload and device.

For WASI, the same harness builds a small direct-VM runner with the checkout's
Bazel WASI toolchain and executes it with Node. Keep `smollm2_cpu.py`,
`smollm2_wasm.py`, and `wasi/smollm2_wasi.c` together. From the repository root,
with the same Python bindings and model dependencies installed:

```sh
python experimental/torchax/run_bazel.py \
  experimental/torchax/examples/smollm2_cpu.py \
  --output-dir /tmp/smollm2-wasi-vmvx --wasi --target-backend=vmvx \
  --retain-state --compiler-arg=--iree-vmvx-reassociate-fp-reductions
python experimental/torchax/run_bazel.py \
  experimental/torchax/examples/smollm2_cpu.py \
  --output-dir /tmp/smollm2-wasi-llvm --wasi --target-backend=llvm-cpu \
  --retain-state
```

The LLVM variant emits wasm32 static executable objects; the VMVX variant uses
the VMVX loader. Both use local-sync, preloaded external parameters and the
maintained WASI entry point. The runner checks every output cache, logits and
predicted token at the same tolerances, and reads retained physical inputs
again after invocation. Omit `--retain-state` to exercise donation in a separate
output directory. Node and Bazel failures propagate. `results.json` records
the build command and runner hashes alongside the model results.
Node can access only the output directory; reused parameter archives must be
stored there rather than symlinked to a file outside it.

The generated `wasi_model` directory is an injected Bazel repository, so no
experimental build package needs to be added to the main workspace. It also
contains `iree_runtime_plugin.cmake` for an existing WASI CMake runtime build
using `IREE_CMAKE_PLUGIN_PATHS`; enable file I/O with
`-DCMAKE_C_FLAGS=-DIREE_FILE_IO_ENABLE=1`. Selecting the VMVX plugin also enables
`IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE` for its required runtime loader. Use
`--prepare-wasi-runner-only` to generate its modules and build files first.
`--prepare-reference-only` writes portable NPY inputs and Torch references
without compiling or executing IREE. The focused checker test is
`examples/wasi/test_runner.py`; it runs small stateful modules through both WASI
loaders and checks numerical, token, file and bytecode failure propagation.

Recorded local validation passed native LLVM CPU, VMVX and Metal, plus WASI
VMVX and LLVM-WASM: 60 model invocations covering both attention paths and both
state ownership modes, with all 60 caches, logits and predicted tokens checked.
Vulkan compilation passed for all eight model modules; Vulkan hardware execution
was unavailable. The comparison policy remains `rtol=atol=1e-4`, with exact
predicted-token equality. Full validation across the applicable CI matrix is
still required before merge.

Backend and device failures propagate; the harness does not skip unavailable
or failing targets.
These f32 runs do not establish BF16 or complete GPU conformance: the broader
suite has a BF16 mixed-matmul discrepancy also present when weights are ordinary
arguments.
Graduation still requires maintained dependency compatibility and a settled
export/environment contract beyond these pretrained f32 checkpoints.

Use a separate output directory with `--retain-state` to disable cache donation
and check that every old input cache remains unchanged after each invocation.
This exercises the functional ownership path in addition to the default donated
state path. The exported signature and recorded `state_donated` setting must
match when reusing an export.
