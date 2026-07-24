---
date: 2026-07-23
authors:
  - egebeysel   # from docs/website/docs/community/blog/.authors.yml
categories:
  - Performance
tags:
  - RISC-V
  - CPU
readtime: 8
---

# Running models on RISC-V with IREE

IREE compiles machine learning models to native RISC-V CPU code, with support for
the RISC-V Vector extension (RVV), hand-written microkernels, and data-tiling.
This post walks through the full flow for a model: importing it from PyTorch,
compiling it for a RISC-V target, running it, and benchmarking the result.

All commands below run under `qemu-riscv64`. The flow on real hardware is
identical — the QEMU invocation is simply replaced by running the tools natively
on the target.

<!-- more -->

## Setup

IREE is a cross-compiler: the compiler is built on the host, the runtime is
cross-compiled for the target, and the runtime is then copied to the target (or
run under QEMU). The
[RISC-V cross-compilation guide](https://iree.dev/building-from-source/riscv/)
covers this in full. In brief:

- `./build_tools/riscv/riscv_bootstrap.sh` downloads a prebuilt clang toolchain
  and QEMU into `~/riscv`.
- Build and install the host compiler, then cross-build the runtime with the
  `build_tools/cmake/linux_riscv64.cmake` toolchain file.
- Point `QEMU_BIN` at `qemu-riscv64` and `RISCV_TOOLCHAIN_ROOT` at the toolchain.

The result is an `iree-compile` on the host and `iree-run-module` /
`iree-benchmark-module` built for RISC-V.

## Importing a model

This post uses a few PyTorch models as a running example, but IREE supports
models from other frameworks such as LiteRT (TensorFlow Lite) and ONNX just as well,
once they have been imported to MLIR. See the
[ML frameworks guides](https://iree.dev/guides/ml-frameworks/) for the
per-framework export/import steps — for example
[PyTorch](https://iree.dev/guides/ml-frameworks/pytorch/),
[LiteRT / TensorFlow Lite](https://iree.dev/guides/ml-frameworks/tflite/), and
[ONNX](https://iree.dev/guides/ml-frameworks/onnx/). Also check out the
IREE community meeting
[presentation](https://youtu.be/UvH9rVe9_KA?si=Z6aYmsspb1SLs0ik) by Artem
Gindinson from Roofline.

For PyTorch, [iree-turbine](https://iree.dev/guides/ml-frameworks/pytorch/)'s
`aot.export` produces the MLIR. The following script exports two torchvision
models, saving an input for each to feed later:

```python
# export.py
import numpy as np, torch, torchvision as tv
import iree.turbine.aot as aot

def dump(name, model, example):
    aot.export(model.eval(), example).save_mlir(f"{name}.mlir")
    np.save(f"{name}_input.npy", example.numpy())

# Vision models (torchvision), NCHW float input.
dump("mobilenet", tv.models.mobilenet_v2(weights="DEFAULT"), torch.randn(1, 3, 224, 224))
dump("resnet18",  tv.models.resnet18(weights="DEFAULT"),     torch.randn(1, 3, 224, 224))
```

The exported entry point is `@main`, see the `--function=main` flag for the
`iree-*-module` invocations below.

Alternatively, you can check the models in the [IREE test suites](https://github.com/iree-org/iree-test-suites/tree/main/torch_models).
We have some ready-to-compile `.mlir` files whose weights are kept in a
separate `.irpa`
(IREE parameter archive) — which keeps the `.mlir` small and lets you swap weights
without recompiling. For example, [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B):

```shell
curl -L -o qwen3.mlir https://raw.githubusercontent.com/iree-org/iree-test-suites/main/torch_models/qwen3-600m/model.mlir
curl -L -o qwen3.irpa https://huggingface.co/roofline/iree-regression-models/resolve/main/qwen3-600m/real_weights.irpa
```

The weights are supplied at run time with `--parameters=` (see below).

## Compiling for RISC-V

The base command to produce RISC-V vector code is:

```shell
iree-compile mobilenet.mlir -o mobilenet_rv64.vmfb \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64 \
  --iree-llvmcpu-target-abi=lp64d \
  --iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+zvl512b,+v
```

The flag that matters most on RISC-V is **`--iree-llvmcpu-target-cpu-features`**,
which specifies the ISA. `+m,+a,+f,+d,+c` is `rv64gc`, `+v` enables RVV 1.0, and
`+zvl512b` declares the minimum vector register width (VLEN) — 512 bits here.
The `zvl`
value should match the target hardware's actual VLEN — 512 for the QEMU
configuration used below, 256 on a device such as a SpaceMiT X60 — since a
mismatch leaves the vector units underutilized. VLEN is the key RISC-V knob: it
drives LLVM's vector codegen *and* the tile sizes IREE selects for data-tiling
(more on that below).

The remaining flags are the optimization knobs. None of them are RISC-V-specific,
but they are where the performance comes from, so they are layered on top of the
base command.

### Data-tiling

`--iree-opt-data-tiling` repacks matmul-shaped operations into a tiled `mmt4d`
layout that maps cleanly onto the vector unit. It is off by default; most models,
especially matmul-heavy models, benefit from this. On RISC-V the tile shape
depends on
VLEN, so the `+zvl*b` value chosen above also determines the produced layout. The
[data-tiling walkthrough](https://iree.dev/community/blog/2025-08-25-data-tiling-walkthrough/)
and [mmt4d blogpost](https://iree.dev/community/blog/2021-10-13-matrix-multiplication-with-mmt4d/)
cover the mechanism in detail.

### im2col for convolutions

`--iree-global-opt-use-im2col-for-convs=true` rewrites convolutions as im2col plus
matmul, so that convolutions use the same optimized matmul, data-tiling, and
microkernel path as everything else. It is also off by default, and is beneficial
for most of the convolution models above. Native data-tiling support for convolutions
is still work-in-progress.

### Microkernels

`--iree-llvmcpu-enable-ukernels=...` selects IREE's hand-written microkernels
instead of relying solely on the generic vectorizer:

- `mmt4d`, `pack`, `unpack` — enable specific microkernels (comma-separated)
- `all` — all of them
- `none` — none
- `default` — IREE's per-target default

Data-tiling together with the `mmt4d` microkernel is the recommended combination
on RISC-V. Enabling data-tiling while disabling microkernels makes the packed
`mmt4d` fall back to generic vectorization; although this also generally produces
efficient code, the microkernel path is currently the most stable one.
For background, see the
[microkernels](https://iree.dev/community/blog/2024-01-22-microkernels/) and
[mmt4d](https://iree.dev/community/blog/2021-10-13-mmt4d/) posts.

### Static vs. scalable RVV

At present, IREE's RISC-V vector path is **static / fixed-length**: the VLEN is
fixed at compile time through `+zvl*b`, and both LLVM's vectorizer and IREE's
tile-size selection specialize to that width. IREE derives its `mmt4d` tile shapes
from the target's fixed-width vector register width, and the microkernels are compiled
for that same `+zvl*b` target, so the VLEN is the single value everything keys
off of.

There is also preliminary support for **scalable, vector-length-agnostic** RVV
codegen — the `vscale`-style path that runs on any VLEN without recompiling, but
it is still a work in progress. The scalable vectorization pipeline can be
activated with `--iree-llvmcpu-enable-scalable-vectorization=true` (which
currently has to be combined with `--iree-experimental-vscale-value=VLEN/64`
flag due to some ongoing work on the host compiler).

Combined, a performance-oriented compilation for the mobilenet example is:

```shell
iree-compile mobilenet.mlir -o mobilenet_rv64.vmfb \
  --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-triple=riscv64 --iree-llvmcpu-target-abi=lp64d \
  --iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+zvl512b,+v \
  --iree-opt-data-tiling \
  --iree-global-opt-use-im2col-for-convs=true
```

## Running the module

Copy the `.vmfb` and the cross-built `iree-run-module` to the target, or run under
QEMU. Vector QEMU requires its `vlen` to match the `+zvl512b` used at compile time:

```shell
${QEMU_BIN} -cpu rv64,Zve64d=true,vlen=512,elen=64,vext_spec=v1.0 \
  -L ${RISCV_TOOLCHAIN_ROOT}/sysroot/ \
  ../iree-build-riscv/tools/iree-run-module \
  --device=local-task \
  --module=mobilenet_rv64.vmfb \
  --function=main \
  --input=@mobilenet_input.npy
```

`--device=local-task` selects the multithreaded runtime; `--device=local-sync`
runs single-threaded and inline. Passing `--expected_output=@ref.npy` compares the
result against a saved reference output as a correctness check. See
`iree-run-module --help` for the `--input` / `--output` formats - inline literals,
splats such as `=0`, or `@file.npy`.

For a model whose weights live in a separate `.irpa` (like the Qwen3 above), pass
them with `--parameters=<scope>=<file>`; the scope is baked into the `.mlir` (here
`model`):

```shell
iree-run-module --device=local-task \
  --module=qwen3_rv64.vmfb --parameters=model=qwen3.irpa \
  --function=main --input=1x5xi64=1
```

## Benchmarking

`iree-benchmark-module` accepts the same module, device, and input flags, and adds
the [Google Benchmark](https://github.com/google/benchmark) options on top:

```shell
${QEMU_BIN} -cpu rv64,Zve64d=true,vlen=512,elen=64,vext_spec=v1.0 \
  -L ${RISCV_TOOLCHAIN_ROOT}/sysroot/ \
  ../iree-build-riscv/tools/iree-benchmark-module \
  --device=local-task \
  --module=mobilenet_rv64.vmfb \
  --function=main \
  --input=@mobilenet_input.npy \
  --benchmark_repetitions=10
```

The output looks like:

```text
Benchmark                     Time             CPU   Iterations
BM_main/real_time          12.3 ms         41.0 ms           57
BM_main/real_time_mean     12.4 ms         41.2 ms           10
BM_main/real_time_median   12.3 ms         41.0 ms           10
BM_main/real_time_stddev    0.2 ms          0.7 ms           10
```

- **`--benchmark_repetitions`** greater than 1 produces the mean, median, and
  standard-deviation rows.
- Other useful options: `--benchmark_min_time=1s` (or `100x` for a fixed iteration
  count) and `--benchmark_format=json`.

To control threading, pin workers to specific cores with
`--task_topology_cpu_ids=0,1,2,3`, or run single-threaded with `--device=local-sync`
(preferably compiled with `--iree-llvmcpu-disable-distribution=true`). See
`iree-run-module --help` for the other `--task_topology_*` options
(worker/group counts, NUMA nodes, performance level).

Note that under QEMU these are functional results rather than representative
performance numbers. Representative timings require real hardware (or cycle-accurate
simulators, which are hardly feasible to use for large programs that ML/AI
models are).

## Summary

This post walks through building IREE, then importing, compiling, running, and
benchmarking a model on RISC-V. For more on the general flow and other CPU
targets, see IREE's [CPU deployment guide](https://iree.dev/guides/deployment-configurations/cpu/).
