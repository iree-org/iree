# Compiler microbenchmarks

Small, single-pattern MLIR programs that make compiler changes visible in two
ways:

* **Structure**: which dispatches the compiler forms for each case and whether
  they are data-tiled. This is deterministic and needs no benchmarking host.
* **Codegen configuration** (optional, `--codegen-configs`): which codegen
  pipeline strategy selection picked for each dispatch and the tile sizes it
  attached to each op. Also deterministic, but target-specific.
* **Time**: the median run time of each case, measured with
  `iree-benchmark-module`. This is for local A/B checks on a quiet machine.

A typical use is checking a compiler change against its merge base: run the
suite with both `iree-compile` binaries and read the structural changes and time
ratios.

## Layout

```text
tests/microbenchmarks/
  README.md
  manifest.json              # targets, configs, per-case overrides (data)
  run_microbenchmarks.py     # runner, Python 3 standard library only
  cases/
    attention/               # attention op, decomposed attention, MHA block
    contraction/             # matmul, batch_matmul, matvec/vecmat, linear layers
    conv/                    # 1-D/2-D/3-D, strided, dilated, depthwise, grouped
    data_movement/           # transpose, reshape, pad, concat, strided slice
    elementwise/             # fusion chains, broadcasts, type conversion, QDQ
    normalization/           # softmax, layer norm, RMS norm, batch norm
    pooling/                 # max/avg/global pooling
    quantized/               # QDQ convs and matmuls
    reduction/               # inner/outer reductions, argmax
```

The directory sits next to `tests/e2e` because the cases are MLIR programs in
the same style as the e2e tests. Unlike the runtime C++ benchmarks under
`runtime/`, they measure what the compiler produces. The runner lives with the
cases so the directory is self-contained.

## Cases

A case is one `.mlir` file. Its id is the path under `cases/` without the
extension, e.g. `conv/shortcut_1024to2048_h14_s2_relu_bias`. Conventions:

* One op pattern per file, static shapes at realistic sizes, a comment at the
  top that says what the pattern is and where the shape comes from.
* Every public `func.func` is benchmarked. Its arguments must be static tensors
  or scalars of a float or integer type. The runner synthesizes splat inputs
  (`0.5` for floats, `1` for integers), so no data files are needed.
* Model parameters (weights, biases, scales, norm parameters) are constant
  globals initialized with a byte pattern:

  ```mlir
  util.global private @weight = #util.byte_pattern<63> : tensor<2048x1024x1x1xi8>
  ```

  As in a real model, the compiler can hoist and const-evaluate work on them,
  e.g. packing a data-tiled weight, and that work stays out of the timed call.
  The byte `0x3F` gives ordinary values in every element type (about 0.75 for
  f32 and bf16, 1.8 for f16, 63 for i8), so no denormals or NaNs.
* Activations are function arguments, so their packing and quantization are
  timed. Cases with two activation operands (e.g. `matmul_*`,
  `batch_matmul_*`) pack both at run time when data tiling is on.

`conv/shortcut_*` are the three ResNet-50 v1.5 downsample shortcuts (1x1 stride
2) in NCHW, plain and with a relu producer and bias. `*_s1_relu_bias` cases are
stride-1 controls with the same channels and input size. `quantized/` repeats
them, plus other shapes, in QDQ form (`iree_linalg_ext.quantize_affine` /
`dequantize_affine`, per-tensor activations with zero point -128, per-tensor or
per-channel weight scales, requantized output).

### Adding a case

Drop a new `.mlir` file into the right category directory. Check that it
compiles:

```shell
python3 tests/microbenchmarks/run_microbenchmarks.py \
  --iree-compile=<build>/tools/iree-compile --structure-only --case='conv/my_case'
```

## Targets and configs

`manifest.json` holds:

* `targets`: flags that select the device (`compile_flags`) and how to run
  (`run_flags`). Choose one with `--target`; `default_target` is used
  otherwise.
* `configs`: named compile flag sets layered on top of the target flags:
  `default`, `dt` (data tiling), `dt_uk` (data tiling and ukernels), and an
  `*_im2col` variant of each. A config can limit itself to cases with
  `include` / `exclude` globs over case ids. The im2col configs only apply to
  convolutions.
* `case_overrides`: `{"<glob>": {...}}` entries with `exclude_configs`,
  `only_configs` or `"skip_benchmark": true` (structure only), plus a free-form
  `reason`.

Adding a config or target is a manifest edit; the runner does not change.
Every config runs unless `--config` selects a subset.

## Running

Single build, structure and time:

```shell
python3 tests/microbenchmarks/run_microbenchmarks.py \
  --iree-compile=<build>/tools/iree-compile -o /tmp/mb
```

`iree-benchmark-module` is taken from the same directory unless
`--iree-benchmark-module` is given. Useful options:

* `--structure-only`: compile to the flow phase only. Takes seconds for the
  whole suite.
* `--codegen-configs`: also compile to the executable-configurations phase
  and record the `translation_info` of each dispatch and the `lowering_config`
  of each op. Use it for changes to backend strategy selection, which leave
  the flow-phase structure untouched.
* `--compile-only`: compile the modules but run nothing. In A/B mode this
  shows which modules differ between the two compilers; it also works for
  cross-compiled targets.
* `--case 'conv/*' --case '*qdq*'`, `--exclude-case`, `--config dt_uk`: select
  work. `--list` prints the selected `case@config` pairs.
* `--repetitions 5 --min-time 0.2s`: Google Benchmark repetitions and minimum
  time per repetition. The reported time is the median over the repetitions.
* `--threads N`: number of task workers (`--task_topology_group_count`).
  `--target host_cpu_sync` runs single-threaded on `local-sync` instead.
* `--dispatch-stats`: also record per-dispatch times
  (`--dispatch_statistics`). This runs each benchmark a second time.
* `-j N`: parallel compiles. Benchmarks always run one at a time.

A/B between two compilers:

```shell
python3 tests/microbenchmarks/run_microbenchmarks.py \
  --base=<base-build>/tools/iree-compile \
  --head=<head-build>/tools/iree-compile \
  --iree-benchmark-module=<build>/tools/iree-benchmark-module \
  --config dt_uk_im2col --case 'conv/*' --case 'quantized/*' -o /tmp/mb-ab
```

Both sides use the same `iree-benchmark-module` unless
`--base-iree-benchmark-module` is given. Base and head runs of the same entry
are interleaved to reduce drift. A/B over flags with one binary works with
`--base-flags` / `--head-flags`, e.g.
`--base-flags='' --head-flags='--iree-opt-experimental-early-data-tiling'`.

In A/B mode the compiled modules are compared first. Entries whose base and
head modules are byte-identical are not timed, because any difference would be
noise; the report says how many there were. `--bench-identical` times them
anyway, which is a way to measure the noise floor of the machine.

Two earlier runs can be compared without recompiling:

```shell
python3 tests/microbenchmarks/run_microbenchmarks.py \
  --compare-results /tmp/old/results.head.json /tmp/new/results.head.json
```

For CI-style checks, `--fail-on-structural-change` and `--fail-on-error` set a
non-zero exit code.

### Other targets and remote machines

The manifest has cross-compiled targets next to the host ones:
`riscv64_rvv`, `x86_64_avx2`, `x86_64_avx512` and `aarch64_sve`. On their own
they support `--structure-only` and `--compile-only`, which is enough to see
what a change does to the selected configurations and the generated modules
on a target you cannot run:

```shell
python3 tests/microbenchmarks/run_microbenchmarks.py \
  --base=<base-build>/tools/iree-compile --head=<build>/tools/iree-compile \
  --target x86_64_avx512 --compile-only --codegen-configs -o /tmp/mb-x86
```

To time a cross-compiled target, run the benchmarks on a matching machine over
ssh. The machine needs an `iree-benchmark-module` built from the same source
(a runtime-only build, `-DIREE_BUILD_COMPILER=OFF`, is enough):

```shell
python3 tests/microbenchmarks/run_microbenchmarks.py \
  --base=<base-build>/tools/iree-compile --head=<build>/tools/iree-compile \
  --target riscv64_rvv --remote-host user@board \
  --remote-iree-benchmark-module /home/user/iree-rt-build/tools/iree-benchmark-module \
  -o /tmp/mb-rv
```

Modules are compiled locally, copied through the ssh connection and run on the
remote host. `--remote-dir` sets the scratch directory there and
`--ssh-command` replaces `ssh` (options, a jump host, a wrapper). The same
settings can live in the target as a `remote` object with `host`, `dir`,
`iree_benchmark_module` and `ssh`. Base and head still alternate per entry;
on a small board, check for a run-order bias by swapping `--base` and `--head`
once.

### Base build in a worktree

A base compiler can be built in a git worktree of the merge base. Submodules
can be cloned from an existing checkout instead of downloading them again.
Match the head build's codegen-relevant CMake cache settings (build type,
assertions, `IREE_DEFAULT_CPU_LLVM_TARGETS`, host compiler, LLVM triples):

```shell
git worktree add ../iree-base <merge-base>
cd ../iree-base
M=<existing-checkout>/.git/modules/third_party
args=(-c protocol.file.allow=always)
for p in $(git config -f .gitmodules --get-regexp '\.path$' | awk '{print $2}'); do
  u=$(git config -f .gitmodules "submodule.$p.url")
  args+=(-c "url.$M/${p#third_party/}.insteadOf=$u")
done
git "${args[@]}" submodule update --init
cmake -G Ninja -B build -S . <same -D options as the head build>
ninja -C build iree-compile iree-benchmark-module iree-run-module
```

`iree-compile` is a thin launcher around `lib/libIREECompiler.*`; the recorded
`sha256` of a compiler is the library's when it is found.

## Output

The output directory contains:

* `report.md`: the table below, also printed to the terminal.
* `results.<side>.json`: all raw data: tool paths, sha256 and versions, host,
  flags, per entry the compile commands, status and first error line,
  dispatches, and per function every benchmark sample, the median, min, max,
  coefficient of variation and the benchmark context (CPU count, load average).
* `structure.<side>.json`: only the deterministic part (status, dispatch
  signature, data-tiled flag per entry). Suitable for checking in as a golden
  file or diffing in CI.
* `comparison.json` (A/B only): per-entry structural and time diffs.
* `<side>/<config>/`: `.vmfb` files, Google Benchmark JSON and logs of failed
  compiles or runs.

### Reading the structure

For each entry the report lists the dispatches in order, with the
`<func>_dispatch_<n>_` prefix removed, e.g.
`matmul_like_2048x196x1024_i8xi8xi32 [enc_in]`. The name is the one the
compiler gives the dispatch and encodes the root op kind, loop bounds and
element types. A conv rewritten to a matmul with the output pixels collapsed
shows one M dimension (`matmul_like_2048x196x1024`); a strided conv kept as a
contraction over the output height and width shows two
(`matmul_like_2048x7x7x1024`).

Data tiling is read from the flow-phase IR of each dispatch:

| tag | meaning |
|---|---|
| `enc_in` | the dispatch reads tensors with an `#iree_encoding`, i.e. its root op runs on data-tiled operands (codegen turns it into `mmt4d`) |
| `enc_out` | the dispatch writes an encoded tensor (a producer with a fused set_encoding) |
| `mmt4d`, `pack`, `unpack` | layouts materialized before dispatch creation, e.g. with early materialization |

`DT` is `yes` when any dispatch has `enc_in` or `mmt4d`. Encodings are used
instead of looking for `mmt4d` in later phases because the flow phase is
target-independent, cheap to reach and already shows whether the contraction
was selected for data tiling.

### Reading the codegen configuration

With `--codegen-configs`, each entry also has a `codegen` list with one item
per dispatch function: its `translation_info` (pipeline and pipeline options,
e.g. `ConvTileAndDecomposeExpert` with `enable_loop_peeling`) and, in order,
every op that carries a `lowering_config` with that config. This is the output
of strategy selection, before any lowering runs. Dispatches added by the
backend, such as the pack dispatches of data-tiled operands, show up here but
not in the flow-phase structure.

Unlike the structure, this depends on the target and its CPU features. An A/B
report gets a **Codegen configuration changes** section with the lines only in
base and only in head, and `structure.<side>.json` includes the configs.

### Reading an A/B report

* **Structural changes** lists every entry whose dispatch list (names and tags,
  in order) or status differs, with the dispatches only in base and only in
  head. Any change is listed, whatever the time does.
* **Codegen configuration changes** (with `--codegen-configs`) lists every
  entry whose pipelines or lowering configs differ. `--fail-on-structural-change`
  covers these too.
* **Time changes** lists entries whose head/base median ratio is outside
  `1 +/- --threshold` (default 5%) and whose base and head samples do not
  overlap (min..max). Ratio > 1 means head is slower.
* **All results** has every entry with both medians and the ratio, and whether
  the two modules are identical. Entries
  over the threshold whose samples overlap are marked `noise` there and not
  listed as time changes.

Compile and run failures are recorded per entry and never stop the run.

## Limits

* Times are noisy. Use a quiet machine, several repetitions, and check the
  `cv` column; a ratio change smaller than the cv means nothing. Rerun
  suspicious entries alone with more repetitions.
* Results are specific to the target and host. Structure on `llvm-cpu` with
  `--iree-llvmcpu-target-cpu=host` depends on the CPU features of the machine
  that compiles; record and compare structure on the same kind of host.
* These are microbenchmarks: one pattern, splat inputs, warm caches. They show
  that a pattern got faster or slower, not by how much a model will. Fusion
  across patterns and memory traffic between layers are not covered; confirm
  with an end-to-end model.
* Timing includes runtime packing of activation operands but not of constant
  parameters, as in a deployed model.

## Upstreaming notes

The suite is not wired into the build. To make it part of the tree:

* CMake/Bazel: none is needed to run it, but a lit or CTest target that runs
  `--structure-only --fail-on-error` against the built `iree-compile` would
  catch cases that stop compiling. Structure could be compared against a
  checked-in `structure.head.json` per target.
* CI: a job on a fixed runner type that runs structure-only on every PR and
  timing A/B on demand.
* Cases currently target `llvm-cpu` only. GPU targets need new entries in
  `targets` and probably per-target `case_overrides`.
