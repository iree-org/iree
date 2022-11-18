# Temporary Branch To Iterate On The Transform Dialect Integration

Some scripts and examples I am using successully:
```
source tests/transform_dialect/ntv_script.sh
```

Then, be sure iree-opt, iree-compile, iree-run-module and iree-benchmarks are built
and in the path (my full setup can be found 
[here](https://github.com/nicolasvasilache/nicolas.vasilache.github.io/blob/master/.venv/mlirdev/bin/activate)).

Some quick getting started CPU instructions:
```
# Compile matmul.mlir with IREE up to HAL, apply call the transform dialect 
# codegen script: tests/transform_dialect/cpu/matmul_codegen_default_spec.mlir
# and print the IR to stdout.
iree-transform-opt tests/transform_dialect/cpu/matmul.mlir -b llvm-cpu -c tests/transform_dialect/cpu/matmul_codegen_default_spec.mlir

# Same as above + run the rest of the IREE pipeline to produce a binary dumped to stdout.
iree-transform-compile tests/transform_dialect/cpu/matmul.mlir -b llvm-cpu -c tests/transform_dialect/cpu/matmul_codegen_default_spec.mlir

# Same compile as above and pipe through iree-run-module with input values for execution.
iree-transform-compile tests/transform_dialect/cpu/matmul.mlir -b llvm-cpu -c tests/transform_dialect/cpu/matmul_codegen_default_spec.mlir | \
  iree-run-module --entry_function=matmul_static  --function_input="3x5xf32=1"  --function_input="5x3xf32=1"  --function_input="3x3xf32=0"

# Same compile as above + add extra args for target-triple and benchmarking then pipe through iree-benchmark-module to benchmark
iree-transform-compile tests/transform_dialect/cpu/matmul.mlir -b llvm-cpu -c tests/transform_dialect/cpu/matmul_codegen_default_spec.mlir \
  -- --iree-llvm-target-triple=x86_64-pc-linux-gnu   --iree-llvm-target-cpu-features=host  --iree-hal-benchmark-dispatch-repeat-count=100 | \
  iree-benchmark-module --device=local-task --task_topology_group_count=0 --batch_size=100 --entry_function=matmul_static  --function_input="3x5xf32=1"  --function_input="5x3xf32=1"  --function_input="3x3xf32=0"
```

Some quick getting started GPU instructions:
```
# Compile matmul.mlir with IREE up to HAL, apply call the transform dialect 
# codegen script: tests/transform_dialect/cuda/reduction_codegen_spec.mlir
iree-transform-opt  tests/transform_dialect/cuda/reduction.mlir -b cuda -c tests/transform_dialect/cuda/reduction_codegen_spec.mlir

# Same as above + run the rest of the IREE pipeline to produce a binary dumped to stdout. This shows ptx.
iree-transform-compile  tests/transform_dialect/cuda/reduction.mlir -b cuda -c tests/transform_dialect/cuda/reduction_codegen_spec.mlir

# Always be sure to run more than a few times because we see a lot of compulsory paging misses on the first run.
iree-transform-compile  tests/transform_dialect/cuda/reduction.mlir -b cuda -c tests/transform_dialect/cuda/reduction_codegen_spec.mlir -- --iree-hal-benchmark-dispatch-repeat-count=5 | \
  nvprof  --print-gpu-trace  iree-run-module --entry_function=reduce --device=cuda 
```

# IREE: Intermediate Representation Execution Environment

IREE (**I**ntermediate **R**epresentation **E**xecution **E**nvironment,
pronounced as "eerie") is an [MLIR](https://mlir.llvm.org/)-based end-to-end
compiler and runtime that lowers Machine Learning (ML) models to a unified IR
that scales up to meet the needs of the datacenter and down to satisfy the
constraints and special considerations of mobile and edge deployments.

See [our website](https://iree-org.github.io/iree/) for project details, user
guides, and instructions on building from source.

[![CI Status](https://github.com/iree-org/iree/actions/workflows/ci.yml/badge.svg?query=branch%3Amain+event%3Apush)](https://github.com/iree-org/iree/actions/workflows/ci.yml?query=branch%3Amain+event%3Apush)

#### Project Status

IREE is still in its early phase. We have settled down on the overarching
infrastructure and are actively improving various software components as well as
project logistics. It is still quite far from ready for everyday use and is made
available without any support at the moment. With that said, we welcome any kind
of feedback on any [communication channels](#communication-channels)!

## Communication Channels

*   [GitHub issues](https://github.com/iree-org/iree/issues): Feature requests,
    bugs, and other work tracking
*   [IREE Discord server](https://discord.gg/26P4xW4): Daily development
    discussions with the core team and collaborators
*   [iree-discuss email list](https://groups.google.com/forum/#!forum/iree-discuss):
    Announcements, general and low-priority discussion

#### Related Project Channels

*   [MLIR topic within LLVM Discourse](https://llvm.discourse.group/c/llvm-project/mlir/31):
    IREE is enabled by and heavily relies on [MLIR](https://mlir.llvm.org). IREE
    sometimes is referred to in certain MLIR discussions. Useful if you are also
    interested in MLIR evolution.

## Architecture Overview

<!-- TODO(scotttodd): switch to <picture> once better supported? https://github.blog/changelog/2022-05-19-specify-theme-context-for-images-in-markdown-beta/ -->
![IREE Architecture](docs/website/docs/assets/images/iree_architecture_dark.svg#gh-dark-mode-only)
![IREE Architecture](docs/website/docs/assets/images/iree_architecture.svg#gh-light-mode-only)

See [our website](https://iree-org.github.io/iree/) for more information.

## Presentations and Talks

*   2021-06-09: IREE Runtime Design Tech Talk ([recording](https://drive.google.com/file/d/1p0DcysaIg8rC7ErKYEgutQkOJGPFCU3s/view) and [slides](https://drive.google.com/file/d/1ikgOdZxnMz1ExqwrAiuTY9exbe3yMWbB/view?usp=sharing))
*   2020-08-20: IREE CodeGen: MLIR Open Design Meeting Presentation
    ([recording](https://drive.google.com/file/d/1325zKXnNIXGw3cdWrDWJ1-bp952wvC6W/view?usp=sharing)
    and
    [slides](https://docs.google.com/presentation/d/1NetHjKAOYg49KixY5tELqFp6Zr2v8_ujGzWZ_3xvqC8/edit))
*   2020-03-18: Interactive HAL IR Walkthrough
    ([recording](https://drive.google.com/file/d/1_sWDgAPDfrGQZdxAapSA90AD1jVfhp-f/view?usp=sharing))
*   2020-01-31: End-to-end MLIR Workflow in IREE: MLIR Open Design Meeting Presentation
    ([recording](https://drive.google.com/open?id=1os9FaPodPI59uj7JJI3aXnTzkuttuVkR)
    and
    [slides](https://drive.google.com/open?id=1RCQ4ZPQFK9cVgu3IH1e5xbrBcqy7d_cEZ578j84OvYI))

## License

IREE is licensed under the terms of the Apache 2.0 License with LLVM Exceptions.
See [LICENSE](LICENSE) for more information.
