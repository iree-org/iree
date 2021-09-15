# IREE: Intermediate Representation Execution Environment

IREE (**I**ntermediate **R**epresentation **E**xecution **E**nvironment,
pronounced as "eerie") is an [MLIR](https://mlir.llvm.org/)-based end-to-end
compiler and runtime that lowers Machine Learning (ML) models to a unified IR
that scales up to meet the needs of the datacenter and down to satisfy the
constraints and special considerations of mobile and edge deployments.

See [our website](https://google.github.io/iree/) for project details, user
guides, and instructions on building from source.

#### Project Status

IREE is still in its early phase. We have settled down on the overarching
infrastructure and are actively improving various software components as well as
project logistics. It is still quite far from ready for everyday use and is made
available without any support at the moment. With that said, we welcome any kind
of feedback on any [communication channels](#communication-channels)!

## Communication Channels

*   [GitHub issues](https://github.com/google/iree/issues): Feature requests,
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

## Build Status


CI System | Build System  | Platform   | Architecture         | Configuration / Component    | Status
:-------: | :-----------: | :--------: | :------------------: | :--------------------------: | :----:
Kokoro    | Bazel         | Linux      | x86-64               |                              | [![kokoro_status_bazel/linux/x86-swiftshader/core](https://storage.googleapis.com/iree-oss-build-badges/bazel/linux/x86-swiftshader/core/main_status.svg)](https://storage.googleapis.com/iree-oss-build-badges/cmake-bazel/linux/x86-swiftshader/main_result.html)
Kokoro    | CMake & Bazel | Linux      | x86-64 (swiftshader) | Integrations                 | [![kokoro status cmake-bazel/linux/x86-swiftshader](https://storage.googleapis.com/iree-oss-build-badges/cmake-bazel/linux/x86-swiftshader/main_status.svg)](https://storage.googleapis.com/iree-oss-build-badges/cmake-bazel/linux/x86-swiftshader/main_result.html)
Kokoro    | CMake & Bazel | Linux      | x86-64 (turing)      | Integrations                 | [![kokoro status cmake-bazel/linux/x86-turing](https://storage.googleapis.com/iree-oss-build-badges/cmake-bazel/linux/x86-turing/main_status.svg)](https://storage.googleapis.com/iree-oss-build-badges/cmake-bazel/linux/x86-turing/main_result.html)
Kokoro    | CMake         | Linux      | x86-64 (swiftshader) |                              | [![kokoro status cmake/linux/x86-swiftshader](https://storage.googleapis.com/iree-oss-build-badges/cmake/linux/x86-swiftshader/main_status.svg)](https://storage.googleapis.com/iree-oss-build-badges/cmake/linux/x86-swiftshader/main_result.html)
Kokoro    | CMake         | Linux      | x86-64 (swiftshader) | asan                         | [![kokoro status cmake/linux/x86-swiftshader-asan](https://storage.googleapis.com/iree-oss-build-badges/cmake/linux/x86-swiftshader-asan/main_status.svg)](https://storage.googleapis.com/iree-oss-build-badges/cmake/linux/x86-swiftshader-asan/main_result.html)
Kokoro    | CMake         | Linux      | x86-64 (turing)      |                              | [![kokoro status cmake/linux/x86-turing](https://storage.googleapis.com/iree-oss-build-badges/cmake/linux/x86-turing/main_status.svg)](https://storage.googleapis.com/iree-oss-build-badges/cmake/linux/x86-turing/main_result.html)
Kokoro    | CMake         | Android    | arm64-v8a            | Runtime (build only)         | [![kokoro status cmake/android/arm64-v8a](https://storage.googleapis.com/iree-oss-build-badges/cmake/android/arm64-v8a/main_status.svg)](https://storage.googleapis.com/iree-oss-build-badges/cmake/android/arm64-v8a/main_result.html)
Kokoro    | CMake         | Bare Metal | risc-v-32            | Runtime                      | [![kokoro status cmake/baremetal/riscv32](https://storage.googleapis.com/iree-oss-build-badges/cmake/baremetal/riscv32/main_status.svg)](https://storage.googleapis.com/iree-oss-build-badges/cmake/baremetal/riscv32/main_result.html)
Kokoro    | CMake         | Linux      | risc-v-64            | Runtime                      | [![kokoro status cmake/linux/riscv64](https://storage.googleapis.com/iree-oss-build-badges/cmake/linux/riscv64/main_status.svg)](https://storage.googleapis.com/iree-oss-build-badges/cmake/linux/riscv64/main_result.html)
Buildkite | CMake         | Android    | arm64-v8a            | Runtime                      | [![buildkite status iree-android-arm64-v8a](https://badge.buildkite.com/a73df0ba9f4aa132650dd6676bc1e6c20d3d99ed6b24db2179.svg?branch=main)](https://buildkite.com/iree/iree-android-arm64-v8a)
BuildKite | CMake         | Android    | arm64-v8a            | Runtime Benchmarks           | [![buildkite status iree-benchmark](https://badge.buildkite.com/62e504b93171f4a19e5c46f8b9a99eb5dba050666640fbc21b.svg?branch=main)](https://buildkite.com/iree/iree-benchmark)
BuildKite | CMake         | Linux      | x86-64               | Tracing + Standalone Runtime | [![buildkite status iree-build-configurations](https://badge.buildkite.com/3bc03ad54a6b785b3fdd0dd3d67fd93ed22ef2b538cb34adc3.svg?branch=main)](https://buildkite.com/iree/iree-build-configurations)

## Architecture Overview

![IREE Architecture](docs/website/docs/assets/images/iree_architecture.svg)

See [our website](https://google.github.io/iree/) for more information.

## Presentations and Talks

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
