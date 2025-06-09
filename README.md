# IREE: Intermediate Representation Execution Environment

<p><img src="docs/website/docs/assets/images/IREE_Logo_Icon_Color.svg" width="48px"></p>

IREE (**I**ntermediate **R**epresentation **E**xecution **E**nvironment,
pronounced as "eerie") is an [MLIR](https://mlir.llvm.org/)-based end-to-end
compiler and runtime that lowers Machine Learning (ML) models to a unified IR
that scales up to meet the needs of the datacenter and down to satisfy the
constraints and special considerations of mobile and edge deployments.

See [our website](https://iree.dev/) for project details, user
guides, and instructions on building from source.

[![IREE Discord Status](https://discordapp.com/api/guilds/689900678990135345/widget.png?style=shield)]([https://discord.gg/wEWh6Z9nMU](https://discord.gg/wEWh6Z9nMU))
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8738/badge)](https://www.bestpractices.dev/projects/8738)

## Project news

* 2025-04-02:
[AMD submitted an IREE-based SDXL implementation to the MLPerf benchmark suite](https://rocm.blogs.amd.com/artificial-intelligence/mi325x-accelerates-mlperf-inference/README.html#stable-diffusion-xl-sdxl-text-to-image-mlperf-inference-benchmark)
* 2024-05-23:
[IREE joins the LF AI & Data Foundation as a sandbox-stage project](https://lfaidata.foundation/blog/2024/05/23/announcing-iree-a-new-initiative-for-machine-learning-deployment/)

## Project status

### Release status

Releases notes are
[published on GitHub releases](https://github.com/iree-org/iree/releases?q=prerelease%3Afalse).

| Package | Release status |
| -- | -- |
GitHub release (stable) | [![GitHub Release](https://img.shields.io/github/v/release/iree-org/iree)](https://github.com/iree-org/iree/releases/latest)
GitHub release (nightly) | [![GitHub Release](https://img.shields.io/github/v/release/iree-org/iree?include_prereleases)](https://github.com/iree-org/iree/releases)
`iree-base-compiler` | [![PyPI version](https://badge.fury.io/py/iree-base-compiler.svg)](https://pypi.org/project/iree-base-compiler)
`iree-base-runtime` | [![PyPI version](https://badge.fury.io/py/iree-base-runtime.svg)](https://pypi.org/project/iree-base-runtime)

For more details on the release process, see
https://iree.dev/developers/general/release-management/.

### Build status

[![CI](https://github.com/iree-org/iree/actions/workflows/ci.yml/badge.svg?query=branch%3Amain+event%3Apush)](https://github.com/iree-org/iree/actions/workflows/ci.yml?query=branch%3Amain+event%3Apush)
[![PkgCI](https://github.com/iree-org/iree/actions/workflows/pkgci.yml/badge.svg?query=branch%3Amain+event%3Apush)](https://github.com/iree-org/iree/actions/workflows/pkgci.yml?query=branch%3Amain+event%3Apush)

#### Nightly build status

| Operating system | Build status |
| -- | --: |
Linux | [![CI - Linux arm64 clang](https://github.com/iree-org/iree/actions/workflows/ci_linux_arm64_clang.yml/badge.svg?query=branch%3Amain+event%3Aschedule)](https://github.com/iree-org/iree/actions/workflows/ci_linux_arm64_clang.yml?query=branch%3Amain+event%3Aschedule)
macOS | [![CI - macOS x64 clang](https://github.com/iree-org/iree/actions/workflows/ci_macos_x64_clang.yml/badge.svg?query=branch%3Amain+event%3Aschedule)](https://github.com/iree-org/iree/actions/workflows/ci_macos_x64_clang.yml?query=branch%3Amain+event%3Aschedule)
macOS | [![CI - macOS arm64 clang](https://github.com/iree-org/iree/actions/workflows/ci_macos_arm64_clang.yml/badge.svg?query=branch%3Amain+event%3Aschedule)](https://github.com/iree-org/iree/actions/workflows/ci_macos_arm64_clang.yml?query=branch%3Amain+event%3Aschedule)

For the full list of workflows see
https://iree.dev/developers/general/github-actions/.

## Communication channels

*   [GitHub issues](https://github.com/iree-org/iree/issues): Feature requests,
    bugs, and other work tracking
*   [IREE Discord server](https://discord.gg/wEWh6Z9nMU): Daily development
    discussions with the core team and collaborators
*   (New) [iree-announce email list](https://lists.lfaidata.foundation/g/iree-announce):
    Announcements
*   (New) [iree-technical-discussion email list](https://lists.lfaidata.foundation/g/iree-technical-discussion):
    General and low-priority discussion
*   (Legacy) [iree-discuss email list](https://groups.google.com/forum/#!forum/iree-discuss):
    Announcements, general and low-priority discussion

### Related project channels

*   [MLIR topic within LLVM Discourse](https://llvm.discourse.group/c/llvm-project/mlir/31):
    IREE is enabled by and heavily relies on [MLIR](https://mlir.llvm.org). IREE
    sometimes is referred to in certain MLIR discussions. Useful if you are also
    interested in MLIR evolution.

## Architecture overview

<!-- TODO(scotttodd): switch to <picture> once better supported? https://github.blog/changelog/2022-05-19-specify-theme-context-for-images-in-markdown-beta/ -->
![IREE Architecture](docs/website/docs/assets/images/iree_architecture_dark.svg#gh-dark-mode-only)
![IREE Architecture](docs/website/docs/assets/images/iree_architecture.svg#gh-light-mode-only)

See [our website](https://iree.dev/) for more information.

## Presentations and talks

Community meeting recordings: [IREE YouTube channel](https://www.youtube.com/@iree4356)

Date | Title | Recording | Slides
---- | ----- | --------- | ------
2025-02-12 | The Long Tail of AI: SPIR-V in IREE and MLIR (Vulkanised) | [recording](https://youtu.be/0zwfc6UkxeE) | [slides](https://www.vulkan.org/user/pages/09.events/vulkanised-2025/T12-Jakub-Kuderski-AMD-IREE-MLIR.pdf)
2024-10-01 | Unveiling the Inner Workings of IREE: An MLIR-Based Compiler for Diverse H/W | [recording](https://www.youtube.com/watch?v=a3T74I9gGH8) |
2021-06-09 | IREE Runtime Design Tech Talk | [recording](https://drive.google.com/file/d/1p0DcysaIg8rC7ErKYEgutQkOJGPFCU3s/view) | [slides](https://drive.google.com/file/d/1ikgOdZxnMz1ExqwrAiuTY9exbe3yMWbB/view?usp=sharing)
2020-08-20 | IREE CodeGen (MLIR Open Design Meeting) | [recording](https://drive.google.com/file/d/1325zKXnNIXGw3cdWrDWJ1-bp952wvC6W/view?usp=sharing) | [slides](https://docs.google.com/presentation/d/1NetHjKAOYg49KixY5tELqFp6Zr2v8_ujGzWZ_3xvqC8/edit)
2020-03-18 | Interactive HAL IR Walkthrough | [recording](https://drive.google.com/file/d/1_sWDgAPDfrGQZdxAapSA90AD1jVfhp-f/view?usp=sharing) |
2020-01-31 | End-to-end MLIR Workflow in IREE (MLIR Open Design Meeting) | [recording](https://drive.google.com/open?id=1os9FaPodPI59uj7JJI3aXnTzkuttuVkR) | [slides](https://drive.google.com/open?id=1RCQ4ZPQFK9cVgu3IH1e5xbrBcqy7d_cEZ578j84OvYI)

## License

IREE is licensed under the terms of the Apache 2.0 License with LLVM Exceptions.
See [LICENSE](LICENSE) for more information.
