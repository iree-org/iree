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

#### Project Status

IREE is still in its early phase. We have settled down on the overarching
infrastructure and are actively improving various software components as well as
project logistics. It is still quite far from ready for everyday use and is made
available without any support at the moment. With that said, we welcome any kind
of feedback on any [communication channels](#communication-channels)

#### Release status

| Package | Release status |
| -- | -- |
GitHub release (stable) | [![GitHub Release](https://img.shields.io/github/v/release/iree-org/iree)](https://github.com/iree-org/iree/releases/latest)
GitHub release (nightly) | [![GitHub Release](https://img.shields.io/github/v/release/iree-org/iree?include_prereleases)](https://github.com/iree-org/iree/releases)
Python iree-compiler | [![PyPI version](https://badge.fury.io/py/iree-compiler.svg)](https://badge.fury.io/py/iree-compiler)
Python iree-runtime | [![PyPI version](https://badge.fury.io/py/iree-runtime.svg)](https://badge.fury.io/py/iree-runtime)

#### Build status

[![CI](https://github.com/iree-org/iree/actions/workflows/ci.yml/badge.svg?query=branch%3Amain+event%3Apush)](https://github.com/iree-org/iree/actions/workflows/ci.yml?query=branch%3Amain+event%3Apush)
[![PkgCI](https://github.com/iree-org/iree/actions/workflows/pkgci.yml/badge.svg?query=branch%3Amain+event%3Apush)](https://github.com/iree-org/iree/actions/workflows/pkgci.yml?query=branch%3Amain+event%3Apush)

| Host platform | Build status |
| -- | --: |
Linux | [![CI - Linux x64 clang](https://github.com/iree-org/iree/actions/workflows/ci_linux_x64_clang.yml/badge.svg?query=branch%3Amain+event%3Apush)](https://github.com/iree-org/iree/actions/workflows/ci_linux_x64_clang.yml?query=branch%3Amain+event%3Apush)<br>[![CI - Linux arm64 clang](https://github.com/iree-org/iree/actions/workflows/ci_linux_arm64_clang.yml/badge.svg?query=branch%3Amain+event%3Aschedule)](https://github.com/iree-org/iree/actions/workflows/ci_linux_arm64_clang.yml?query=branch%3Amain+event%3Aschedule)
macOS | [![CI - macOS x64 clang](https://github.com/iree-org/iree/actions/workflows/ci_macos_x64_clang.yml/badge.svg?query=branch%3Amain+event%3Aschedule)](https://github.com/iree-org/iree/actions/workflows/ci_macos_x64_clang.yml?query=branch%3Amain+event%3Aschedule)
Windows | [![CI - Windows x64 MSVC](https://github.com/iree-org/iree/actions/workflows/ci_windows_x64_msvc.yml/badge.svg?query=branch%3Amain+event%3Aschedule)](https://github.com/iree-org/iree/actions/workflows/ci_windows_x64_msvc.yml?query=branch%3Amain+event%3Aschedule)

For the full list of workflows see
https://iree.dev/developers/general/github-actions/.

## Communication Channels

*   [GitHub issues](https://github.com/iree-org/iree/issues): Feature requests,
    bugs, and other work tracking
*   [IREE Discord server](https://discord.gg/wEWh6Z9nMU): Daily development
    discussions with the core team and collaborators
*   [iree-announce email list](https://lists.lfaidata.foundation/g/iree-announce):
    Announcements
*   [iree-technical-discussion email list](https://lists.lfaidata.foundation/g/iree-technical-discussion):
    General and low-priority discussion

#### Related Project Channels

*   [MLIR topic within LLVM Discourse](https://llvm.discourse.group/c/llvm-project/mlir/31):
    IREE is enabled by and heavily relies on [MLIR](https://mlir.llvm.org). IREE
    sometimes is referred to in certain MLIR discussions. Useful if you are also
    interested in MLIR evolution.

## Architecture Overview

<!-- TODO(scotttodd): switch to <picture> once better supported? https://github.blog/changelog/2022-05-19-specify-theme-context-for-images-in-markdown-beta/ -->
![IREE Architecture](docs/website/docs/assets/images/iree_architecture_dark.svg#gh-dark-mode-only)
![IREE Architecture](docs/website/docs/assets/images/iree_architecture.svg#gh-light-mode-only)

See [our website](https://iree.dev/) for more information.

## Presentations and Talks

Community meeting recordings: [IREE YouTube channel](https://www.youtube.com/@iree4356)

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
