# IREE: An Experimental MLIR Execution Environment

**DISCLAIMER**: This is an early phase project that we hope will graduate into a
supported form someday, but it is far from ready for everyday use and is made
available without any support. With that said, feel free to browse the issues
and reach out on the
[iree-discuss mailing list](https://groups.google.com/forum/#!forum/iree-discuss).

## Table of Contents

-   [Contact](#contact)
-   [Build Status](#build-status)
-   [Quickstart](#quickstart)
-   [Project Goals](#project-goals)
-   [Milestones](#milestones)
-   [Status](#status)
-   [Dependencies](#dependencies)
-   [License](#license)

## Communication Channels

*   [GitHub Issues](https://github.com/google/iree/issues): Preferred for
    specific issues and coordination on upcoming features.
*   [Google IREE Discord Server](https://discord.gg/26P4xW4): The core team and
    collaborators hang out here.
*   [Google Groups Email List](https://groups.google.com/forum/#!forum/iree-discuss):
    General, low-priority discussion.

### Related Project Channels

*   [MLIR topic within LLVM Discourse](https://llvm.discourse.group/c/llvm-project/mlir/31):
    Often, discussions that span IREE and various infrastructure topics will
    fork into topics on this forum.

## Build Status

CI System      | Build System | Platform | Status
-------------- | ------------ | -------- | ------
GitHub Actions | Bazel        | Linux    | [Workflow History](https://github.com/google/iree/actions?query=event%3Apush+workflow%3A%22Bazel+Build%22)
Kokoro         | Bazel        | Linux    | [![kokoro-status-bazel-linux](https://storage.googleapis.com/iree-oss-build-badges/bazel/build_status_linux.svg)](https://storage.googleapis.com/iree-oss-build-badges/bazel/build_result_linux.html)
Kokoro         | CMake        | Linux    | [![kokoro-status-cmake-linux](https://storage.googleapis.com/iree-oss-build-badges/cmake/build_status_linux.svg)](https://storage.googleapis.com/iree-oss-build-badges/cmake/build_result_linux.html)

## Quickstart

More Coming soon! Performing full model translation may require a few steps
(such as ensuring you have a working TensorFlow build), however we'll have
pre-translated example models that allow independent testing of the runtime
portions.

*   [Getting Started on Windows](docs/getting_started_on_windows.md)
*   [Getting Started on Linux](docs/getting_started_on_linux.md)
*   [Getting Started](docs/getting_started.md)

See also:

*   [Using Colab](docs/using_colab.md)
*   [Vulkan and SPIR-V](docs/vulkan_and_spirv.md)
*   [Function ABI](docs/function_abi.md)
*   [MNIST example on IREE](docs/mnist_example.md)
*   [Details of IREE Repository Management](docs/repository_management.md)

## Talks

We occasionally have either productive, recorded meetings or talks and will post
them here.

*   [March 18, 2020: Interactive HAL IR Walkthrough (Ben Vanik and core team)](https://drive.google.com/open?id=1FDrW9wvmiCQsVNSNzTD_V0bYmBl0XTyQ)
*   [Jan 31, 2020: IREE Presentation at MLIR ODM](https://drive.google.com/open?id=1os9FaPodPI59uj7JJI3aXnTzkuttuVkR)
    ([slides](https://drive.google.com/open?id=1RCQ4ZPQFK9cVgu3IH1e5xbrBcqy7d_cEZ578j84OvYI))

## Project Goals

IREE (**I**ntermediate **R**epresentation **E**xecution **E**nvironment,
pronounced as "eerie") is an experimental compiler backend for
[MLIR](https://mlir.llvm.org/) that lowers ML models to an IR that is optimized
for real-time mobile/edge inference against heterogeneous hardware accelerators.

The IR produced contains the sequencing information required to communicate
pipelined data dependencies and parallelism to low-level hardware APIs like
Vulkan and embed hardware/API-specific binaries such as SPIR-V or compiled ARM
code. As the IR is specified against an abstract execution environment there are
many potential ways to run a compiled model, and one such way is included as an
example and testbed for runtime optimization experiments.

The included layered runtime scales from generated code for a particular API
(such as emitting C code calling external DSP kernels), to a HAL (**H**ardware
**A**bstraction **L**ayer) that allows the same generated code to target
multiple APIs (like Vulkan and Direct3D 12), to a full VM allowing runtime model
loading for flexible deployment options and heterogeneous execution. Consider
both the compiler and the included runtime a toolbox for making it easier - via
the versatility of MLIR - to take ML models from their source to some varying
degree of integration with your application.

### Demonstrate MLIR

IREE has been developed alongside MLIR and is used as an example of how
non-traditional ML compiler backends and runtimes can be built: it focuses more
on the math being performed and how that math is scheduled rather than graphs of
"ops" and in some cases allows doing away with a runtime entirely. It seeks to
show how more holistic approaches that exploit the MLIR framework and its
various dialects can be both easy to understand and powerful in the
optimizations to code size, runtime complexity, and performance they enable.

### Demonstrate Advanced Models

By using models with much greater complexity than the usual references (such as
MobileNet) we want to show how weird things can get when model authors are
allowed to get creative: dynamic shapes, dynamic flow control, dynamic
multi-model dispatch (including models that conditionally dispatch other
models), streaming models, tree-based search algorithms, etc. We are trying to
build IREE from the ground-up to enable these models and run them efficiently on
modern hardware. Many of our example models are sequence-to-sequence language
models from the [Lingvo](https://github.com/tensorflow/lingvo) project
representing cutting edge speech recognition and translation work.

### Demonstrate ML-as-a-Game-Engine

An observation that has driven the development of IREE is one of ML workloads
not being much different than traditional game rendering workloads: math is
performed on buffers with varying levels of concurrency and ordering in a
pipelined fashion against accelerators designed to make such operations fast. In
fact, most ML is performed on the same hardware that was designed for games! Our
approach is to use the compiler to transform ML workloads to ones that look
eerily _(pun intended)_ similar to what a game performs in per-frame render
workloads, optimize for low-latency and predictable execution, and integrate
well into existing systems both for batched and interactive usage. The IREE
runtime is designed to feel more like game engine middleware than a standalone
ML inference system, though we still have much work towards that goal. This
should make it easy to use existing tools for high-performance/low-power
optimization of GPU workloads, identify driver or system issues introducing
latency, and help to improve the ecosystem overall.

### Demonstrate Standards-based ML via Vulkan and SPIR-V

With the above observation that ML can look like games from the systems
perspective it follows that APIs and technologies good for games should probably
also be good for ML. In many cases we've identified only a few key differences
that exist and just as extensions have been added and API improvements have been
made to graphics/compute standards for decades we hope to demonstrate and
evaluate small, tactical changes that can have large impacts on ML performance
through these standard APIs. We would love to allow hardware vendors to be able
to make ML efficient on their hardware without the need for bespoke runtimes and
special access such that any ML workload produced by any tool runs well. We'd
consider the IREE experiment a success if what resulted was some worked examples
that help advance the entire ecosystem!

## Milestones

We are currently just at the starting line, with basic
[MNIST MLP](https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py)
running end-to-end on both a CPU interpreter and Vulkan. As we scale out the
compiler we will be increasing the complexity of the models that can run and
demonstrating more of the optimizations we've found useful in previous efforts
to run them efficiently on edge devices.

A short-term
[Roadmap](https://github.com/google/iree/blob/master/docs/roadmap.md) is
available talking about the major areas where are focusing on in addition to the
more infrastructure-focused work listed below.

We'll be setting up GitHub milestones with issues tracking major feature work we
are planning. For now, our areas of work are:

*   Allocation tracking and aliasing in the compiler
*   Pipelined scheduler in the VM for issuing proper command buffers
*   New CPU interpreter that enables lightweight execution on ARM and x86
*   C code generator and API to demonstrate "runtimeless" mode
*   Quantization using the MLIR quantization framework
*   High-level integration and examples when working with TensorFlow 2.0
*   Switching from IREE's XLA-to-SPIR-V backend to the general MLIR SPIR-V
    backend

Things we are interested in but don't yet have in-progress:

*   Ahead-of-time compiled ARM NEON backend (perhaps via
    [SPIRV-LLVM](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/),
    [SPIRV-to-ISPC](https://github.com/GameTechDev/SPIRV-Cross), or some other
    technique)
*   HAL backends for Metal 2 and Direct3D 12
*   Profile-guided optimization support for scheduling feedback

## Current Status

### Documentation

Coming soon :)

### Build System and CI

*   We support Bazel for builds of all parts of the project.
*   We also maintain a CMake build for a subset of runtime components designed
    to be used in other systems.

### Code and Style

The project is currently _very_ early and a mix of code written prior to a lot
of the more recent ergonomics improvements in MLIR and its tablegen. Future
changes will replace the legacy code style with prettier forms and simplify the
project structure to make it easier to separate the different components. Some
entire portions of the code (such as the CPU interpreter) will likely be dropped
or rewritten. For now, assume churn!

The compiler portions of the code (almost exclusively under `iree/compiler/`)
follows the LLVM style guide and has the same system requirements as MLIR
itself. It general requires a more modern C++ compiler.

The runtime portions vary but most are designed to work with C++11 and use
[Abseil](https://github.com/abseil/abseil-cpp) to bring in future C++14 and
C++17 features.

### Hardware Support

We are mostly targeting Vulkan and Metal on recent mobile devices and as such
have limited our usage of hardware features and vendor extensions to those we
have broad access to there. This is mainly just to keep our focus tight and does
not preclude usage of features outside the standard sets or for other hardware
types (in fact, we have a lot of fun ideas for
`VK_NVX_device_generated_commands` and Metal 2.1's Indirect Command Buffers!).

## Dependencies

NOTE: during the initial open source release we are still cleaning things up. If
there's weird dependencies/layering that makes life difficult for your
particular use case please file an issue so we can make sure to fix it.

### Compiler

The compiler has several layers that allow scaling the dependencies required
based on the source and target formats. In all cases
[MLIR](https://mlir.llvm.org/) is required and for models not originating from
TensorFlow (or already in XLA HLO format) it is the only dependency. When
targeting the IREE Runtime VM and HAL
[FlatBuffers](https://google.github.io/flatbuffers/) is required for
serialization. Converting from TensorFlow models requires a dependency on
TensorFlow (however only those parts required for conversion).

### Runtime VM

The VM providing dynamic model deployment and advanced scheduling behavior
requires [Abseil](https://github.com/abseil/abseil-cpp) for its common types,
however contributions are welcome to make it possible to replace Abseil with
other libraries via shims/forwarding. The core types used by the runtime
(excluding command line flags and such in tools) are limited to types coming in
future C++ versions (variant, optional, string_view, etc), cheap types
(absl::Span), or simple standard containers (absl::InlinedVector).
[FlatBuffers](https://google.github.io/flatbuffers/) is used to load compiled
modules.

### Runtime HAL

The HAL and the provided implementations (Vulkan, etc) also use
[Abseil](https://github.com/abseil/abseil-cpp). Contributions are welcome to
allow other types to be swapped in. A C99 HAL API is planned for code generation
targets that will use no dependencies.

### Testing and Tooling

[Swiftshader](https://github.com/google/swiftshader) is used to provide fast
hardware-independent testing of the Vulkan and SPIR-V portions of the toolchain.

## License
                                 Apache License
                           Version 2.0, January 2004
                        https://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright  Rolando Gopez Lacuata.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       https://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

