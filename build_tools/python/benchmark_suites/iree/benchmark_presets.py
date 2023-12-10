## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Presets to group benchmarks by their characteristics.

To put a benchmark into a preset, add the preset to its `presets` field.
"""

# Default Android CPU execution benchmarks.
ANDROID_CPU = "android-cpu"
# Default Android GPU execution benchmarks.
ANDROID_GPU = "android-gpu"
# Default CUDA execution benchmarks.
CUDA = "cuda"
# Large CUDA execution benchmarks.
CUDA_LARGE = "cuda-large"
# Default RISC-V execution benchamrks.
RISCV = "riscv"
# Default Vulkan NVIDIA execution benchamrks.
VULKAN_NVIDIA = "vulkan-nvidia"
# Default x86_64 execution benchmarks.
X86_64 = "x86_64"
# x86_64 execution benchmarks with only data-tiling.
X86_64_DT_ONLY = "x86_64-dt-only"
# Large x86_64 execution benchmarks.
X86_64_LARGE = "x86_64-large"

# Default compilation benchmark preset.
COMP_STATS = "comp-stats"
# Large compilation benchmark preset.
COMP_STATS_LARGE = "comp-stats-large"

# Default execution benchmark presets.
DEFAULT_PRESETS = [
    ANDROID_CPU,
    ANDROID_GPU,
    CUDA,
    RISCV,
    VULKAN_NVIDIA,
    X86_64,
]
# Large execution benchmark presets.
LARGE_PRESETS = [
    CUDA_LARGE,
    X86_64_LARGE,
]

ALL_EXECUTION_PRESETS = DEFAULT_PRESETS + LARGE_PRESETS + [X86_64_DT_ONLY]
ALL_COMPILATION_PRESETS = [COMP_STATS, COMP_STATS_LARGE]
