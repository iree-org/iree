## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Benchmark presets to pass through build system and tools, which present
collections of benchmarks to compile and run.
"""

# Architecture-based execution benchmark presets:
ANDROID_CPU = "android-cpu"
ANDROID_GPU = "android-gpu"
CUDA = "cuda"
CUDA_LARGE = "cuda-large"
RISCV = "riscv"
VULKAN_NVIDIA = "vulkan-nvidia"
X86_64 = "x86_64"
X86_64_LARGE = "x86_64-large"
# Size-based execution benchmark presets:
DEFAULT = "default"
LARGE = "large"
# All execution benchmark presets:
EXECUTION_PRESETS = [
    ANDROID_CPU,
    ANDROID_GPU,
    CUDA,
    CUDA_LARGE,
    DEFAULT,
    LARGE,
    RISCV,
    VULKAN_NVIDIA,
    X86_64,
    X86_64_LARGE,
]

# Compilation benchmark presets:
COMP_STATS = "comp-stats"
COMP_STATS_LARGE = "comp-stats-large"
# All compilation benchmark presets:
COMPILATION_PRESETS = [
    COMP_STATS,
    COMP_STATS_LARGE,
]
