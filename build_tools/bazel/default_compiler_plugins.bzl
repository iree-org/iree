# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Default compiler plugin registry for Bazel builds.

IREE's plugin aggregation BUILD file loads this file from the root workspace
with @//build_tools/bazel:default_compiler_plugins.bzl. When IREE is built as
part of a larger Bazel workspace, that root workspace may provide a file at the
same path to register its own compiler plugins or override the default set.
"""

def iree_default_compiler_plugin_ids():
    """Returns plugin IDs enabled by plain Bazel invocation."""
    return [
        "input_stablehlo",
        "input_tosa",
        "hal_target_cuda",
        "hal_target_llvm_cpu",
        "hal_target_local",
        "hal_target_metal_spirv",
        "hal_target_rocm",
        "hal_target_vmvx",
        "hal_target_vulkan_spirv",
        "example",
        "simple_io_sample",
    ]

def iree_default_compiler_plugins():
    """Returns all known compiler plugin registration targets."""
    return {
        # Input plugins.
        "input_stablehlo": "//compiler/plugins/input/StableHLO:registration",
        "input_tosa": "//compiler/plugins/input/TOSA:registration",
        "input_torch": "//compiler/plugins/input/Torch:registration",
        # Target plugins.
        "hal_target_cuda": "//compiler/plugins/target/CUDA",
        "hal_target_llvm_cpu": "//compiler/plugins/target/LLVMCPU",
        "hal_target_local": "//compiler/plugins/target/Local",
        "hal_target_metal_spirv": "//compiler/plugins/target/MetalSPIRV",
        "hal_target_rocm": "//compiler/plugins/target/ROCM",
        "hal_target_vmvx": "//compiler/plugins/target/VMVX",
        "hal_target_vulkan_spirv": "//compiler/plugins/target/VulkanSPIRV",
        # Sample plugins.
        "example": "//samples/compiler_plugins/example:registration",
        "simple_io_sample": "//samples/compiler_plugins/simple_io_sample:registration",
    }
