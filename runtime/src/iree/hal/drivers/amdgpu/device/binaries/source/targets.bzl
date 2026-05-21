# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source-built AMDGPU builtin device binary targets."""

load(
    "@iree_amdgpu_device_toolchain//:paths.bzl",
    "AMDGPU_CLANG_RESOURCE_HEADERS",
    "AMDGPU_CLANG_RESOURCE_INCLUDE_ARGS",
    "AMDGPU_CLANG_RESOURCE_MARKER",
    "AMDGPU_CLANG_TOOL",
    "AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE",
    "AMDGPU_DEVICE_TOOLCHAIN_ERROR",
    "AMDGPU_LLD_TOOL",
    "AMDGPU_LLVM_LINK_TOOL",
    "AMDGPU_LLVM_OBJCOPY_TOOL",
)
load(
    "//runtime/src/iree/hal/drivers/amdgpu/device/binaries:target_map.bzl",
    "IREE_HAL_AMDGPU_DEVICE_BINARY_CODE_OBJECT_TARGETS",
)

_GENERATOR_SCRIPT = "//build_tools/scripts:amdgpu_device_binaries.py"
_GENERATOR_SCRIPT_DEPS = [
    _GENERATOR_SCRIPT,
    "//build_tools/scripts:amdgpu_target_map.py",
]

_GENERATOR_INPUTS = [
    "//runtime/src/iree/hal/drivers/amdgpu/device:bitcode_srcs",
    "//runtime/src/iree/hal/drivers/amdgpu/device:bitcode_hdrs",
    "//runtime/src/iree/hal/drivers/amdgpu/abi:amdgpu_abi_bitcode_hdrs",
    "//runtime/src/iree/hal/drivers/amdgpu/device:device_bitcode_sources.bzl",
]

# Source-built blobs need an explicit optional toolchain, so the targets must
# not appear in //... CI enumeration when the toolchain repo is the inert stub.
_INCOMPATIBLE_TARGET = ["@platforms//:incompatible"]

def _generator_srcs():
    srcs = list(_GENERATOR_INPUTS)
    if AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE and AMDGPU_CLANG_RESOURCE_HEADERS:
        srcs.append(AMDGPU_CLANG_RESOURCE_HEADERS)
    return srcs

def _generator_tools():
    tools = list(_GENERATOR_SCRIPT_DEPS)
    if AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE:
        tools.extend([
            AMDGPU_CLANG_TOOL,
            AMDGPU_LLVM_LINK_TOOL,
            AMDGPU_LLD_TOOL,
            AMDGPU_LLVM_OBJCOPY_TOOL,
        ])
        if AMDGPU_CLANG_RESOURCE_MARKER:
            tools.append(AMDGPU_CLANG_RESOURCE_MARKER)
    return tools

def _generator_command(code_object_target):
    if not AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE:
        return "echo '{}' >&2; false".format(AMDGPU_DEVICE_TOOLCHAIN_ERROR)

    return " ".join([
        "$(location %s)" % (_GENERATOR_SCRIPT,),
        "--repo-root .",
        "--binary-root $(BINDIR)",
        "--output-dir $(@D)",
        "--targets %s" % (code_object_target,),
        "--clang $(location %s)" % (AMDGPU_CLANG_TOOL,),
        "--llvm-link $(location %s)" % (AMDGPU_LLVM_LINK_TOOL,),
        "--lld $(location %s)" % (AMDGPU_LLD_TOOL,),
        "--llvm-objcopy $(location %s)" % (AMDGPU_LLVM_OBJCOPY_TOOL,),
    ] + AMDGPU_CLANG_RESOURCE_INCLUDE_ARGS)

def iree_hal_amdgpu_source_device_binaries():
    target_compatible_with = [] if AMDGPU_DEVICE_TOOLCHAIN_AVAILABLE else _INCOMPATIBLE_TARGET
    for code_object_target in IREE_HAL_AMDGPU_DEVICE_BINARY_CODE_OBJECT_TARGETS:
        native.genrule(
            name = "amdgcn-amd-amdhsa--%s" % (code_object_target,),
            srcs = _generator_srcs(),
            outs = ["amdgcn-amd-amdhsa--%s.so" % (code_object_target,)],
            cmd = _generator_command(code_object_target),
            target_compatible_with = target_compatible_with,
            tools = _generator_tools(),
            message = "Generating AMDGPU builtin device binary for %s..." % (code_object_target,),
        )
