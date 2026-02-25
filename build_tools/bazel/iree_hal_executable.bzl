# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rule for compiling IREE HAL executables from MLIR sources.

HAL executables are backend-specific binaries produced by iree-compile in
hal-executable mode. Unlike full VM bytecode modules (iree_bytecode_module),
these contain only the device-side code for a single target backend.

Typical usage:

    load("//build_tools/bazel:iree_hal_executable.bzl", "iree_hal_executable")

    # VMVX backend via the local device.
    iree_hal_executable(
        name = "dispatch_test_vmvx",
        src = "testdata/command_buffer_dispatch_test.mlir",
        target_device = "local",
        flags = [
            "--iree-hal-local-target-device-backends=vmvx",
        ],
    )

    # Vulkan backend (no sub-backend flag needed).
    iree_hal_executable(
        name = "dispatch_test_vulkan",
        src = "testdata/command_buffer_dispatch_test.mlir",
        target_device = "vulkan",
    )

The output can be chained with iree_c_embed_data() for embedding in test
binaries by setting the c_identifier parameter.
"""

load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")

def iree_hal_executable(
        name,
        src,
        target_device,
        flags = [],
        executable_name = None,
        compile_tool = "//tools:iree-compile",
        linker_tool = "@llvm-project//lld:lld",
        c_identifier = "",
        deps = [],
        testonly = True,
        **kwargs):
    """Compiles an MLIR source to a HAL executable binary.

    Args:
        name: Name of the target.
        src: MLIR source file to compile.
        target_device: Target device for compilation (e.g., "local", "vulkan",
            "hip", "cuda", "metal"). Generates --iree-hal-target-device=<value>.
            For devices with sub-backends (like "local"), pass the backend
            selection flag via the flags parameter (e.g.,
            "--iree-hal-local-target-device-backends=vmvx").
        flags: Additional compiler flags beyond target device selection.
        executable_name: Output filename. Defaults to `{name}.bin`.
        compile_tool: Compiler binary to use. Defaults to iree-compile.
        linker_tool: Linker binary for embedded ELF linking.
        c_identifier: If set, generates embedded C data via iree_c_embed_data.
        deps: Dependencies for the generated C library (when c_identifier is set).
        testonly: If True, only testonly targets can depend on this target.
            Defaults to True since HAL executables are primarily used for testing.
        **kwargs: Additional attributes passed to the underlying rules.
    """

    if not executable_name:
        executable_name = "%s.bin" % (name)

    out_files = [executable_name]
    all_flags = [
        "--compile-mode=hal-executable",
        "--iree-hal-target-device=%s" % (target_device),
        "--mlir-print-op-on-diagnostic=false",
    ] + list(flags)

    native.genrule(
        name = name,
        srcs = [src],
        outs = out_files,
        cmd = " ".join([
            "$(location %s)" % (compile_tool),
            " ".join(all_flags),
            "--iree-llvmcpu-embedded-linker-path=$(location %s)" % (linker_tool),
            "--iree-llvmcpu-wasm-linker-path=$(location %s)" % (linker_tool),
            "-o $(location %s)" % (executable_name),
            "$(location %s)" % (src),
        ]),
        tools = [compile_tool, linker_tool],
        message = "Compiling HAL executable %s..." % (name),
        output_to_bindir = 1,
        testonly = testonly,
        **kwargs
    )

    if c_identifier:
        iree_c_embed_data(
            name = "%s_c" % (name),
            identifier = c_identifier,
            srcs = [executable_name],
            c_file_output = "%s_c.c" % (name),
            h_file_output = "%s_c.h" % (name),
            flatten = True,
            testonly = testonly,
            deps = deps,
            **kwargs
        )
