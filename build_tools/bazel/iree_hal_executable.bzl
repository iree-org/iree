# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for compiling IREE HAL executables from MLIR sources.

HAL executables are backend-specific binaries produced by iree-compile in
hal-executable mode. Unlike full VM bytecode modules (iree_bytecode_module),
these contain only the device-side code for a single target backend.

Two rules are provided:

    iree_hal_executable()   - compile a single MLIR source to one binary.
    iree_hal_executables()  - compile multiple MLIR sources and bundle into
                              a single iree_c_embed_data() cc_library.

Typical usage (single file):

    load("//build_tools/bazel:iree_hal_executable.bzl", "iree_hal_executable")

    iree_hal_executable(
        name = "dispatch_test_vmvx",
        src = "testdata/dispatch_test.mlir",
        target_device = "local",
        flags = ["--iree-hal-local-target-device-backends=vmvx"],
    )

Typical usage (batch compile + embed):

    load("//build_tools/bazel:iree_hal_executable.bzl", "iree_hal_executables")

    # Each call compiles all sources for one format and bundles them into
    # an iree_c_embed_data cc_library. Multiple calls in the same package
    # (for different formats) are safe — outputs are namespaced by target
    # name to avoid collisions.
    iree_hal_executables(
        name = "testdata_vmvx",
        srcs = ["//path/to:dispatch.mlir", "//path/to:compute.mlir"],
        target_device = "local",
        flags = ["--iree-hal-local-target-device-backends=vmvx"],
        identifier = "iree_cts_testdata_vmvx",
    )

    iree_hal_executables(
        name = "testdata_llvm_cpu",
        srcs = ["//path/to:dispatch.mlir", "//path/to:compute.mlir"],
        target_device = "local",
        flags = ["--iree-hal-local-target-device-backends=llvm-cpu"],
        identifier = "iree_cts_testdata_llvm_cpu",
    )
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

def iree_hal_executables(
        name,
        srcs,
        target_device,
        flags = [],
        identifier = None,
        compile_tool = "//tools:iree-compile",
        linker_tool = "@llvm-project//lld:lld",
        testonly = True,
        **kwargs):
    """Compiles multiple MLIR sources to HAL executables and bundles them.

    Batch form of iree_hal_executable(). For each source in srcs, compiles
    it to a HAL executable binary, then bundles all outputs into a single
    iree_c_embed_data() cc_library target.

    TOC entry names are the source stems with a .bin extension (e.g.,
    "dispatch.mlir" produces TOC entry "dispatch.bin"). This is independent
    of the format being compiled, so the same test code can look up
    executables by stem regardless of which backend compiled them.

    Multiple calls in the same package (for different formats) are safe:
    each call's compiled outputs are placed in a {name}/ subdirectory to
    avoid filename collisions, and flatten=True on the embedded data strips
    the directory prefix from TOC entries.

    Args:
        name: Name of the output iree_c_embed_data cc_library target.
        srcs: MLIR source file labels to compile. Each produces one
            executable binary in the embedded data.
        target_device: Target device for iree-compile (e.g., "local",
            "vulkan", "hip", "cuda").
        flags: Backend-specific compiler flags.
        identifier: C identifier for the generated embed data functions.
            Defaults to name. The generated header exposes
            {identifier}_create() and {identifier}_size().
        compile_tool: Compiler binary to use.
        linker_tool: Linker binary for embedded ELF linking.
        testonly: Defaults to True.
        **kwargs: Additional attributes (e.g., target_compatible_with)
            passed to the underlying iree_hal_executable() and
            iree_c_embed_data() rules.
    """
    if identifier == None:
        identifier = name

    bin_outputs = []
    for src in srcs:
        # Derive the stem from the source label.
        # "//pkg:foo.mlir" -> "foo", "//pkg:bar/foo.mlir" -> "bar/foo".
        src_str = str(src)
        if ":" in src_str:
            file_part = src_str.rsplit(":", 1)[-1]
        else:
            file_part = src_str.rsplit("/", 1)[-1]
        if file_part.endswith(".mlir"):
            stem = file_part[:-5]
        else:
            stem = file_part

        # Outputs are placed under a {name}/ subdirectory so that multiple
        # iree_hal_executables() calls in the same package (for different
        # formats) produce unique output paths. With flatten=True on the
        # embed data, TOC entries use just "{stem}.bin".
        executable_name = "%s/%s.bin" % (name, stem)
        rule_name = "%s_%s" % (name, stem)

        iree_hal_executable(
            name = rule_name,
            src = src,
            executable_name = executable_name,
            target_device = target_device,
            flags = flags,
            compile_tool = compile_tool,
            linker_tool = linker_tool,
            testonly = testonly,
            **kwargs
        )
        bin_outputs.append(":%s" % executable_name)

    iree_c_embed_data(
        name = name,
        srcs = sorted(bin_outputs),
        c_file_output = "%s.c" % name,
        h_file_output = "%s.h" % name,
        identifier = identifier,
        flatten = True,
        testonly = testonly,
        **kwargs
    )
