# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for compiling IREE HAL executables from MLIR sources.

HAL executables are backend-specific binaries produced by iree-compile in
hal-executable mode. Unlike full VM bytecode modules (iree_bytecode_module),
these contain only the device-side code for a single target backend.

Two compilation rules handle the actual iree-compile invocations:

    _iree_hal_executable   - Starlark rule: one MLIR source -> one .bin file.
    _iree_hal_executables  - Starlark rule: multiple MLIR sources -> multiple
                             .bin files. Accepts filegroups (label_list
                             expansion happens at analysis time, not loading
                             time), enabling cross-package file discovery.

Two macro wrappers compose the compilation rules with iree_c_embed_data:

    iree_hal_executable()  - compile + optional embed into a cc_library.
    iree_hal_executables() - batch compile + bundle into a single
                             iree_c_embed_data cc_library.

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
        srcs = ["//path/to/testdata:executable_srcs"],  # filegroup!
        target_device = "local",
        flags = ["--iree-hal-local-target-device-backends=vmvx"],
        identifier = "iree_cts_testdata_vmvx",
    )
"""

load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")

def _compile_hal_executable(ctx, src, output):
    """Runs iree-compile in hal-executable mode for a single source file.

    Shared action logic used by both _iree_hal_executable and
    _iree_hal_executables rules.
    """
    args = ctx.actions.args()
    args.add("--compile-mode=hal-executable")
    args.add("--iree-hal-target-device=" + ctx.attr.target_device)
    args.add("--mlir-print-op-on-diagnostic=false")
    args.add(ctx.executable._linker_tool,
             format = "--iree-llvmcpu-embedded-linker-path=%s")
    args.add(ctx.executable._linker_tool,
             format = "--iree-llvmcpu-wasm-linker-path=%s")
    for flag in ctx.attr.flags:
        args.add(flag)
    args.add("-o", output)
    args.add(src)

    ctx.actions.run(
        inputs = depset([src], transitive = [depset(ctx.files._linker_tool)]),
        outputs = [output],
        executable = ctx.executable._compile_tool,
        arguments = [args],
        mnemonic = "IreeCompileHalExecutable",
        progress_message = "Compiling HAL executable %%{label} (%s)" % src.basename,
    )

def _iree_hal_executable_impl(ctx):
    """Compiles a single MLIR source to a HAL executable binary."""
    src = ctx.file.src
    output = ctx.actions.declare_file(ctx.attr.name + ".bin")
    _compile_hal_executable(ctx, src, output)
    return [DefaultInfo(files = depset([output]))]

_iree_hal_executable = rule(
    implementation = _iree_hal_executable_impl,
    attrs = {
        "src": attr.label(mandatory = True, allow_single_file = [".mlir"]),
        "target_device": attr.string(mandatory = True),
        "flags": attr.string_list(),
        "_compile_tool": attr.label(
            default = "//tools:iree-compile",
            executable = True,
            cfg = "exec",
        ),
        "_linker_tool": attr.label(
            default = "@llvm-project//lld:lld",
            executable = True,
            cfg = "exec",
        ),
    },
)

def _iree_hal_executables_impl(ctx):
    """Compiles multiple MLIR sources to HAL executable binaries.

    Iterates ctx.files.srcs (which expands filegroups at analysis time)
    and creates one compile action per source. Outputs are placed under
    a {name}/ subdirectory to avoid collisions when multiple
    _iree_hal_executables targets coexist in the same package.
    """
    outputs = []
    linker_inputs = depset(ctx.files._linker_tool)
    for src in ctx.files.srcs:
        if src.basename.endswith(".mlir"):
            stem = src.basename[:-5]
        else:
            stem = src.basename
        output = ctx.actions.declare_file(ctx.attr.name + "/" + stem + ".bin")
        _compile_hal_executable(ctx, src, output)
        outputs.append(output)
    return [DefaultInfo(files = depset(outputs))]

_iree_hal_executables = rule(
    implementation = _iree_hal_executables_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = [".mlir"]),
        "target_device": attr.string(mandatory = True),
        "flags": attr.string_list(),
        "_compile_tool": attr.label(
            default = "//tools:iree-compile",
            executable = True,
            cfg = "exec",
        ),
        "_linker_tool": attr.label(
            default = "@llvm-project//lld:lld",
            executable = True,
            cfg = "exec",
        ),
    },
)

def iree_hal_executable(
        name,
        src,
        target_device,
        flags = [],
        c_identifier = "",
        deps = [],
        testonly = True,
        **kwargs):
    """Compiles an MLIR source to a HAL executable binary.

    Args:
        name: Name of the target. Output is {name}.bin.
        src: MLIR source file to compile.
        target_device: Target device for compilation (e.g., "local", "vulkan",
            "hip", "cuda", "metal"). Generates --iree-hal-target-device=<value>.
            For devices with sub-backends (like "local"), pass the backend
            selection flag via the flags parameter (e.g.,
            "--iree-hal-local-target-device-backends=vmvx").
        flags: Additional compiler flags beyond target device selection.
        c_identifier: If set, generates embedded C data via iree_c_embed_data.
        deps: Dependencies for the generated C library (when c_identifier is set).
        testonly: If True, only testonly targets can depend on this target.
            Defaults to True since HAL executables are primarily used for testing.
        **kwargs: Additional attributes (e.g., target_compatible_with)
            passed to the underlying rules.
    """
    _iree_hal_executable(
        name = name,
        src = src,
        target_device = target_device,
        flags = flags,
        testonly = testonly,
        **kwargs
    )

    if c_identifier:
        iree_c_embed_data(
            name = "%s_c" % (name),
            identifier = c_identifier,
            srcs = [":%s" % name],
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
        testonly = True,
        **kwargs):
    """Compiles multiple MLIR sources to HAL executables and bundles them.

    Batch form of iree_hal_executable(). Compiles each source to a HAL
    executable binary, then bundles all outputs into a single
    iree_c_embed_data() cc_library target.

    srcs can include filegroup labels — the underlying rule expands them
    at analysis time, enabling cross-package file discovery without
    requiring the caller to enumerate individual files.

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
        srcs: MLIR source file labels or filegroup labels to compile.
            Each source produces one executable binary in the embedded data.
        target_device: Target device for iree-compile (e.g., "local",
            "vulkan", "hip", "cuda").
        flags: Backend-specific compiler flags.
        identifier: C identifier for the generated embed data functions.
            Defaults to name. The generated header exposes
            {identifier}_create() and {identifier}_size().
        testonly: Defaults to True.
        **kwargs: Additional attributes (e.g., target_compatible_with)
            passed to the underlying rules.
    """
    if identifier == None:
        identifier = name

    compile_name = "%s_bin" % name
    _iree_hal_executables(
        name = compile_name,
        srcs = srcs,
        target_device = target_device,
        flags = flags,
        testonly = testonly,
        **kwargs
    )

    iree_c_embed_data(
        name = name,
        srcs = [":%s" % compile_name],
        c_file_output = "%s.c" % name,
        h_file_output = "%s.h" % name,
        identifier = identifier,
        flatten = True,
        testonly = testonly,
        **kwargs
    )
