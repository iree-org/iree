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

Flags can contain {PLACEHOLDER} template variables resolved via the
flag_values dict. Each entry maps a placeholder name to a label.
Resolution is determined by the target type:

    Build settings (string_flag) — replaced with the setting's value.
    File targets — the flag is repeated for each output file, with
        the placeholder replaced by the file's path. For directory
        targets (TreeArtifacts via iree_directory), this produces
        a single flag with the directory path.

Example:

    iree_hal_executables(
        name = "testdata_amdgpu",
        srcs = ["//path/to/testdata:executable_srcs"],
        target_device = "amdgpu",
        flags = [
            "--iree-rocm-target={ROCM_TARGET}",
            "--iree-rocm-bc-dir={ROCM_BC_DIR}",
        ],
        flag_values = {
            "ROCM_TARGET": "//build_tools/bazel:rocm_test_target",
            "ROCM_BC_DIR": "@amdgpu_device_libs//:bitcode",
        },
    )

Override build settings at build time:
    --//build_tools/bazel:rocm_test_target=gfx942
"""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")

def _resolve_flags(ctx):
    """Resolves {PLACEHOLDER} template variables in flags via flag_values.

    Each flag_values entry maps a placeholder name to a target label.
    Resolution depends on the target type:

      Build settings (BuildSettingInfo) — the setting's string value
          replaces {PLACEHOLDER} in each flag.
      File targets — each flag containing {PLACEHOLDER} is repeated
          for every output file, with the placeholder replaced by
          the file's path. For TreeArtifact targets (iree_directory),
          this produces a single flag with the directory path.
    """
    flags = list(ctx.attr.flags)
    for target, placeholder in ctx.attr.flag_values.items():
        template = "{%s}" % placeholder
        if BuildSettingInfo in target:
            value = target[BuildSettingInfo].value
            flags = [f.replace(template, value) for f in flags]
        else:
            files = target.files.to_list()
            if not files:
                fail("{%s} references a label that produces no files" %
                     placeholder)
            expanded = []
            for flag in flags:
                if template in flag:
                    for f in files:
                        expanded.append(flag.replace(template, f.path))
                else:
                    expanded.append(flag)
            flags = expanded
    return flags

def _compile_hal_executable(ctx, src, output, flags):
    """Runs iree-compile in hal-executable mode for a single source file.

    Shared action logic used by both _iree_hal_executable and
    _iree_hal_executables rules.

    Args:
        ctx: Rule context (for tools and action factory).
        src: Source File to compile.
        output: Output File to produce.
        flags: Resolved compiler flags (template variables already expanded).
    """
    args = ctx.actions.args()
    args.add("--compile-mode=hal-executable")
    args.add("--iree-hal-target-device=" + ctx.attr.target_device)
    args.add("--mlir-print-op-on-diagnostic=false")
    args.add(
        ctx.executable._linker_tool,
        format = "--iree-llvmcpu-embedded-linker-path=%s",
    )
    args.add(
        ctx.executable._linker_tool,
        format = "--iree-llvmcpu-wasm-linker-path=%s",
    )
    for flag in flags:
        args.add(flag)
    args.add("-o", output)
    args.add(src)

    transitive_inputs = [depset(ctx.files._linker_tool)]
    if ctx.files.data:
        transitive_inputs.append(depset(ctx.files.data))
    for target in ctx.attr.flag_values:
        if BuildSettingInfo not in target:
            transitive_inputs.append(target.files)

    # Set PATH so that backends using findTool() (e.g., ROCM's search for
    # iree-lld/lld) can locate the linker in the sandbox.
    env = {"PATH": ctx.executable._linker_tool.dirname}

    ctx.actions.run(
        inputs = depset([src], transitive = transitive_inputs),
        outputs = [output],
        executable = ctx.executable._compile_tool,
        arguments = [args],
        env = env,
        mnemonic = "IreeCompileHalExecutable",
        progress_message = "Compiling HAL executable %%{label} (%s)" % src.basename,
    )

def _iree_hal_executable_impl(ctx):
    """Compiles a single MLIR source to a HAL executable binary."""
    src = ctx.file.src
    output = ctx.actions.declare_file(ctx.attr.name + ".bin")
    _compile_hal_executable(ctx, src, output, _resolve_flags(ctx))
    return [DefaultInfo(files = depset([output]))]

_COMMON_ATTRS = {
    "target_device": attr.string(mandatory = True),
    "flags": attr.string_list(),
    "flag_values": attr.label_keyed_string_dict(
        allow_files = True,
    ),
    "data": attr.label_list(allow_files = True),
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
}

_iree_hal_executable = rule(
    implementation = _iree_hal_executable_impl,
    attrs = dict(
        _COMMON_ATTRS,
        src = attr.label(mandatory = True, allow_single_file = [".mlir"]),
    ),
)

def _iree_hal_executables_impl(ctx):
    """Compiles multiple MLIR sources to HAL executable binaries.

    Iterates ctx.files.srcs (which expands filegroups at analysis time)
    and creates one compile action per source. Outputs are placed under
    a {name}/ subdirectory to avoid collisions when multiple
    _iree_hal_executables targets coexist in the same package.
    """
    flags = _resolve_flags(ctx)
    outputs = []
    for src in ctx.files.srcs:
        if src.basename.endswith(".mlir"):
            stem = src.basename[:-5]
        else:
            stem = src.basename
        output = ctx.actions.declare_file(ctx.attr.name + "/" + stem + ".bin")
        _compile_hal_executable(ctx, src, output, flags)
        outputs.append(output)
    return [DefaultInfo(files = depset(outputs))]

_iree_hal_executables = rule(
    implementation = _iree_hal_executables_impl,
    attrs = dict(
        _COMMON_ATTRS,
        srcs = attr.label_list(allow_files = [".mlir"]),
    ),
)

def iree_hal_executable(
        name,
        src,
        target_device,
        flags = [],
        flag_values = {},
        data = [],
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
            May contain {PLACEHOLDER} template variables resolved from
            flag_values. See module docstring for resolution semantics.
        flag_values: Dict mapping placeholder names to target labels.
            Build settings (string_flag) resolve to their string value.
            File targets resolve to their output paths, repeating the
            flag for each file. iree_directory targets (TreeArtifacts)
            resolve to the directory path. Example:
                flag_values = {
                    "ROCM_TARGET": "//build_tools/bazel:rocm_test_target",
                    "ROCM_BC_DIR": "@amdgpu_device_libs//:bitcode",
                }
        data: Additional files to include in the compile action's inputs.
        c_identifier: If set, generates embedded C data via iree_c_embed_data.
        deps: Dependencies for the generated C library (when c_identifier is set).
        testonly: If True, only testonly targets can depend on this target.
            Defaults to True since HAL executables are primarily used for testing.
        **kwargs: Additional attributes (e.g., target_compatible_with)
            passed to the underlying rules.
    """

    # Invert flag_values: user writes {"PLACEHOLDER": "//label"} (readable),
    # rule attr is label_keyed_string_dict so needs {"//label": "PLACEHOLDER"}.
    rule_flag_values = {v: k for k, v in flag_values.items()}
    _iree_hal_executable(
        name = name,
        src = src,
        target_device = target_device,
        flags = flags,
        flag_values = rule_flag_values,
        data = data,
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
        flag_values = {},
        data = [],
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
        flags: Backend-specific compiler flags. May contain {PLACEHOLDER}
            template variables resolved from flag_values.
        flag_values: Dict mapping placeholder names to target labels.
            See iree_hal_executable() for details.
        data: Additional files to include in compile action inputs.
        identifier: C identifier for the generated embed data functions.
            Defaults to name. The generated header exposes
            {identifier}_create() and {identifier}_size().
        testonly: Defaults to True.
        **kwargs: Additional attributes (e.g., target_compatible_with)
            passed to the underlying rules.
    """
    if identifier == None:
        identifier = name

    # Invert flag_values: user writes {"PLACEHOLDER": "//label"} (readable),
    # rule attr is label_keyed_string_dict so needs {"//label": "PLACEHOLDER"}.
    rule_flag_values = {v: k for k, v in flag_values.items()}
    compile_name = "%s_bin" % name
    _iree_hal_executables(
        name = compile_name,
        srcs = srcs,
        target_device = target_device,
        flags = flags,
        flag_values = rule_flag_values,
        data = data,
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
