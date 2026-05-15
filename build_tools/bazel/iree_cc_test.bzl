# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Platform-adaptive test rule.

iree_cc_test wraps a cc_binary as a test target. On native platforms it
symlinks to the binary. On wasm32 platforms it bundles the .wasm with a
WASI test entry point and runs via Node.js.

On wasm32, the entry point is determined by:
  1. If an iree_wasm_entry target is found in cc_deps, its main JS file
     is used as the entry point (with its srcs as bundler sandbox inputs).
  2. Otherwise, the default wasm_test_main.mjs generic WASI harness is used.
"""

load(
    "//build_tools/bazel:iree_wasm_library.bzl",
    "IreeWasmEntryCollectionInfo",
    "collect_and_bundle",
    "collect_wasm_js",
)

def _iree_cc_test_impl(ctx):
    is_wasm = ctx.target_platform_has_constraint(
        ctx.attr._wasm32_constraint[platform_common.ConstraintValueInfo],
    )

    binary_info = ctx.attr.binary[DefaultInfo]
    binary_executable = binary_info.files_to_run.executable

    # Collect data files into runfiles.
    data_runfiles = ctx.runfiles(files = ctx.files.data)

    # Build environment dict, expanding $(rootpath) for data labels.
    env = {}
    for key, value in ctx.attr.env.items():
        env[key] = ctx.expand_location(value, ctx.attr.data)

    if not is_wasm:
        # Native: symlink to the cc_binary executable.
        executable = ctx.actions.declare_file(ctx.label.name)
        ctx.actions.symlink(
            output = executable,
            target_file = binary_executable,
        )
        runfiles = ctx.runfiles().merge(binary_info.default_runfiles).merge(data_runfiles)
        return [
            DefaultInfo(
                executable = executable,
                runfiles = runfiles,
            ),
            testing.TestEnvironment(env),
        ]

    # Wasm: discover entry point from deps, falling back to the generic
    # WASI test harness if no iree_wasm_entry target is present.
    main_js = None
    main_srcs = []
    for dep in ctx.attr.cc_deps:
        if IreeWasmEntryCollectionInfo in dep:
            for entry in dep[IreeWasmEntryCollectionInfo].entries.to_list():
                if main_js != None:
                    fail("Multiple iree_wasm_entry targets found in deps " +
                         "of %s; expected at most one" % ctx.label)
                main_js = entry.main
                main_srcs = list(entry.srcs)

    if main_js == None:
        main_js = ctx.file._test_main

    output_mjs = collect_and_bundle(
        ctx = ctx,
        wasm_binary = binary_executable,
        main_js = main_js,
        cc_deps = ctx.attr.cc_deps,
        bundler = ctx.executable._bundler,
        main_srcs = main_srcs,
    )

    # Shell wrapper: locates the bundled .mjs in the runfiles tree and
    # executes it via node. Test arguments are forwarded.
    executable = ctx.actions.declare_file(ctx.label.name)
    wrapper_content = (
        "#!/bin/bash\n" +
        'RUNFILES="${{RUNFILES_DIR:-$0.runfiles}}"\n' +
        'exec node "$RUNFILES/{workspace}/{mjs_path}" "$@"\n'
    ).format(
        workspace = ctx.workspace_name,
        mjs_path = output_mjs.short_path,
    )
    ctx.actions.write(
        output = executable,
        content = wrapper_content,
        is_executable = True,
    )

    runfiles = ctx.runfiles(
        files = [binary_executable, output_mjs],
    ).merge(binary_info.default_runfiles)
    return [DefaultInfo(
        executable = executable,
        runfiles = runfiles,
    )]

iree_cc_test = rule(
    implementation = _iree_cc_test_impl,
    test = True,
    attrs = {
        "binary": attr.label(
            mandatory = True,
            providers = [DefaultInfo],
            doc = "The cc_binary target to wrap as a test.",
        ),
        "cc_deps": attr.label_list(
            aspects = [collect_wasm_js],
            doc = "Same deps as the cc_binary (for wasm JS companion collection).",
        ),
        "data": attr.label_list(
            allow_files = True,
            doc = "Data files available to the test at runtime.",
        ),
        "env": attr.string_dict(
            doc = "Environment variables set for the test.",
        ),
        "_wasm32_constraint": attr.label(
            default = "@platforms//cpu:wasm32",
        ),
        "_test_main": attr.label(
            default = "//build_tools/wasm:wasm_test_main.mjs",
            allow_single_file = True,
        ),
        "_bundler": attr.label(
            default = "//build_tools/wasm:wasm_binary_bundler",
            executable = True,
            cfg = "exec",
        ),
    },
    doc = "Platform-adaptive test rule. On native platforms, runs the " +
          "cc_binary directly. On wasm32 platforms, bundles with a WASI " +
          "test entry point and runs via Node.js. If an iree_wasm_entry " +
          "target is found in deps, it overrides the default entry point.",
)
