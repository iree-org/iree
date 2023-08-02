# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for compiling IREE C modules."""

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library")

def iree_c_module(
        name,
        src,
        h_file_output,
        flags,
        deps = [
            "//runtime/src/iree/vm",
            "//runtime/src/iree/vm:ops",
            "//runtime/src/iree/vm:ops_emitc",
            "//runtime/src/iree/vm:shims_emitc",
        ],
        compile_tool = "//tools:iree-compile",
        no_runtime = None,
        static_lib_path = "",
        **kwargs):
    """Builds an IREE C module.

    Args:
        name: Name of the target
        src: mlir source file to be compiled to an IREE module.
        h_file_output: The H header file to output.
        flags: additional flags to pass to the compile tool.
            `--output-format=vm-c` is included automatically.
        deps: Optional. Dependencies to add to the generated library.
        compile_tool: the compiler to use to generate the module.
            Defaults to iree-compile.
        static_lib_path: When set, the module is compiled into a LLVM static
            library with the specified library path.
        no_runtime: When set, this target will be built without the
            runtime library support.
        **kwargs: any additional attributes to pass to the underlying rules.
    """

    out_files = [h_file_output]
    flags.append("--output-format=vm-c")
    if static_lib_path:
        static_header_path = static_lib_path.replace(".o", ".h")
        out_files.extend([static_lib_path, static_header_path])
        flags += [
            "--iree-llvmcpu-link-embedded=false",
            "--iree-llvmcpu-link-static",
            "--iree-llvmcpu-static-library-output-path=$(location %s)" % (static_lib_path),
        ]

    native.genrule(
        name = name + "_gen",
        srcs = [src],
        outs = out_files,
        cmd = " && ".join([
            " ".join([
                "$(location %s)" % (compile_tool),
                " ".join(flags),
                "-o $(location %s)" % (h_file_output),
                "$(location %s)" % (src),
            ]),
        ]),
        tools = [compile_tool],
        message = "Generating C module  %s..." % (name),
        output_to_bindir = 1,
        **kwargs
    )

    deps_list = None
    if not no_runtime:
        deps_list = deps

    iree_runtime_cc_library(
        name = name,
        hdrs = [h_file_output],
        srcs = ["//runtime/src/iree/vm:module_impl_emitc.c", h_file_output],
        copts = [
            "-DEMITC_IMPLEMENTATION='\"$(location %s)\"'" % h_file_output,
        ],
        deps = deps_list,
        **kwargs
    )
