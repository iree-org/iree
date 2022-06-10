# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for compiling IREEi C modules."""

load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_library")

def iree_c_module(
        name,
        src,
        h_file_output,
        flags = [
          "--compile-mode=vm",
          "--output-format=vm-c",
        ],
        deps = [
          "//runtime/src/iree/vm",
          "//runtime/src/iree/vm:ops",
          "//runtime/src/iree/vm:ops_emitc",
          "//runtime/src/iree/vm:shims_emitc",
        ],
        # TODO: Rename this to 'compile_tool'
        translate_tool = "//tools:iree-compile",
        **kwargs):
    """Builds an IREE C module.

    Args:
        name: Name of the target
        src: mlir source file to be compiled to an IREE module.
        h_file_output: The H header file to output.
        flags: additional flags to pass to the compiler. Bytecode
            translation and backend flags are passed automatically.
        deps: Optional. Dependencies to add to the generated library.
        translate_tool: the compiler to use to generate the module.
            Defaults to iree-compile.
        **kwargs: any additional attributes to pass to the underlying rules.
    """

    native.genrule(
        name = name + "_gen",
        srcs = [src],
        outs = [h_file_output],
        cmd = " && ".join([
            " ".join([
                "$(location %s)" % (translate_tool),
                " ".join(flags),
                "-o $(location %s)" % (h_file_output),
                "$(location %s)" % (src),
            ]),
        ]),
        tools = [translate_tool],
        message = "Generating C module  %s..." % (name),
        output_to_bindir = 1,
        **kwargs
    )
    iree_runtime_cc_library(
      name = name,
      hdrs = [h_file_output],
      srcs = ["//runtime/src/iree/vm:module_impl_emitc.c", h_file_output],
      copts = [
        "-DEMITC_IMPLEMENTATION='\"$(location %s)\"'" % h_file_output,
      ],
      deps = deps,
      **kwargs
    )


