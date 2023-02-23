# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for compiling with clang to produce bitcode libraries."""

def iree_bitcode_library(
        name,
        srcs,
        hdrs = [],
        copts = [],
        defines = [],
        data = [],
        out = None,
        clang_tool = "@llvm-project//clang:clang",
        link_tool = "@llvm-project//llvm:llvm-link",
        builtin_headers_dep = "@llvm-project//clang:builtin_headers_gen",
        builtin_headers_path = "external/llvm-project/clang/staging/include/",
        **kwargs):
    """Builds an LLVM bitcode library from an input file via clang.

    Args:
        name: Name of the target.
        srcs: source files to pass to clang.
        hdrs: additional headers included by the source files.
        copts: additional flags to pass to clang.
        defines: preprocessor definitions to pass to clang.
        data: additional data required during compilation.
        out: output file name (defaults to name.bc).
        clang_tool: the clang to use to compile the source.
        link_tool: llvm-link tool used for linking bitcode files.
        builtin_headers_dep: clang builtin headers (stdbool, stdint, etc).
        builtin_headers_path: relative path to the builtin headers rule.
        **kwargs: any additional attributes to pass to the underlying rules.
    """

    bitcode_files = []
    for bitcode_src in srcs:
        bitcode_out = "%s_%s.bc" % (name, bitcode_src)
        bitcode_files.append(bitcode_out)
        native.genrule(
            name = "gen_%s" % (bitcode_out),
            srcs = [bitcode_src],
            outs = [bitcode_out],
            cmd = " && ".join([
                " ".join([
                    "$(location %s)" % (clang_tool),
                    "-isystem $(BINDIR)/%s" % (builtin_headers_path),
                    " ".join(copts),
                    " ".join(["-D%s" % (define) for define in defines]),
                    "-o $(location %s)" % (bitcode_out),
                    "$(location %s)" % (bitcode_src),
                ]),
            ]),
            tools = hdrs + data + [
                clang_tool,
                builtin_headers_dep,
            ],
            message = "Compiling %s to %s..." % (bitcode_src, bitcode_out),
            output_to_bindir = 1,
            **kwargs
        )

    if not out:
        out = "%s.bc" % (name)
    native.genrule(
        name = name,
        srcs = bitcode_files,
        outs = [out],
        cmd = " && ".join([
            " ".join([
                "$(location %s)" % (link_tool),
                "-o $(location %s)" % (out),
                " ".join(["$(locations %s)" % (src) for src in bitcode_files]),
            ]),
        ]),
        tools = data + [link_tool],
        message = "Linking bitcode library %s to %s..." % (name, out),
        output_to_bindir = 1,
        **kwargs
    )
