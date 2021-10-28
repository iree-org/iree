# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for compiling IREE executables, modules, and archives."""

load("//build_tools/embed_data:build_defs.bzl", "c_embed_data")

# TODO(benvanik): port to a full starlark rule, document, etc.
def iree_bytecode_module(
        name,
        src,
        flags = ["-iree-mlir-to-vm-bytecode-module"],
        translate_tool = "//iree/tools:iree-translate",
        embedded_linker_tool = "@llvm-project//lld:lld",
        opt_tool = "//iree/tools:iree-opt",
        opt_flags = [],
        c_identifier = "",
        **kwargs):
    translate_src = src
    if opt_flags:
        translate_src = "%s.opt.mlir" % (name)
        native.genrule(
            name = "%s_opt" % (name),
            srcs = [src],
            outs = [translate_src],
            cmd = " ".join([
                "$(location %s)" % (opt_tool),
                " ".join([('"%s"' % flag) for flag in opt_flags]),
                "$(location %s)" % (src),
                "-o $(location %s)" % (translate_src),
            ]),
            tools = [opt_tool],
            message = "Transforming MLIR source for IREE module %s..." % (name),
            output_to_bindir = 1,
        )

    native.genrule(
        name = name,
        srcs = [translate_src],
        outs = [
            "%s.vmfb" % (name),
        ],
        cmd = " && ".join([
            " ".join([
                "$(location %s)" % (translate_tool),
                " ".join(flags),
                "-iree-llvm-embedded-linker-path=$(location %s)" % (embedded_linker_tool),
                "-o $(location %s.vmfb)" % (name),
                "$(location %s)" % (translate_src),
            ]),
        ]),
        tools = [translate_tool, embedded_linker_tool],
        message = "Compiling IREE module %s..." % (name),
        output_to_bindir = 1,
        **kwargs
    )

    # Embed the module for use in C.
    if c_identifier:
        c_embed_data(
            name = "%s_c" % (name),
            identifier = c_identifier,
            srcs = ["%s.vmfb" % (name)],
            c_file_output = "%s_c.c" % (name),
            h_file_output = "%s_c.h" % (name),
            flatten = True,
            **kwargs
        )
