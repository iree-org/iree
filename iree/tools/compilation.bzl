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
        c_identifier = "",
        **kwargs):
    native.genrule(
        name = name,
        srcs = [src],
        outs = [
            "%s.vmfb" % (name),
        ],
        cmd = " && ".join([
            " ".join([
                "$(location %s)" % (translate_tool),
                " ".join(flags),
                "-o $(location %s.vmfb)" % (name),
                "$(location %s)" % (src),
            ]),
        ]),
        tools = [translate_tool],
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
