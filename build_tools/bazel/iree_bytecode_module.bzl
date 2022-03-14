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
        module = None,
        flags = ["-iree-mlir-to-vm-bytecode-module"],
        translate_tool = "//iree/tools:iree-translate",
        linker_tool = "@llvm-project//lld:lld",
        opt_tool = "//iree/tools:iree-opt",
        opt_flags = [],
        c_identifier = "",
        **kwargs):
    """Builds an IREE bytecode module.

    Args:
        name: Name of the target
        src: mlir source file to be compiled to an IREE module.
        flags: additional flags to pass to the compiler. Bytecode
            translation and backend flags are passed automatically.
        translate_tool: the compiler to use to generate the module.
            Defaults to iree-translate.
        linker_tool: the linker to use.
            Defaults to the lld from the llvm-project directory.
        opt_tool: Defaulting to iree-opt. Tool used to preprocess the source file
            if opt_flags is specified.
        opt_flags: If specified, source files are preprocessed with opt_tool with
            these flags.
        module: Optional. Specifies the  path to use for the enerated IREE module (.vmfb).
        c_identifier: Optional. Enables embedding the module as C data.
        **kwargs: any additional attributes to pass to the underlying rules.
    """

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
            **kwargs
        )

    if not module:
        module = "%s.vmfb" % (name)

    native.genrule(
        name = name,
        srcs = [translate_src],
        outs = [
            module,
        ],
        cmd = " && ".join([
            " ".join([
                "$(location %s)" % (translate_tool),
                " ".join(flags),
                "-iree-llvm-embedded-linker-path=$(location %s)" % (linker_tool),
                "-iree-llvm-system-linker-path=$(location %s)" % (linker_tool),
                "-o $(location %s)" % (module),
                "$(location %s)" % (translate_src),
            ]),
        ]),
        tools = [translate_tool, linker_tool],
        message = "Compiling IREE module %s..." % (name),
        output_to_bindir = 1,
        **kwargs
    )

    # Embed the module for use in C.
    if c_identifier:
        c_embed_data(
            name = "%s_c" % (name),
            identifier = c_identifier,
            srcs = [module],
            c_file_output = "%s_c.c" % (name),
            h_file_output = "%s_c.h" % (name),
            flatten = True,
            **kwargs
        )
