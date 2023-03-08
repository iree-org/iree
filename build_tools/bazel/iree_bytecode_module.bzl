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
        flags,
        module_name = None,
        compile_tool = "//tools:iree-compile",
        linker_tool = "@llvm-project//lld:lld",
        c_identifier = "",
        static_lib_path = "",
        deps = [],
        **kwargs):
    """Builds an IREE bytecode module.

    Args:
        name: Name of the target
        src: mlir source file to be compiled to an IREE module.
        flags: additional flags to pass to the compiler.
            `--output-format=vm-bytecode` is included automatically.
        module_name: Optional name for the generated IREE module.
            Defaults to `name.vmfb`.
        compile_tool: the compiler to use to generate the module.
            Defaults to iree-compile.
        linker_tool: the linker to use.
            Defaults to the lld from the llvm-project directory.
        c_identifier: Optional. Enables embedding the module as C data.
        static_lib_path: When set, the module is compiled into a LLVM static
            library with the specified library path.
        deps: Optional. Dependencies to add to the generated library.
        **kwargs: any additional attributes to pass to the underlying rules.
    """

    if not module_name:
        module_name = "%s.vmfb" % (name)

    out_files = [module_name]
    flags.append("--output-format=vm-bytecode")
    if static_lib_path:
        static_header_path = static_lib_path.replace(".o", ".h")
        out_files.extend([static_lib_path, static_header_path])
        flags += [
            "--iree-llvmcpu-link-embedded=false",
            "--iree-llvmcpu-link-static",
            "--iree-llvmcpu-static-library-output-path=$(location %s)" % (static_lib_path),
        ]

    native.genrule(
        name = name,
        srcs = [src],
        outs = out_files,
        cmd = " && ".join([
            " ".join([
                "$(location %s)" % (compile_tool),
                " ".join(flags),
                "--iree-llvmcpu-embedded-linker-path=$(location %s)" % (linker_tool),
                "--iree-llvmcpu-wasm-linker-path=$(location %s)" % (linker_tool),
                # Note: --iree-llvmcpu-system-linker-path is left unspecified.
                "-o $(location %s)" % (module_name),
                "$(location %s)" % (src),
            ]),
        ]),
        tools = [compile_tool, linker_tool],
        message = "Compiling IREE module %s..." % (name),
        output_to_bindir = 1,
        **kwargs
    )

    # Embed the module for use in C.
    if c_identifier:
        c_embed_data(
            name = "%s_c" % (name),
            identifier = c_identifier,
            srcs = [module_name],
            c_file_output = "%s_c.c" % (name),
            h_file_output = "%s_c.h" % (name),
            flatten = True,
            deps = deps,
            **kwargs
        )
