# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for assembling IREE VM assembly modules."""

load("//build_tools/embed_data:build_defs.bzl", "iree_c_embed_data")

def iree_vmasm_module(
        name,
        src,
        module_name = None,
        assemble_tool = "//tools:iree-as-module",
        c_identifier = "",
        deps = [],
        **kwargs):
    """Builds an IREE bytecode module from VM assembly.

    Args:
        name: Name of the target.
        src: VM assembly source file to assemble.
        module_name: Optional output VMFB path. Defaults to `<name>.vmfb`.
        assemble_tool: Assembler tool used to produce the VMFB.
        c_identifier: Optional identifier for embedding the VMFB as C data.
        deps: Optional dependencies to add to the generated embed library.
        **kwargs: Additional attributes forwarded to generated rules.
    """

    if not module_name:
        module_name = "%s.vmfb" % (name)

    native.genrule(
        name = name,
        srcs = [src],
        outs = [module_name],
        cmd = " ".join([
            "$(location %s)" % (assemble_tool),
            "--output=$(location %s)" % (module_name),
            "$(location %s)" % (src),
        ]),
        tools = [assemble_tool],
        message = "Assembling IREE VM module %s..." % (name),
        output_to_bindir = 1,
        **kwargs
    )

    if c_identifier:
        iree_c_embed_data(
            name = "%s_c" % (name),
            identifier = c_identifier,
            srcs = [module_name],
            c_file_output = "%s_c.c" % (name),
            h_file_output = "%s_c.h" % (name),
            flatten = True,
            deps = deps,
            **kwargs
        )
