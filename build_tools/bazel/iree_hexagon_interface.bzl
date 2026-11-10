# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for generating Hexagon FastRPC interfaces with the qaic IDL compiler.

See build_tools/cmake/iree_hexagon_interface.cmake for the CMake equivalent.
"""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

# The Hexagon driver runs on Android aarch64 devices. Restricting all targets
# to that platform keeps host builds (e.g. `bazel build //...`) from fetching
# the Hexagon SDK.
_TARGET_COMPATIBLE_WITH = [
    "@platforms//cpu:aarch64",
    "@platforms//os:android",
]

def _iree_hexagon_qaic_impl(ctx):
    include_dirs = []
    for include in ctx.files._qaic_includes:
        include_dir = include.dirname
        if include_dir not in include_dirs:
            include_dirs.append(include_dir)

    basename = ctx.file.idl.basename.removesuffix(".idl")
    header = ctx.actions.declare_file("{}/{}.h".format(ctx.attr.out_dir, basename))
    stub = ctx.actions.declare_file("{}/{}_stub.c".format(ctx.attr.out_dir, basename))
    skel = ctx.actions.declare_file("{}/{}_skel.c".format(ctx.attr.out_dir, basename))
    ctx.actions.run(
        executable = ctx.file._qaic,
        arguments = ["-mdll", "-o", header.dirname] +
                    ["-I" + include_dir for include_dir in include_dirs] +
                    [ctx.file.idl.path],
        inputs = [ctx.file.idl] + ctx.files._qaic_includes,
        outputs = [header, stub, skel],
        mnemonic = "HexagonQaic",
        progress_message = "Generating Hexagon RPC interface from %{input}",
    )
    return [
        DefaultInfo(files = depset([header, stub, skel])),
        OutputGroupInfo(
            header = depset([header]),
            stub = depset([stub]),
            skel = depset([skel]),
        ),
    ]

_iree_hexagon_qaic = rule(
    implementation = _iree_hexagon_qaic_impl,
    attrs = {
        "idl": attr.label(allow_single_file = [".idl"], mandatory = True),
        "out_dir": attr.string(mandatory = True),
        "_qaic": attr.label(
            default = "@hexagon_sdk//:qaic",
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
        "_qaic_includes": attr.label(
            default = "@hexagon_sdk//:qaic_includes",
            allow_files = True,
        ),
    },
)

def iree_hexagon_interface(name, idl, **kwargs):
    """Generates a Hexagon FastRPC interface from an IDL file.

    Creates the following targets:
      name: cc_library with the generated header and the client stub for
            Android aarch64 (the host calling into the DSP).
      name_skel_srcs: filegroup with the generated header and DSP skeleton
                      source, consumed by iree_hexagon_dsp_library.

    Args:
        name: Name of the interface; also names the output directory.
        idl: The *.idl interface definition.
        **kwargs: Additional attributes passed to all generated targets.
    """
    kwargs.setdefault("target_compatible_with", _TARGET_COMPATIBLE_WITH)
    _iree_hexagon_qaic(
        name = name + "_gen",
        idl = idl,
        out_dir = name,
        **kwargs
    )
    for group in ["header", "stub", "skel"]:
        native.filegroup(
            name = "{}_{}".format(name, group),
            srcs = [":{}_gen".format(name)],
            output_group = group,
            **kwargs
        )
    native.filegroup(
        name = name + "_skel_srcs",
        srcs = [
            ":{}_header".format(name),
            ":{}_skel".format(name),
        ],
        **kwargs
    )
    cc_library(
        name = name,
        srcs = [":{}_stub".format(name)],
        hdrs = [":{}_header".format(name)],
        # The generated code is not warning-free.
        copts = ["-w"],
        includes = [name],
        deps = ["@hexagon_sdk//:cdsprpc_android_aarch64"],
        **kwargs
    )
