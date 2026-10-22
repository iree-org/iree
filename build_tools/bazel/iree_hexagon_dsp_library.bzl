# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for building shared libraries running on the Hexagon DSP.

The Hexagon SDK compiler is invoked directly (similar to iree_amdgpu_binary)
instead of via a cc_toolchain, as the libraries only contain self-contained C
sources. See build_tools/cmake/iree_hexagon_dsp_library.cmake for the CMake
equivalent.
"""

def iree_hexagon_dsp_library(
        name,
        srcs,
        interface,
        arch = "v79",
        copts = [],
        linkopts = [],
        **kwargs):
    """Builds a FastRPC skeleton shared library (lib<name>.so) for the DSP.

    Args:
        name: Name of the target; the output is lib<name>.so.
        srcs: C sources and headers implementing the interface.
        interface: iree_hexagon_interface target whose skeleton is linked in.
        arch: Hexagon architecture version (e.g. v79).
        copts: Additional flags passed to hexagon-clang when compiling.
        linkopts: Additional flags passed to hexagon-clang when linking.
        **kwargs: Additional attributes passed to the genrule.
    """

    # The library is deployed with the Hexagon driver on Android aarch64
    # devices. Restricting it to that platform keeps host builds (e.g.
    # `bazel build //...`) from fetching the Hexagon SDK.
    kwargs.setdefault("target_compatible_with", [
        "@platforms//cpu:aarch64",
        "@platforms//os:android",
    ])
    clang = "@hexagon_sdk//:hexagon_clang"
    skel_srcs = interface + "_skel_srcs"
    out = "lib{}.so".format(name)

    # The SDK root is 5 levels above tools/HEXAGON_Tools/<ver>/Tools/bin.
    sdk_dir = "$$(dirname $(location {}))/../../../../..".format(clang)
    base_copts = [
        "-m" + arch,
        "-G0",
        "-fpic",
        "-O2",
        "-Wall",
        # FastRPC loads the skeleton via dlopen and looks up
        # <interface>_skel_handle_invoke, so symbols have to be exported.
        "-fvisibility=default",
        "-I$${interface_dir}",
        "-I" + sdk_dir + "/incs",
        "-I" + sdk_dir + "/incs/stddef",
        "-I" + sdk_dir + "/rtos/qurt/compute" + arch + "/include/qurt",
        "-I" + sdk_dir + "/rtos/qurt/compute" + arch + "/include/posix",
    ]
    base_linkopts = [
        "-m" + arch,
        "-G0",
        "-fpic",
        "-shared",
        "-Wl,-Bsymbolic",
    ]

    object_dir = "$(@D)/{}.objects".format(name)
    native.genrule(
        name = name,
        srcs = srcs + [skel_srcs, "@hexagon_sdk//:dsp_headers"],
        outs = [out],
        cmd = " && ".join([
            "set -e",
            "object_dir=\"{}\"".format(object_dir),
            "rm -rf \"$${object_dir}\"",
            "mkdir -p \"$${object_dir}\"",
            "interface_dir=\"\"",
            "for src in $(locations {}); do case \"$${{src}}\" in *.h) interface_dir=$$(dirname \"$${{src}}\");; esac; done".format(skel_srcs),
            "object_index=0",
            "for src in $(locations {}) {}; do case \"$${{src}}\" in *.c) {} -c \"$${{src}}\" -o \"$${{object_dir}}/$${{object_index}}.o\"; object_index=$$((object_index + 1));; esac; done".format(
                skel_srcs,
                " ".join(["$(locations {})".format(src) for src in srcs]),
                " ".join(["$(location {})".format(clang)] + base_copts + copts),
            ),
            " ".join(["$(location {})".format(clang)] + base_linkopts + linkopts + [
                "\"$${object_dir}\"/*.o",
                "-o $(location {})".format(out),
            ]),
        ]),
        tools = [
            clang,
            "@hexagon_sdk//:hexagon_toolchain_files",
        ],
        message = "Building Hexagon DSP library {}...".format(out),
        **kwargs
    )
