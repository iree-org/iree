# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for compiling with clang to produce AMDGPU libraries."""

def iree_amdgpu_binary(
        name,
        target,
        arch,
        srcs,
        internal_hdrs = [],
        copts = [],
        linkopts = [],
        **kwargs):
    """Builds an LLVM shared library for AMDGPU from input files via clang.

    Args:
        name: Name of the target.
        target: LLVM `-target` flag.
        arch: LLVM `-march` flag.
        srcs: source files to pass to clang.
        internal_hdrs: all headers transitively included by the source files.
                       Unlike typical Bazel `hdrs`, these are not exposed as
                       interface headers. This would normally be part of `srcs`,
                       but separating it was easier for `bazel_to_cmake`, as
                       CMake does not need this, and making this explicitly
                       Bazel-only allows using `filegroup` on the Bazel side.
        copts: additional flags to pass to clang.
        linkopts: additional flags to pass to lld.
        **kwargs: any additional attributes to pass to the underlying rules.
    """

    clang_tool = "@llvm-project//clang:clang"
    link_tool = "@llvm-project//llvm:llvm-link"
    lld_tool = "@llvm-project//lld:lld"
    builtin_headers_dep = "@llvm-project//clang:builtin_headers_gen"
    builtin_headers_path = "external/llvm-project/clang/staging/include/"

    base_copts = [
        # C configuration.
        "-x c",
        "-std=c23",
        "-Xclang -finclude-default-header",
        "-nogpulib",
        "-fno-short-wchar",

        # Target architecture/machine.
        "-target %s" % (target),
        "-march=%s" % (arch),
        "-fgpu-rdc",  # NOTE: may not be required for all targets

        # Header paths for builtins and our own includes.
        "-isystem $(BINDIR)/%s" % builtin_headers_path,
        "-I$(BINDIR)/runtime/src",
        "-Iruntime/src",

        # Avoid warnings about things we do that are not compatible across
        # compilers but are fine because we're only ever compiling with clang.
        "-Wno-gnu-pointer-arith",

        # Optimized.
        "-fno-ident",
        "-fvisibility=hidden",
        "-O3",

        # Object file only in bitcode format.
        "-c",
        "-emit-llvm",
    ]

    bitcode_files = []

    for src in srcs:
        bitcode_out = "%s_%s.bc" % (name, src)
        bitcode_files.append(bitcode_out)
        native.genrule(
            name = "gen_%s" % (bitcode_out),
            srcs = [src, builtin_headers_dep] + internal_hdrs,
            outs = [bitcode_out],
            cmd = " && ".join([
                " ".join([
                    "$(location %s)" % (clang_tool),
                    " ".join(base_copts + copts),
                    "-o $(location %s)" % (bitcode_out),
                    "$(location %s)" % (src),
                ]),
            ]),
            tools = [clang_tool],
            message = "Compiling %s to %s..." % (src, bitcode_out),
            output_to_bindir = 1,
            **kwargs
        )

    archive_out = "%s.a" % (name)
    native.genrule(
        name = "archive_%s" % (name),
        srcs = bitcode_files,
        outs = [archive_out],
        cmd = " && ".join([
            " ".join([
                "$(location %s)" % (link_tool),
                " ".join(["$(locations %s)" % (src) for src in bitcode_files]),
                "-o $(location %s)" % (archive_out),
            ]),
        ]),
        tools = [link_tool],
        message = "Archiving bitcode libraries %s to %s..." % (bitcode_files, archive_out),
        output_to_bindir = 1,
        **kwargs
    )

    link_out = "%s.bc" % (name)
    native.genrule(
        name = "link_%s" % (name),
        srcs = [archive_out],
        outs = [link_out],
        cmd = " && ".join([
            " ".join([
                "$(location %s)" % (link_tool),
                "-internalize",
                "-only-needed",
                "$(locations %s)" % (archive_out),
                "-o $(location %s)" % (link_out),
            ]),
        ]),
        tools = [link_tool],
        message = "Linking bitcode library %s to %s..." % (name, link_out),
        output_to_bindir = 1,
        **kwargs
    )

    base_linkopts = [
        "-m elf64_amdgpu",
        "--build-id=none",
        "--no-undefined",
        "-shared",
        "-plugin-opt=mcpu=%s" % (arch),
        "-plugin-opt=O3",
        "--lto-CGO3",
        "--no-whole-archive",
        "--gc-sections",
        "--strip-debug",
        "--discard-all",
        "--discard-locals",
    ]

    out = "%s.so" % (name)
    native.genrule(
        name = name,
        srcs = [link_out],
        outs = [out],
        cmd = " && ".join([
            " ".join([
                "$(location %s)" % (lld_tool),
                "-flavor gnu",
                " ".join(base_linkopts + linkopts),
                "$(location %s)" % (link_out),
                "-o $(location %s)" % (out),
            ]),
        ]),
        tools = [lld_tool],
        message = "Generating OpenCL binary %s to %s..." % (name, out),
        output_to_bindir = 1,
        **kwargs
    )
