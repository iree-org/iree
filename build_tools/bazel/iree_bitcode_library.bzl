# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for compiling with clang to produce bitcode libraries."""

def iree_arch_to_llvm_arch(
        iree_arch = None):
    """Converts an IREE_ARCH value to the corresponding LLVM arch name.

    Similar to the CMake function with the same name.

    Args:
        iree_arch: IREE_ARCH string value.

    Returns:
        The LLVM name for that architecture (first component of target triple).
    """

    if not iree_arch:
        return None
    if iree_arch == "arm_64":
        return "aarch64"
    if iree_arch == "arm_32":
        return "arm"
    if iree_arch == "x86_64":
        return "x86_64"
    if iree_arch == "x86_32":
        return "i386"
    if iree_arch == "riscv_64":
        return "riscv64"
    if iree_arch == "riscv_32":
        return "riscv32"
    if iree_arch == "wasm_64":
        return "wasm64"
    if iree_arch == "wasm_32":
        return "wasm32"
    fail("Unhandled IREE_ARCH value %s" % iree_arch)

def iree_bitcode_library(
        name,
        srcs,
        internal_hdrs = [],
        copts = [],
        out = None,
        arch = None,
        **kwargs):
    """Builds an LLVM bitcode library from an input file via clang.

    Args:
        name: Name of the target.
        arch: Target architecture to compile for, in IREE_ARCH format. If left
              empty, will produce architecture-independent bitcode by stripping
              target triple and target attributes; that only makes sense if the
              sources being compiled are truly architecture-independent.
        srcs: source files to pass to clang.
        internal_hdrs: all headers transitively included by the source files.
                       Unlike typical Bazel `hdrs`, these are not exposed as
                       interface headers. This would normally be part of `srcs`,
                       but separating it was easier for `bazel_to_cmake`, as
                       CMake does not need this, and making this explicitly
                       Bazel-only allows using `filegroup` on the Bazel side.
        copts: additional flags to pass to clang.
        out: output file name (defaults to name.bc).
        **kwargs: any additional attributes to pass to the underlying rules.
    """

    clang_tool = "@llvm-project//clang:clang"
    link_tool = "@llvm-project//llvm:llvm-link"
    builtin_headers_dep = "@llvm-project//clang:builtin_headers_gen"
    builtin_headers_path = "external/llvm-project/clang/staging/include/"

    base_copts = [
        # C17 with no system deps.
        "-std=c17",
        "-nostdinc",
        "-ffreestanding",

        # Optimized and unstamped.
        "-O3",
        "-DNDEBUG",
        "-fno-ident",
        "-fdiscard-value-names",

        # Set the size of wchar_t to 4 bytes (instead of 2 bytes).
        # This must match what the runtime is built with.
        "-fno-short-wchar",

        # Object file only in bitcode format:
        "-c",
        "-emit-llvm",

        # Force the library into standalone mode (not depending on build-directory
        # configuration).
        "-DIREE_DEVICE_STANDALONE=1",
    ]

    llvmir_processing_tool = None
    if arch:
        # Compile to the specified target architecture.
        base_copts.extend(["-target", iree_arch_to_llvm_arch(arch)])
    else:
        # Output text rather than binary serialization of LLVM IR for processing
        base_copts.append("-S")

        # Strip target information from generated LLVM IR.
        llvmir_processing_tool = "//build_tools/scripts:strip_target_info"

    bitcode_files = []
    for src in srcs:
        bitcode_out = "%s_%s.bc" % (name, src)
        native.genrule(
            name = "gen_%s" % (bitcode_out),
            srcs = [src, builtin_headers_dep] + internal_hdrs,
            outs = [bitcode_out],
            cmd = " && ".join([
                " ".join([
                    "$(location %s)" % (clang_tool),
                    "-isystem $(BINDIR)/%s" % builtin_headers_path,
                    " ".join(base_copts + copts),
                    " ".join(["-I $(BINDIR)/runtime/src"]),
                    " ".join(["-I runtime/src"]),
                    "-o $(location %s)" % (bitcode_out),
                    "$(location %s)" % (src),
                ]),
            ]),
            tools = [
                clang_tool,
            ],
            message = "Compiling %s to %s..." % (src, bitcode_out),
            output_to_bindir = 1,
            **kwargs
        )

        if llvmir_processing_tool:
            processed_bitcode_out = "%s_%s.processed.bc" % (name, src)
            native.genrule(
                name = "gen_%s" % (processed_bitcode_out),
                srcs = [bitcode_out],
                outs = [processed_bitcode_out],
                cmd = " ".join([
                    "$(location %s)" % (llvmir_processing_tool),
                    "< $(location %s)" % bitcode_out,
                    "> $(location %s)" % processed_bitcode_out,
                ]),
                tools = [
                    llvmir_processing_tool,
                ],
                message = "Processing %s into %s using %s..." % (bitcode_out, processed_bitcode_out, llvmir_processing_tool),
                output_to_bindir = 1,
                **kwargs
            )
            bitcode_files.append(processed_bitcode_out)
        else:
            bitcode_files.append(bitcode_out)

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
        tools = [link_tool],
        message = "Linking bitcode library %s to %s..." % (name, out),
        output_to_bindir = 1,
        **kwargs
    )

def iree_link_bitcode(
        name,
        bitcode_files,
        out = None,
        link_tool = "@llvm-project//llvm:llvm-link",
        **kwargs):
    """Builds an LLVM bitcode library from an input file via clang.

    Args:
        name: Name of the target.
        bitcode_files: bitcode files to link together.
        out: output file name (defaults to name.bc).
        link_tool: llvm-link tool used for linking bitcode files.
        **kwargs: any additional attributes to pass to the underlying rules.
    """

    bitcode_files_qualified = [(("//" + native.package_name() + "/" + b) if b.count(":") else b) for b in bitcode_files]

    if not out:
        out = "%s.bc" % (name)
    native.genrule(
        name = name,
        srcs = bitcode_files_qualified,
        outs = [out],
        cmd = " && ".join([
            " ".join([
                "$(location %s)" % (link_tool),
                "-o $(location %s)" % (out),
                " ".join(["$(locations %s)" % (src) for src in bitcode_files_qualified]),
            ]),
        ]),
        tools = [link_tool],
        message = "Linking bitcode library %s to %s..." % (name, out),
        output_to_bindir = 1,
        **kwargs
    )
