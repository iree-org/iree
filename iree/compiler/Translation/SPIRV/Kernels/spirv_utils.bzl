"""Utilities for handling hand-written SPIR-V files."""

load("//third_party/glslang:build_defs.bzl", "glsl_vulkan")
load("//tools/build_defs/cc:cc_embed_data.bzl", "cc_embed_data")

def spirv_kernel_cc_library(name, srcs):
    """Compiles GLSL files into SPIR-V binaries and embeds them in a cc_library.

    Args:
        name: cc_library name to depend on.
        srcs: a list of GLSL source files.
    """
    spv_files = []
    for src in srcs:
        spv_name = src.split(".")[-2]
        glsl_vulkan(
            name = spv_name,
            srcs = [src],
        )
        spv_files.append(spv_name + ".spv")
    native.filegroup(
        name = name + "_files",
        srcs = spv_files,
    )
    cc_embed_data(
        name = name,
        srcs = spv_files,
        outs = [
            name + ".cc",
            name + ".h",
        ],
        embedopts = [
            "--namespace=mlir::iree_compiler::spirv_kernels",
        ],
    )
