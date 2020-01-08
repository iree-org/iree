# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for handling hand-written SPIR-V files."""

load("//iree:build_defs.bzl", "iree_glsl_vulkan")
load("//build_tools/embed_data:build_defs.bzl", "cc_embed_data")

def spirv_kernel_cc_library(name, srcs):
    """Compiles GLSL files into SPIR-V binaries and embeds them in a cc_library.

    Args:
        name: cc_library name to depend on.
        srcs: a list of GLSL source files.
    """
    spv_files = []
    for src in srcs:
        spv_name = src.split(".")[-2]
        iree_glsl_vulkan(
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
        cc_file_output = name + ".cc",
        h_file_output = name + ".h",
        cpp_namespace = "mlir::iree_compiler::spirv_kernels",
        flatten = True,
    )
