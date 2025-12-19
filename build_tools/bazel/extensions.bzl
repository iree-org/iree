# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bzlmod extension for IREE repository rules."""

load("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository", "new_local_repository")
load("//build_tools/bazel:workspace.bzl", "cuda_auto_configure")

def _iree_extension_impl(module_ctx):
    """Implementation of the IREE module extension."""

    # Create llvm-raw only when IREE is the root module.
    # This allows downstream consumers to provide their own LLVM.
    if any([m.is_root and m.name == "iree_core" for m in module_ctx.modules]):
        new_local_repository(
            name = "llvm-raw",
            build_file_content = "# empty",
            path = "third_party/llvm-project",
        )

    # Googletest
    local_repository(
        name = "com_google_googletest",
        path = "third_party/googletest",
    )

    # Flatcc
    new_local_repository(
        name = "com_github_dvidelabs_flatcc",
        build_file = "@iree_core//:build_tools/third_party/flatcc/BUILD.overlay",
        path = "third_party/flatcc",
    )

    # Vulkan headers
    new_local_repository(
        name = "vulkan_headers",
        build_file = "@iree_core//:build_tools/third_party/vulkan_headers/BUILD.overlay",
        path = "third_party/vulkan_headers",
    )

    # StableHLO
    local_repository(
        name = "stablehlo",
        path = "third_party/stablehlo",
    )

    # Benchmark
    local_repository(
        name = "com_google_benchmark",
        path = "third_party/benchmark",
    )

    # SPIRV-Cross
    new_local_repository(
        name = "spirv_cross",
        build_file = "@iree_core//:build_tools/third_party/spirv_cross/BUILD.overlay",
        path = "third_party/spirv_cross",
    )

    # Tracy
    new_local_repository(
        name = "tracy_client",
        build_file = "@iree_core//:build_tools/third_party/tracy_client/BUILD.overlay",
        path = "third_party/tracy",
    )

    # NCCL
    new_local_repository(
        name = "nccl",
        build_file = "@iree_core//:build_tools/third_party/nccl/BUILD.overlay",
        path = "third_party/nccl",
    )

    # HSA runtime headers
    new_local_repository(
        name = "hsa_runtime_headers",
        build_file = "@iree_core//:build_tools/third_party/hsa-runtime-headers/BUILD.overlay",
        path = "third_party/hsa-runtime-headers",
    )

    # WebGPU headers
    new_local_repository(
        name = "webgpu_headers",
        build_file = "@iree_core//:build_tools/third_party/webgpu-headers/BUILD.overlay",
        path = "third_party/webgpu-headers",
    )

    # CUDA auto-configuration
    cuda_auto_configure(
        name = "iree_cuda",
        iree_repo_alias = "@iree_core",
    )

iree_extension = module_extension(
    implementation = _iree_extension_impl,
)
