# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bzlmod extension for IREE repository rules."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository", "new_local_repository")
load("//build_tools/bazel:workspace.bzl", "cuda_auto_configure")
load("//build_tools/wasm:wasi_sdk_repo.bzl", "wasi_sdk_repo")

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

    # HIP API headers
    new_local_repository(
        name = "hip_api_headers",
        build_file = "@iree_core//:build_tools/third_party/hip-api-headers/BUILD.overlay",
        path = "third_party/hip-build-deps",
    )

    # HSA runtime headers
    new_local_repository(
        name = "hsa_runtime_headers",
        build_file = "@iree_core//:build_tools/third_party/hsa-runtime-headers/BUILD.overlay",
        path = "third_party/hsa-runtime-headers",
    )

    # RCCL
    new_local_repository(
        name = "rccl",
        build_file = "@iree_core//:build_tools/third_party/rccl/BUILD.overlay",
        path = "third_party/rccl",
    )

    # Doug Lea's malloc (dlmalloc v2.8.6, MIT-0 license)
    new_local_repository(
        name = "dlmalloc",
        build_file = "@iree_core//:build_tools/third_party/dlmalloc/BUILD.overlay",
        path = "third_party/dlmalloc",
    )

    # WebGPU headers
    new_local_repository(
        name = "webgpu_headers",
        build_file = "@iree_core//:build_tools/third_party/webgpu-headers/BUILD.overlay",
        path = "third_party/webgpu-headers",
    )

    # Dawn (Tint SPIR-V → WGSL translation for the WebGPU compiler target).
    # Only fetched when //compiler/plugins/target/WebGPUSPIRV is built.
    # Dawn's Tint targets have native Bazel support (auto-generated BUILD.bazel
    # files under src/tint/). Dawn's submodule deps (abseil, spirv-tools,
    # spirv-headers) are fetched separately since GitHub tarballs exclude them.
    _DAWN_COMMIT = "851ba3e50c354ef66d16c518d4341c01ed6828cc"
    http_archive(
        name = "dawn",
        urls = ["https://github.com/ArthurSonzogni/dawn/archive/%s.tar.gz" % _DAWN_COMMIT],
        strip_prefix = "dawn-%s" % _DAWN_COMMIT,
    )
    http_archive(
        name = "abseil_cpp",
        urls = ["https://github.com/abseil/abseil-cpp/archive/04f3bc01d12cf58c90a1bb68990f087fa3c3ed19.tar.gz"],
        strip_prefix = "abseil-cpp-04f3bc01d12cf58c90a1bb68990f087fa3c3ed19",
    )
    http_archive(
        name = "spirv_headers",
        urls = ["https://github.com/KhronosGroup/SPIRV-Headers/archive/465055f6c9128772e20082e893d974146acf7a02.tar.gz"],
        strip_prefix = "SPIRV-Headers-465055f6c9128772e20082e893d974146acf7a02",
    )
    http_archive(
        name = "spirv_tools",
        urls = ["https://github.com/KhronosGroup/SPIRV-Tools/archive/5a1eea1546c372a945a27d9b10e0a059db6cc651.tar.gz"],
        strip_prefix = "SPIRV-Tools-5a1eea1546c372a945a27d9b10e0a059db6cc651",
    )

    # AMDGPU device library bitcode (ocml, ockl) for ROCM compilation.
    # Matches the CMake fetch in compiler/plugins/target/ROCM/CMakeLists.txt.
    http_archive(
        name = "amdgpu_device_libs",
        urls = ["https://github.com/shark-infra/amdgpu-device-libs/releases/download/v20231101/amdgpu-device-libs-llvm-6086c272a3a59eb0b6b79dcbe00486bf4461856a.tgz"],
        sha256 = "336362416c68fdd8bb80328f65ca7ebaa0c119ea19c95df6df30c832a4df39b9",
        build_file = "@iree_core//:build_tools/third_party/amdgpu_device_libs/BUILD.overlay",
    )

    # CUDA auto-configuration
    cuda_auto_configure(
        name = "iree_cuda",
        iree_repo_alias = "@iree_core",
    )

    # wasi-sdk: clang + lld + wasi-libc + libc++ + compiler-rt for wasm targets.
    # Downloads the host-appropriate release from GitHub on first build.
    wasi_sdk_repo(
        name = "wasi_sdk",
    )

iree_extension = module_extension(
    implementation = _iree_extension_impl,
)
