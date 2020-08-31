# Copyright 2020 Google LLC
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

# Tests for end-to-end IREE support starting from the XLA HLO dialect.

load("//iree:lit_test.bzl", "iree_lit_test_suite")
load("//iree:build_defs.oss.bzl", "IREE_DRIVER_MODULES", "PLATFORM_VULKAN_DEPS")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

iree_lit_test_suite(
    name = "lit",
    srcs = glob(
        ["*.cpp"],
        exclude = ["Bench*"],
    ),
    data = [
        # runtime libraries
        "@llvm-project//mlir:tools/libvulkan-runtime-wrappers.so",
        "runtime-support.so",
        # Tests.
        ":test-dot-prod",
        "test-mnist-jit",
        "test-simple-jit",
        "test-simple-jit-vulkan",
        "test-simple-mlir",
        "test-vector-transfers-jit",
        "test-matmul-vulkan",
        "test-vec-to-gpu",
        # FileChecker.
        "//iree/tools:IreeFileCheck",
    ],
)

cc_binary(
    name = "test-dot-prod",
    srcs = ["TestDotProdJIT.cpp"],
    deps = [
        "//experimental/ModelBuilder",
        "//experimental/ModelBuilder:ModelRunner",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFTransforms",
    ],
)

cc_binary(
    name = "test-vector-transfers-jit",
    srcs = ["TestVectorTransfersJIT.cpp"],
    deps = [
        ":runtime-support.so",
        "//experimental/ModelBuilder",
        "//experimental/ModelBuilder:ModelRunner",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFTransforms",
    ],
)

cc_binary(
    name = "test-mnist-jit",
    srcs = ["TestMNISTJIT.cpp"],
    deps = [
        "//experimental/ModelBuilder",
        "//experimental/ModelBuilder:ModelRunner",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFTransforms",
        # mlir_runner_utils with iostream needed for printMemRef atm
        "@llvm-project//mlir:mlir_runner_utils",
    ],
)

cc_binary(
    name = "test-simple-jit",
    srcs = ["TestSimpleJIT.cpp"],
    deps = [
        "//experimental/ModelBuilder",
        "//experimental/ModelBuilder:ModelRunner",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SCFTransforms",
    ],
)

cc_binary(
    name = "test-simple-jit-vulkan",
    srcs = ["TestSimpleJITVulkan.cpp"],
    deps = [
        "//experimental/ModelBuilder",
        "//experimental/ModelBuilder:ModelRunner",
        "//iree/base:initializer",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:mlir_runner_utils",
    ] + PLATFORM_VULKAN_DEPS + IREE_DRIVER_MODULES,
)

cc_binary(
    name = "test-matmul-vulkan",
    srcs = ["TestMatMulVulkan.cpp"],
    deps = [
        "//experimental/ModelBuilder",
        "//experimental/ModelBuilder:ModelRunner",
        "//experimental/ModelBuilder:VulkanLaunchWrapper",
        "//iree/base:initializer",
        "//iree/compiler/Conversion/CodegenUtils",
        "//iree/compiler/Conversion/LinalgToSPIRV",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:GPUToSPIRVTransforms",
        "@llvm-project//mlir:GPUToVulkanTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgToLLVM",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:StandardToSPIRVConversions",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TargetLLVMIR",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorToLLVM",
        # mlir_runner_utils with iostream needed for printMemRef atm
        "@llvm-project//mlir:mlir_runner_utils",
    ] + PLATFORM_VULKAN_DEPS + IREE_DRIVER_MODULES,
)

cc_binary(
    name = "test-vec-to-gpu",
    srcs = ["TestVectorToGPU.cpp"],
    tags = [
        "noga",
    ],
    deps = [
        "//experimental/ModelBuilder",
        "//experimental/ModelBuilder:ModelRunner",
        "//experimental/ModelBuilder:VulkanLaunchWrapper",
        "//iree/base:initializer",
        "//iree/compiler/Conversion/CodegenUtils",
        "//iree/compiler/Conversion/LinalgToSPIRV",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:GPUToVulkanTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgToLLVM",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:StandardToSPIRVConversions",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:mlir_c_runner_utils",
        # mlir_runner_utils with iostream needed for printMemRef atm
        "@llvm-project//mlir:mlir_runner_utils",
    ] + PLATFORM_VULKAN_DEPS + IREE_DRIVER_MODULES,
)

cc_binary(
    name = "bench-matmul-gpu",
    srcs = ["BenchMatMulVectorGPU.cpp"],
    tags = [
        "noga",
    ],
    deps = [
        "//experimental/ModelBuilder",
        "//experimental/ModelBuilder:ModelRunner",
        "//experimental/ModelBuilder:VulkanLaunchWrapper",
        "//iree/base:initializer",
        "//iree/compiler/Conversion/CodegenUtils",
        "//iree/compiler/Conversion/LinalgToSPIRV",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:GPUToVulkanTransforms",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMTransforms",
        "@llvm-project//mlir:LinalgOps",
        "@llvm-project//mlir:LinalgToLLVM",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:StandardToSPIRVConversions",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:VectorOps",
        "@llvm-project//mlir:mlir_c_runner_utils",
        # mlir_runner_utils with iostream needed for printMemRef atm
        "@llvm-project//mlir:mlir_runner_utils",
    ] + PLATFORM_VULKAN_DEPS + IREE_DRIVER_MODULES,
)

cc_binary(
    name = "test-simple-mlir",
    srcs = ["TestSimpleMLIR.cpp"],
    deps = [
        "//experimental/ModelBuilder",
        "//experimental/ModelBuilder:ModelRunner",
    ],
)

# Benchmarks.

# Matrix x Vector expressed in Vector dialect.
cc_binary(
    name = "bench-matvec-vector-jit",
    srcs = ["BenchMatVecVectorJIT.cpp"],
    deps = [
        "//experimental/ModelBuilder",
        "//experimental/ModelBuilder:ModelRunner",
        "@com_google_benchmark//:benchmark",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:IR",
    ],
)

# Matrix x Matrix expressed in Vector dialect.
cc_binary(
    name = "bench-matmul-vector-jit",
    srcs = ["BenchMatMulVectorJIT.cpp"],
    deps = [
        "//experimental/ModelBuilder",
        "//experimental/ModelBuilder:ModelRunner",
        "@com_google_benchmark//:benchmark",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:IR",
    ],
)

# Matrix x Matrix expressed in Vector dialect (using intrinsics).
cc_binary(
    name = "bench-matmul-vector-column-major-llvm-intrinsics-jit",
    srcs = ["BenchMatMulVectorColumnMajorLLVMIntrinsicsJIT.cpp"],
    copts = ["-O3"],
    deps = [
        "//experimental/ModelBuilder",
        "//experimental/ModelBuilder:ModelRunner",
        "@com_google_benchmark//:benchmark",
        "@llvm-project//mlir:EDSC",
        "@llvm-project//mlir:IR",
    ],
)

# Runtime support library (for printing vectors).

cc_binary(
    name = "runtime-support.so",
    linkshared = True,
    linkstatic = False,
    deps = [
        "@llvm-project//mlir:mlir_c_runner_utils",
    ],
)
