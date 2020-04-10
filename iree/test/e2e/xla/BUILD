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

# Tests for end-to-end IREE support starting from the XLA HLO dialect.

load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")
load("//iree:lit_test.bzl", "iree_lit_test_suite")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

VMLA_CHECK_TESTS = [
    "abs.mlir",
    "add.mlir",
    "batch_norm_inference.mlir",
    "broadcast.mlir",
    "broadcast_in_dim.mlir",
    "concatenate.mlir",
    "convert_int.mlir",
    "cos.mlir",
    "dot.mlir",
    "dot_general.mlir",
    "exp.mlir",
    "floor.mlir",
    "gather.mlir",
    "gather_concat.mlir",
    "gemm.mlir",
    "gemm_large.mlir",
    "log.mlir",
    "max_float.mlir",
    "max_int.mlir",
    "min_float.mlir",
    "min_int.mlir",
    "multiply.mlir",
    "negate.mlir",
    "pad.mlir",
    "reduce_float.mlir",
    "reduce_int.mlir",
    "reduce_window.mlir",
    "rem.mlir",
    "reshape.mlir",
    "reshape_adddims.mlir",
    "reshape_dropdims.mlir",
    "reverse.mlir",
    "rsqrt.mlir",
    "select.mlir",
    "sin.mlir",
    "slice.mlir",
    "sqrt.mlir",
    "while.mlir",
]

VULKAN_CHECK_TESTS = [
    "abs.mlir",
    "add.mlir",
    "broadcast.mlir",
    "broadcast_in_dim.mlir",
    "concatenate.mlir",
    "convert_int.mlir",
    "cos.mlir",
    "dot.mlir",
    "exp.mlir",
    "floor.mlir",
    "gather.mlir",
    "gather_concat.mlir",
    "gemm.mlir",
    "gemm_large.mlir",
    "log.mlir",
    "max_float.mlir",
    "max_int.mlir",
    "min_float.mlir",
    "min_int.mlir",
    "multiply.mlir",
    "pad.mlir",
    "reduce_float.mlir",
    "reduce_int.mlir",
    "rem.mlir",
    "reshape.mlir",
    "reshape_adddims.mlir",
    "reshape_dropdims.mlir",
    "reverse.mlir",
    "rsqrt.mlir",
    "select.mlir",
    "sin.mlir",
    "slice.mlir",
    "sqrt.mlir",
    "while.mlir",
]

LINALG_TO_SPIRV_VULKAN_CHECK_TESTS = [
    "abs.mlir",
    "add.mlir",
    "batch_norm_inference.mlir",
    "broadcast_in_dim.mlir",
    "cos.mlir",
    "exp.mlir",
    "gemm.mlir",
    "gemm_large.mlir",
    "log.mlir",
    "max_float.mlir",
    "max_int.mlir",
    "min_float.mlir",
    "min_int.mlir",
    "rem.mlir",
    "reshape_adddims.mlir",
    "reshape_dropdims.mlir",
    "rsqrt.mlir",
    "select.mlir",
    "sqrt.mlir",
]

LLVMJIT_CHECK_TESTS = [
    "abs.mlir",
    "add.mlir",
    "batch_norm_inference.mlir",
    "broadcast_in_dim.mlir",
    "cos.mlir",
    "dot.mlir",
    "exp.mlir",
    "gemm.mlir",
    "gemm_large.mlir",
    "log.mlir",
    "max_float.mlir",
    "max_int.mlir",
    "min_float.mlir",
    "min_int.mlir",
    "multiply.mlir",
    "negate.mlir",
    "reduce_float.mlir",
    "reduce_int.mlir",
    "reduce_window.mlir",
    "rem.mlir",
    "reshape.mlir",
    "reshape_adddims.mlir",
    "reshape_dropdims.mlir",
    "rsqrt.mlir",
    "select.mlir",
    "sqrt.mlir",
    "while.mlir",
]

LEGACY_LIT_TESTS = [
    "collatz.mlir",
    "compare.mlir",
    "constants.mlir",
    "conv.mlir",
    "dynamic_abs.mlir",
    "fragment_000.mlir",
    "fullyconnected.mlir",
    "globals.mlir",
    "load_i16.mlir",
    "load_i8.mlir",
    "mnist.mlir",
    "multiple_return.mlir",
    "scalar.mlir",
    "store.mlir",
    "unidirectional_lstm.mlir",
]

iree_lit_test_suite(
    name = "lit",
    srcs = LEGACY_LIT_TESTS,
    data = [
        "//iree/tools:IreeFileCheck",
        "//iree/tools:iree-run-mlir",
    ],
    tags = ["hostonly"],
)

iree_check_single_backend_test_suite(
    name = "check_vmla_vmla",
    srcs = VMLA_CHECK_TESTS,
    driver = "vmla",
    target_backend = "vmla",
)

iree_check_single_backend_test_suite(
    name = "check_vulkan-spirv_vulkan",
    srcs = VULKAN_CHECK_TESTS,
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

iree_check_single_backend_test_suite(
    name = "check_linalg-to-spirv_vulkan",
    srcs = LINALG_TO_SPIRV_VULKAN_CHECK_TESTS,
    compiler_flags = ["-iree-use-linalg-to-spirv-path"],
    driver = "vulkan",
    target_backend = "vulkan-spirv",
)

iree_check_single_backend_test_suite(
    name = "check_llvm-ir_llvm",
    srcs = LLVMJIT_CHECK_TESTS,
    driver = "llvm",
    target_backend = "llvm-ir",
)

test_suite(
    name = "check",
    tests = [
        ":check_linalg-to-spirv_vulkan",
        ":check_llvm-ir_llvm",
        ":check_vmla_vmla",
        ":check_vulkan-spirv_vulkan",
    ],
)
