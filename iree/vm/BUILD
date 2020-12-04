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

load("//iree/tools:compilation.bzl", "iree_bytecode_module")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

#===------------------------------------------------------------------------===#
# Public API
#===------------------------------------------------------------------------===#

cc_library(
    name = "vm",
    hdrs = [
        "api.h",
    ],
    deps = [
        ":impl",
        "//iree/base:api",
    ],
)

# TODO(benvanik): make these srcs and only expose an api_cc.h.
cc_library(
    name = "cc",
    hdrs = [
        "module_abi_packing.h",
        "native_module_cc.h",
        "ref_cc.h",
    ],
    deps = [
        ":vm",
        "//iree/base:api",
        "//iree/base:ref_ptr",
        "//iree/base:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

#===------------------------------------------------------------------------===#
# Implementation
#===------------------------------------------------------------------------===#

cc_library(
    name = "impl",
    srcs = [
        "builtin_types.c",
        "context.c",
        "instance.c",
        "invocation.c",
        "list.c",
        "module.c",
        "native_module.c",
        "ref.c",
        "stack.c",
    ],
    hdrs = [
        "builtin_types.h",
        "context.h",
        "instance.h",
        "invocation.h",
        "list.h",
        "module.h",
        "native_module.h",
        "ref.h",
        "stack.h",
        "type_def.h",
        "value.h",
    ],
    deps = [
        "//iree/base:api",
        "//iree/base:core_headers",
        "//iree/base:tracing",
    ],
)

cc_test(
    name = "list_test",
    srcs = ["list_test.cc"],
    deps = [
        ":impl",
        "//iree/base:api",
        "//iree/base:ref_ptr",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "native_module_test",
    srcs = ["native_module_test.cc"],
    deps = [
        ":cc",
        ":impl",
        ":native_module_test_hdrs",
        "//iree/base:api",
        "//iree/base:status",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "native_module_test_hdrs",
    hdrs = [
        "native_module_test.h",
    ],
    deps = [
        ":impl",
        "//iree/base:api",
    ],
)

cc_test(
    name = "native_module_benchmark",
    srcs = ["native_module_benchmark.cc"],
    deps = [
        ":impl",
        ":native_module_test_hdrs",
        "//iree/base:api",
        "//iree/base:logging",
        "//iree/testing:benchmark_main",
        "@com_google_benchmark//:benchmark",
    ],
)

cc_test(
    name = "ref_test",
    srcs = ["ref_test.cc"],
    deps = [
        ":impl",
        "//iree/base:api",
        "//iree/base:ref_ptr",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "stack_test",
    srcs = ["stack_test.cc"],
    deps = [
        ":impl",
        "//iree/base:api",
        "//iree/base:ref_ptr",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

#===------------------------------------------------------------------------===#
# Bytecode interpreter module
#===------------------------------------------------------------------------===#

cc_library(
    name = "bytecode_module",
    srcs = [
        "bytecode_dispatch.c",
        "bytecode_dispatch_util.h",
        "bytecode_module.c",
        "bytecode_module_impl.h",
        "bytecode_op_table.h",
    ],
    hdrs = [
        "bytecode_module.h",
    ],
    deps = [
        ":vm",
        "//iree/base:api",
        "//iree/base:core_headers",
        "//iree/base:flatcc",
        "//iree/base:tracing",
        "//iree/schemas:bytecode_module_def_c_fbs",
    ],
)

# TODO(#357): Commented out for now, but a script should be added to update
# https://github.com/google/iree/blob/main/bindings/java/com/google/iree/native/bytecode_op_table.h
# gentbl(
#     name = "bytecode_op_table_gen",
#     tbl_outs = [
#         ("-gen-iree-vm-op-table-defs", "bytecode_op_table.h"),
#     ],
#     tblgen = "//third_party/iree/iree/tools:iree-tblgen",
#     td_file = "//third_party/iree/iree/compiler/Dialect/VM/IR:VMOps.td",
#     td_srcs = [
#         "//third_party/iree/iree/compiler/Dialect/IREE/IR:td_files",
#         "//third_party/iree/iree/compiler/Dialect/VM/IR:td_files",
#         "//third_party/llvm/llvm-project/mlir:OpBaseTdFiles",
#         "//third_party/llvm/llvm-project/mlir:include/mlir/IR/SymbolInterfaces.td",
#         "//third_party/llvm/llvm-project/mlir:include/mlir/Interfaces/CallInterfaces.td",
#         "//third_party/llvm/llvm-project/mlir:include/mlir/Interfaces/ControlFlowInterfaces.td",
#         "//third_party/llvm/llvm-project/mlir:SideEffectTdFiles",
#     ],
# )

cc_test(
    name = "bytecode_module_test",
    srcs = [
        "bytecode_dispatch_test.cc",
        "bytecode_module_test.cc",
    ],
    deps = [
        ":bytecode_module",
        ":vm",
        "//iree/base:logging",
        "//iree/base:status",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
        "//iree/vm/test:all_bytecode_modules_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "bytecode_module_benchmark",
    srcs = ["bytecode_module_benchmark.cc"],
    deps = [
        ":bytecode_module",
        ":bytecode_module_benchmark_module_cc",
        ":vm",
        "//iree/base:api",
        "//iree/base:logging",
        "//iree/testing:benchmark_main",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_benchmark//:benchmark",
    ],
)

iree_bytecode_module(
    name = "bytecode_module_benchmark_module",
    src = "bytecode_module_benchmark.mlir",
    cc_namespace = "iree::vm",
    flags = ["-iree-vm-ir-to-bytecode-module"],
)

cc_test(
    name = "bytecode_module_size_benchmark",
    srcs = ["bytecode_module_size_benchmark.cc"],
    deps = [
        ":bytecode_module",
        ":bytecode_module_size_benchmark_module_cc",
        ":vm",
        "//iree/base:api",
    ],
)

iree_bytecode_module(
    name = "bytecode_module_size_benchmark_module",
    src = "bytecode_module_size_benchmark.mlir",
    cc_namespace = "iree::vm",
    flags = ["-iree-vm-ir-to-bytecode-module"],
)

#===------------------------------------------------------------------------===#
# Emit-C modules
#===------------------------------------------------------------------------===#

cc_library(
    name = "c_funcs",
    hdrs = [
        "c_funcs.h",
    ],
)
