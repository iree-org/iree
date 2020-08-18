# Bytecode VM.

load("//iree/tools:compilation.bzl", "iree_bytecode_module")
load("//build_tools/bazel:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "builtin_types",
    srcs = ["builtin_types.c"],
    hdrs = ["builtin_types.h"],
    deps = [
        ":list",
        ":ref",
        "//iree/base:api",
    ],
)

cc_test(
    name = "bytecode_dispatch_test",
    srcs = ["bytecode_dispatch_test.cc"],
    deps = [
        ":builtin_types",
        ":bytecode_module",
        ":context",
        ":instance",
        ":invocation",
        ":module",
        "//iree/base:logging",
        "//iree/base:status",
        "//iree/testing:gtest_main",
        "//iree/vm/test:all_bytecode_modules_cc",
        "@com_google_absl//absl/strings",
    ],
)

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
        ":builtin_types",
        ":bytecode_op_table_gen",
        ":list",
        ":module",
        ":ref",
        ":stack",
        ":type_def",
        ":value",
        "//iree/base:alignment",
        "//iree/base:api",
        "//iree/base:tracing",
        "//iree/schemas:bytecode_module_def_c_fbs",
        "@com_github_dvidelabs_flatcc//:runtime",
    ],
)

cc_test(
    name = "bytecode_module_benchmark",
    srcs = ["bytecode_module_benchmark.cc"],
    deps = [
        ":bytecode_module",
        ":bytecode_module_benchmark_module_cc",
        ":module",
        ":stack",
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

cc_test(
    name = "bytecode_module_test",
    srcs = ["bytecode_module_test.cc"],
    deps = [
        ":bytecode_module",
        "//iree/testing:gtest_main",
    ],
)

gentbl(
    name = "bytecode_op_table_gen",
    tbl_outs = [
        ("-gen-iree-vm-op-table-defs", "bytecode_op_table.h"),
    ],
    tblgen = "//iree/tools:iree-tblgen",
    td_file = "//iree/compiler/Dialect/VM/IR:VMOps.td",
    td_srcs = [
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "//iree/compiler/Dialect/VM/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/IR/SymbolInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/CallInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/ControlFlowInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
    ],
)

cc_library(
    name = "context",
    srcs = ["context.c"],
    hdrs = ["context.h"],
    deps = [
        ":instance",
        ":module",
        ":stack",
        "//iree/base:api",
        "//iree/base:atomics",
        "//iree/base:tracing",
    ],
)

cc_library(
    name = "instance",
    srcs = ["instance.c"],
    hdrs = ["instance.h"],
    deps = [
        ":builtin_types",
        "//iree/base:api",
        "//iree/base:atomics",
    ],
)

cc_library(
    name = "invocation",
    srcs = ["invocation.c"],
    hdrs = ["invocation.h"],
    deps = [
        ":context",
        ":list",
        ":module",
        "//iree/base:api",
        "//iree/base:tracing",
    ],
)

cc_library(
    name = "list",
    srcs = ["list.c"],
    hdrs = ["list.h"],
    deps = [
        ":ref",
        ":type_def",
        ":value",
        "//iree/base:alignment",
        "//iree/base:api",
    ],
)

cc_test(
    name = "list_test",
    srcs = ["list_test.cc"],
    deps = [
        ":builtin_types",
        ":list",
        "//iree/base:api",
        "//iree/base:ref_ptr",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "module",
    srcs = ["module.c"],
    hdrs = ["module.h"],
    deps = [
        "//iree/base:alignment",
        "//iree/base:api",
        "//iree/base:atomics",
    ],
)

cc_library(
    name = "module_abi_cc",
    hdrs = [
        "module_abi_cc.h",
        "module_abi_packing.h",
    ],
    deps = [
        ":builtin_types",
        ":module",
        ":ref",
        ":ref_cc",
        ":stack",
        "//iree/base:api",
        "//iree/base:ref_ptr",
        "//iree/base:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "ref",
    srcs = ["ref.c"],
    hdrs = ["ref.h"],
    deps = [
        "//iree/base:api",
        "//iree/base:atomics",
    ],
)

cc_test(
    name = "ref_test",
    srcs = ["ref_test.cc"],
    deps = [
        ":ref",
        "//iree/base:api",
        "//iree/base:ref_ptr",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "ref_cc",
    hdrs = ["ref_cc.h"],
    deps = [
        ":ref",
        "//iree/base:api",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "stack",
    srcs = ["stack.c"],
    hdrs = ["stack.h"],
    deps = [
        ":module",
        ":ref",
        "//iree/base:alignment",
        "//iree/base:api",
        "//iree/base:tracing",
    ],
)

cc_test(
    name = "stack_test",
    srcs = ["stack_test.cc"],
    deps = [
        ":ref",
        ":stack",
        "//iree/base:api",
        "//iree/base:ref_ptr",
        "//iree/testing:gtest",
        "//iree/testing:gtest_main",
    ],
)

cc_library(
    name = "type_def",
    hdrs = ["type_def.h"],
    deps = [
        ":ref",
        ":value",
    ],
)

cc_library(
    name = "value",
    hdrs = ["value.h"],
)

cc_library(
    name = "vm",
    hdrs = [
        "api.h",
    ],
    deps = [
        ":builtin_types",
        ":context",
        ":instance",
        ":invocation",
        ":list",
        ":module",
        ":ref",
        ":stack",
        ":type_def",
        ":value",
        "//iree/base:api",
    ],
)
