# Bytecode VM.

load("//iree/tools:compilation.bzl", "iree_bytecode_module")
load("@local_config_mlir//:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_test(
    name = "bytecode_dispatch_test",
    srcs = ["bytecode_dispatch_test.cc"],
    deps = [
        ":bytecode_dispatch_test_module_cc",
        ":bytecode_module",
        ":module",
        "//iree/base:logging",
        "//iree/testing:gtest_main",
        "@com_google_absl//absl/strings",
    ],
)

iree_bytecode_module(
    name = "bytecode_dispatch_test_module",
    src = "bytecode_dispatch_test.mlir",
    cc_namespace = "iree::vm",
    translation = "-iree-vm-ir-to-bytecode-module",
)

cc_library(
    name = "bytecode_module",
    srcs = [
        "bytecode_dispatch.c",
        "bytecode_module.cc",
        "bytecode_module_impl.h",
        "bytecode_op_table.h",
    ],
    hdrs = [
        "bytecode_module.h",
    ],
    deps = [
        ":bytecode_op_table_gen",
        ":module",
        "//iree/base:api",
        "//iree/base:flatbuffer_util",
        "//iree/base:target_platform",
        "//iree/schemas:bytecode_module_def_cc_fbs",
        "@com_github_google_flatbuffers//:flatbuffers",
    ],
)

cc_test(
    name = "bytecode_module_benchmark",
    srcs = ["bytecode_module_benchmark.cc"],
    deps = [
        ":bytecode_module",
        ":bytecode_module_benchmark_module_cc",
        ":module",
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
    translation = "-iree-vm-ir-to-bytecode-module",
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
        "//iree/compiler/Dialect:td_files",
        "//iree/compiler/Dialect/VM/IR:td_files",
        "@local_config_mlir//:OpBaseTdFiles",
        "@local_config_mlir//:include/mlir/Analysis/CallInterfaces.td",
    ],
)

cc_library(
    name = "module",
    srcs = [
        "ref.c",
        "stack.c",
    ],
    hdrs = [
        "module.h",
        "ref.h",
        "stack.h",
    ],
    deps = [
        "//iree/base:api",
    ],
)

cc_library(
    name = "module_base",
    srcs = ["module_base.cc"],
    hdrs = ["module_base.h"],
    deps = [
        ":module",
        "//iree/base:api",
        "//iree/base:api_util",
        "//iree/base:ref_ptr",
        "//iree/base:status",
        "//iree/base:tracing",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "ref_test",
    srcs = ["ref_test.cc"],
    deps = [
        ":module",
        "//iree/base:api",
        "//iree/base:ref_ptr",
        "//iree/testing:gtest_main",
    ],
)

cc_test(
    name = "stack_test",
    srcs = ["stack_test.cc"],
    deps = [
        ":module",
        "//iree/base:api",
        "//iree/base:ref_ptr",
        "//iree/testing:gtest_main",
    ],
)
