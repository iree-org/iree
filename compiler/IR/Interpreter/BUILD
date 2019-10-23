package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

load("@local_config_mlir//:tblgen.bzl", "gentbl")

filegroup(
    name = "td_files",
    srcs = glob(["*.td"]),
)

cc_library(
    name = "Interpreter",
    srcs = [
        "HLDialect.cpp",
        "HLOps.cpp",
        "HLOps.cpp.inc",
        "LLDialect.cpp",
        "LLOps.cpp",
        "LLOps.cpp.inc",
        "OpWriters.cpp",
    ],
    hdrs = [
        "HLDialect.h",
        "HLOps.h",
        "HLOps.h.inc",
        "LLDialect.h",
        "LLOps.h",
        "LLOps.h.inc",
        "OpWriters.h",
    ],
    deps = [
        ":HLOpsGen",
        ":LLOpsGen",
        "///compiler/IR",
        "///compiler/Serialization",
        "///compiler/Utils",
        "///schemas/bytecode:interpreter_bytecode_v0",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:StandardOps",
    ],
    alwayslink = 1,
)

gentbl(
    name = "HLOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "HLOps.h.inc"),
        ("-gen-op-defs", "HLOps.cpp.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "HLOps.td",
    td_srcs = [
        ":td_files",
        "@local_config_mlir//:include/mlir/IR/OpBase.td",
        "///compiler/IR:OpBase.td",
    ],
)

gentbl(
    name = "LLOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "LLOps.h.inc"),
        ("-gen-op-defs", "LLOps.cpp.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "LLOps.td",
    td_srcs = [
        ":td_files",
        "@local_config_mlir//:include/mlir/IR/OpBase.td",
        "///compiler/IR:OpBase.td",
    ],
)
