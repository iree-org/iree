package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

load("//third_party/llvm/llvm/projects/google_mlir:tblgen.bzl", "gentbl")

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
        "//iree/compiler/IR",
        "//iree/compiler/Serialization",
        "//iree/compiler/Utils",
        "//iree/schemas/bytecode:interpreter_bytecode_v0",
        "//third_party/llvm/llvm:support",
        "//third_party/llvm/llvm/projects/google_mlir:IR",
        "//third_party/llvm/llvm/projects/google_mlir:StandardOps",
    ],
    alwayslink = 1,
)

gentbl(
    name = "HLOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "HLOps.h.inc"),
        ("-gen-op-defs", "HLOps.cpp.inc"),
    ],
    tblgen = "//third_party/llvm/llvm/projects/google_mlir:mlir-tblgen",
    td_file = "HLOps.td",
    td_srcs = [
        ":td_files",
        "//third_party/llvm/llvm/projects/google_mlir:include/mlir/IR/OpBase.td",
        "//iree/compiler/IR:OpBase.td",
    ],
)

gentbl(
    name = "LLOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "LLOps.h.inc"),
        ("-gen-op-defs", "LLOps.cpp.inc"),
    ],
    tblgen = "//third_party/llvm/llvm/projects/google_mlir:mlir-tblgen",
    td_file = "LLOps.td",
    td_srcs = [
        ":td_files",
        "//third_party/llvm/llvm/projects/google_mlir:include/mlir/IR/OpBase.td",
        "//iree/compiler/IR:OpBase.td",
    ],
)
