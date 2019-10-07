package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

load("//third_party/llvm/llvm/projects/google_mlir:tblgen.bzl", "gentbl")

exports_files(["OpBase.td"])

filegroup(
    name = "td_files",
    srcs = glob(["*.td"]),
)

cc_library(
    name = "IR",
    srcs = [
        "ConfigOps.cpp",
        "ConfigOps.cpp.inc",
        "Dialect.cpp",
        "Enums.cpp.inc",
        "Ops.cpp",
        "Ops.cpp.inc",
        "StructureOps.cpp",
        "StructureOps.cpp.inc",
        "Traits.cpp",
        "Types.cpp",
    ],
    hdrs = [
        "ConfigOps.h",
        "ConfigOps.h.inc",
        "Dialect.h",
        "Enums.h.inc",
        "Ops.h",
        "Ops.h.inc",
        "StructureOps.h",
        "StructureOps.h.inc",
        "Traits.h",
        "Types.h",
    ],
    deps = [
        ":ConfigOpsGen",
        ":EnumsGen",
        ":OpsGen",
        ":StructureOpsGen",
        "//third_party/llvm/llvm:support",
        "//third_party/llvm/llvm/projects/google_mlir:IR",
        "//third_party/llvm/llvm/projects/google_mlir:StandardOps",
        "//third_party/llvm/llvm/projects/google_mlir:Support",
    ],
    alwayslink = 1,
)

gentbl(
    name = "ConfigOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "ConfigOps.h.inc"),
        ("-gen-op-defs", "ConfigOps.cpp.inc"),
    ],
    tblgen = "//third_party/llvm/llvm/projects/google_mlir:mlir-tblgen",
    td_file = "ConfigOps.td",
    td_srcs = [
        ":td_files",
        "//third_party/llvm/llvm/projects/google_mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "EnumsGen",
    tbl_outs = [
        ("-gen-enum-decls", "Enums.h.inc"),
        ("-gen-enum-defs", "Enums.cpp.inc"),
    ],
    tblgen = "//third_party/llvm/llvm/projects/google_mlir:mlir-tblgen",
    td_file = "OpBase.td",
    td_srcs = [
        ":td_files",
        "//third_party/llvm/llvm/projects/google_mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "OpsGen",
    tbl_outs = [
        ("-gen-op-decls", "Ops.h.inc"),
        ("-gen-op-defs", "Ops.cpp.inc"),
    ],
    tblgen = "//third_party/llvm/llvm/projects/google_mlir:mlir-tblgen",
    td_file = "Ops.td",
    td_srcs = [
        ":td_files",
        "//third_party/llvm/llvm/projects/google_mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "StructureOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "StructureOps.h.inc"),
        ("-gen-op-defs", "StructureOps.cpp.inc"),
    ],
    tblgen = "//third_party/llvm/llvm/projects/google_mlir:mlir-tblgen",
    td_file = "StructureOps.td",
    td_srcs = [
        ":td_files",
        "//third_party/llvm/llvm/projects/google_mlir:OpBaseTdFiles",
    ],
)
