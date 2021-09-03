load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],
)

exports_files(glob(["include/iree-dialects/Dialect/IREE/*.td"]))

filegroup(
    name = "TdFilegroup",
    srcs = glob([
        "include/iree-dialects/Dialect/IREE/*.td",
        "include/iree-dialects/Dialect/IREEPyDM/IR/*.td",
    ]),
)

td_library(
    name = "TdFiles",
    srcs = glob([
        "include/iree-dialects/Dialect/IREE/*.td",
        "include/iree-dialects/Dialect/IREEPyDM/IR/*.td",
    ]),
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
    ],
)

################################################################################
# IREE dialect
################################################################################

gentbl_cc_library(
    name = "IREEOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "include/iree-dialects/Dialect/IREE/IREEOpsDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "include/iree-dialects/Dialect/IREE/IREEOpsDialect.cpp.inc",
        ),
        (
            ["-gen-op-decls"],
            "include/iree-dialects/Dialect/IREE/IREEOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/iree-dialects/Dialect/IREE/IREEOps.cpp.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "include/iree-dialects/Dialect/IREE/IREEOpsTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "include/iree-dialects/Dialect/IREE/IREEOpsTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/iree-dialects/Dialect/IREE/IREEOps.td",
    deps = [":TdFiles"],
)

cc_library(
    name = "IREEDialect",
    srcs = glob([
        "lib/Dialect/IREE/*.cpp",
    ]),
    hdrs = glob(["include/iree-dialects/Dialect/IREE/*.h"]),
    includes = ["include"],
    deps = [
        ":IREEOpsIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

################################################################################
# IREEPyDM Dialect
################################################################################

gentbl_cc_library(
    name = "IREEPyDMOpsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "include/iree-dialects/Dialect/IREEPyDM/IR/IREEPyDMOpsDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "include/iree-dialects/Dialect/IREEPyDM/IR/IREEPyDMOpsDialect.cpp.inc",
        ),
        (
            ["-gen-op-decls"],
            "include/iree-dialects/Dialect/IREEPyDM/IR/IREEPyDMOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/iree-dialects/Dialect/IREEPyDM/IR/IREEPyDMOps.cpp.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "include/iree-dialects/Dialect/IREEPyDM/IR/IREEPyDMOpsTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "include/iree-dialects/Dialect/IREEPyDM/IR/IREEPyDMOpsTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/iree-dialects/Dialect/IREEPyDM/IR/IREEPyDMOps.td",
    deps = [
        ":TdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
    ],
)

cc_library(
    name = "IREEPyDMDialect",
    srcs = glob([
        "lib/Dialect/IREEPyDM/IR/*.cpp",
    ]),
    hdrs = glob(["include/iree-dialects/Dialect/IREEPyDM/IR/*.h"]),
    includes = ["include"],
    deps = [
        ":IREEPyDMOpsIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

################################################################################
# CAPI
################################################################################

cc_library(
    name = "CAPI",
    srcs = [
        "lib/CAPI/Dialects.cpp",
    ],
    hdrs = [
        "include/iree-dialects-c/Dialects.h",
    ],
    deps = [
        ":IREEDialect",
        ":IREEPyDMDialect",
        "@llvm-project//mlir:CAPIIR",
    ],
)
