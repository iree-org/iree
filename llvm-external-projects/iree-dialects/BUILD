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
        "include/iree-dialects/Dialect/IREEPyDM/Transforms/*.td",
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
            "include/iree-dialects/Dialect/IREEPyDM/IR/Dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "include/iree-dialects/Dialect/IREEPyDM/IR/Dialect.cpp.inc",
        ),
        (
            ["-gen-op-decls"],
            "include/iree-dialects/Dialect/IREEPyDM/IR/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/iree-dialects/Dialect/IREEPyDM/IR/Ops.cpp.inc",
        ),
        (
            ["-gen-typedef-decls"],
            "include/iree-dialects/Dialect/IREEPyDM/IR/Types.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "include/iree-dialects/Dialect/IREEPyDM/IR/Types.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/iree-dialects/Dialect/IREEPyDM/IR/Ops.td",
    deps = [
        ":TdFiles",
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "IREEPyDMInterfacesIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-type-interface-decls"],
            "include/iree-dialects/Dialect/IREEPyDM/IR/TypeInterfaces.h.inc",
        ),
        (
            ["-gen-type-interface-defs"],
            "include/iree-dialects/Dialect/IREEPyDM/IR/TypeInterfaces.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/iree-dialects/Dialect/IREEPyDM/IR/Interfaces.td",
    deps = [
        ":TdFiles",
    ],
)

gentbl_cc_library(
    name = "IREEPyDMTransformsIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["-gen-pass-decls"],
            "include/iree-dialects/Dialect/IREEPyDM/Transforms/Passes.h.inc",
        ),
        (
            ["-gen-pass-capi-header"],
            "include/iree-dialects/Dialect/IREEPyDM/Transforms/Passes.capi.h.inc",
        ),
        (
            ["-gen-pass-capi-impl"],
            "include/iree-dialects/Dialect/IREEPyDM/Transforms/Passes.capi.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/iree-dialects/Dialect/IREEPyDM/Transforms/Passes.td",
    deps = [
        ":TdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "IREEPyDMDialect",
    srcs = glob([
        "lib/Dialect/IREEPyDM/IR/*.cpp",
    ]),
    hdrs = glob([
        "include/iree-dialects/Dialect/IREEPyDM/IR/*.h",
        "include/iree-dialects/Dialect/IREEPyDM/Transforms/*.h",
    ]),
    includes = ["include"],
    deps = [
        ":IREEPyDMInterfacesIncGen",
        ":IREEPyDMOpsIncGen",
        ":IREEPyDMTransformsIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
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
