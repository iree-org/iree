load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],
)

exports_files(glob(["include/iree-dialects/Dialect/IREE/*.td"]))

exports_files(glob(["python/*.cpp"]))

################################################################################
# Package-by-package filegroup exports.
# Some tools can make use of such non-nested groups.
################################################################################

filegroup(
    name = "python_dialects",
    srcs = glob(["python/iree/compiler/dialects/*.py"]) + [
        ":python/iree/compiler/_iree_ops_gen.py",
        ":python/iree/compiler/_iree_pydm_ops_gen.py",
    ],
)

filegroup(
    name = "python_dialects_iree_pydm",
    srcs = glob(["python/iree/compiler/dialects/iree_pydm/*.py"]),
)

filegroup(
    name = "python_dialects_iree_pydm_importer",
    srcs = glob(["python/iree/compiler/dialects/iree_pydm/importer/*.py"]),
)

filegroup(
    name = "python_dialects_iree_pydm_rtl",
    srcs = glob(["python/iree/compiler/dialects/iree_pydm/rtl/*.py"]),
)

filegroup(
    name = "python_dialects_iree_pydm_rtl_modules",
    srcs = glob(["python/iree/compiler/dialects/iree_pydm/rtl/modules/*.py"]),
)

################################################################################
# Tablegen exports
################################################################################

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
        "python/iree/compiler/dialects/*.td",
    ]) + [
        "@llvm-project//mlir:include/mlir/Bindings/Python/Attributes.td",
    ],
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

gentbl_filegroup(
    name = "IREEDialectPyGen",
    tbl_outs = [
        (
            [
                "-gen-python-op-bindings",
                "-bind-dialect=iree",
            ],
            "python/iree/compiler/_iree_ops_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "python/iree/compiler/dialects/IreeBinding.td",
    deps = [
        ":TdFiles",
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

gentbl_filegroup(
    name = "IREEPyDMDialectPyGen",
    tbl_outs = [
        (
            [
                "-gen-python-op-bindings",
                "-bind-dialect=iree_pydm",
            ],
            "python/iree/compiler/_iree_pydm_ops_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "python/iree/compiler/dialects/IreePyDmBinding.td",
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
    hdrs = glob([
        "include/iree-dialects/Dialect/IREEPyDM/IR/*.h",
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

cc_library(
    name = "IREEPyDMTransforms",
    srcs = glob([
        "lib/Dialect/IREEPyDM/Transforms/RTL/*.cpp",
        "lib/Dialect/IREEPyDM/Transforms/ToIREE/*.cpp",
    ]),
    hdrs = glob([
        "include/iree-dialects/Dialect/IREEPyDM/Transforms/*.h",
        "include/iree-dialects/Dialect/IREEPyDM/Transforms/**/*.h",
        "lib/Dialect/IREEPyDM/Transforms/*.h",
    ]),
    deps = [
        ":IREEDialect",
        ":IREEPyDMDialect",
        ":IREEPyDMTransformsIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

################################################################################
# CAPI
################################################################################

cc_library(
    name = "CAPI",
    srcs = glob(["lib/CAPI/*.cpp"]),
    hdrs = glob(["include/iree-dialects-c/*.h"]),
    includes = ["include"],
    deps = [
        ":IREEDialect",
        ":IREEPyDMDialect",
        ":IREEPyDMTransforms",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
