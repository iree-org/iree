load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],
)

exports_files(glob(["include/iree-dialects/Dialect/Input/*.td"]))

exports_files(glob(["python/*.cpp"]))

################################################################################
# Package-by-package filegroup exports.
# Some tools can make use of such non-nested groups.
################################################################################

filegroup(
    name = "python_dialects",
    srcs = glob(["python/iree/compiler/dialects/**/*.py"]) + [
        ":python/iree/compiler/dialects/_iree_ops_gen.py",
    ],
)

################################################################################
# Tablegen exports
################################################################################

filegroup(
    name = "TdFilegroup",
    srcs = glob([
        "include/iree-dialects/Dialect/Input/*.td",
    ]),
)

td_library(
    name = "TdFiles",
    srcs = glob([
        "include/iree-dialects/Dialect/Input/*.td",
        "python/iree/compiler/dialects/*.td",
    ]) + [
        "@llvm-project//mlir:include/mlir/Bindings/Python/Attributes.td",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PDLDialectTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:TransformDialectTdFiles",
    ],
)

################################################################################
# IREEInput dialect
################################################################################

gentbl_cc_library(
    name = "IREEInputIncGen",
    strip_include_prefix = "include",
    tbl_outs = [
        (
            ["--gen-dialect-decls"],
            "include/iree-dialects/Dialect/Input/InputDialect.h.inc",
        ),
        (
            ["--gen-dialect-defs"],
            "include/iree-dialects/Dialect/Input/InputDialect.cpp.inc",
        ),
        (
            ["--gen-op-decls"],
            "include/iree-dialects/Dialect/Input/InputOps.h.inc",
        ),
        (
            ["--gen-op-defs"],
            "include/iree-dialects/Dialect/Input/InputOps.cpp.inc",
        ),
        (
            ["--gen-typedef-decls"],
            "include/iree-dialects/Dialect/Input/InputTypes.h.inc",
        ),
        (
            ["--gen-typedef-defs"],
            "include/iree-dialects/Dialect/Input/InputTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/iree-dialects/Dialect/Input/InputOps.td",
    deps = [":TdFiles"],
)

cc_library(
    name = "IREEInputDialect",
    srcs = glob([
        "lib/Dialect/Input/*.cpp",
    ]),
    hdrs = glob(["include/iree-dialects/Dialect/Input/*.h"]),
    includes = ["include"],
    deps = [
        ":IREEInputIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_filegroup(
    name = "IREEInputDialectPyGen",
    tbl_outs = [
        (
            [
                "--gen-python-op-bindings",
                "--bind-dialect=iree",
            ],
            "python/iree/compiler/dialects/_iree_ops_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "python/iree/compiler/dialects/IreeInputBinding.td",
    deps = [
        ":TdFiles",
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
        ":IREEInputDialect",
        "@llvm-project//mlir:CAPIIR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransformOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformDialect",
    ],
)

################################################################################
# Test lib
################################################################################

cc_library(
    name = "IREEDialectsTest",
    srcs = glob([
        "test/lib/**/*.cpp",
    ]),
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:Transforms",
    ],
)

################################################################################
# Tools
################################################################################

cc_binary(
    name = "iree-dialects-opt",
    srcs = [
        "tools/iree-dialects-opt/iree-dialects-opt.cpp",
    ],
    tags = ["hostonly"],
    deps = [
        ":IREEDialectsTest",
        ":IREEInputDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgTransformOps",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:PDLDialect",
        "@llvm-project//mlir:PDLInterpDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:SCFTransformOps",
        "@llvm-project//mlir:SCFTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformDialect",
        "@llvm-project//mlir:Transforms",
    ],
)
