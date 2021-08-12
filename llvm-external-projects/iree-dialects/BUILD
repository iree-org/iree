load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],
)

exports_files(glob(["include/iree-dialects/Dialect/IREE/*.td"]))

filegroup(
    name = "TdFilegroup",
    srcs = glob(["include/iree-dialects/Dialect/IREE/*.td"]),
)

td_library(
    name = "TdFiles",
    srcs = glob(["include/iree-dialects/Dialect/IREE/*.td"]),
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
    ],
)

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
        "@llvm-project//mlir:CAPIIR",
    ],
)
