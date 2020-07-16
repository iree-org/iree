# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//build_tools/bazel:iree_tablegen_doc.bzl", "iree_tablegen_doc")
load("//build_tools/bazel:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["HALBase.td"])

filegroup(
    name = "td_files",
    srcs = glob(["*.td"]),
)

cc_library(
    name = "IR",
    srcs = [
        "HALOpFolders.cpp",
        "HALOps.cpp",
        "HALTypes.cpp",
    ],
    hdrs = [
        "HALOps.h",
        "HALTypes.h",
    ],
    textual_hdrs = [
        "HALEnums.cpp.inc",
        "HALEnums.h.inc",
        "HALOps.cpp.inc",
        "HALOps.h.inc",
        "HALOpInterface.cpp.inc",
        "HALOpInterface.h.inc",
        "HALStructs.cpp.inc",
        "HALStructs.h.inc",
    ],
    deps = [
        ":HALOpsGen",
        ":HALStructsGen",
        ":HALTypesGen",
        "//iree/compiler/Dialect/IREE/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "HALDialect",
    srcs = ["HALDialect.cpp"],
    hdrs = ["HALDialect.h"],
    deps = [
        ":IR",
        "//iree/compiler/Dialect/HAL:hal_imports",
        "//iree/compiler/Dialect/HAL/Conversion/HALToVM",
        "//iree/compiler/Dialect/VM/Conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl(
    name = "HALOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "HALOps.h.inc"),
        ("-gen-op-defs", "HALOps.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HALOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:include/mlir/IR/SymbolInterfaces.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl(
    name = "HALStructsGen",
    tbl_outs = [
        ("-gen-iree-struct-attr-decls", "HALStructs.h.inc"),
        ("-gen-iree-struct-attr-defs", "HALStructs.cpp.inc"),
    ],
    tblgen = "//iree/tools:iree-tblgen",
    td_file = "HALBase.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "HALTypesGen",
    tbl_outs = [
        ("-gen-enum-decls", "HALEnums.h.inc"),
        ("-gen-enum-defs", "HALEnums.cpp.inc"),
        ("-gen-op-interface-decls", "HALOpInterface.h.inc"),
        ("-gen-op-interface-defs", "HALOpInterface.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HALBase.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

iree_tablegen_doc(
    name = "HALDialectDocGen",
    tbl_outs = [
        ("-gen-dialect-doc", "HALDialect.md"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HALOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:include/mlir/IR/SymbolInterfaces.td",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)
