# Copyright 2020 Google LLC
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

filegroup(
    name = "td_files",
    srcs = glob(["*.td"]),
)

cc_library(
    name = "IR",
    srcs = [
        "SequenceDialect.cpp",
        "SequenceEnums.cpp.inc",
        "SequenceOpInterface.cpp.inc",
        "SequenceOps.cpp",
        "SequenceOps.cpp.inc",
        "SequenceTypes.cpp",
    ],
    hdrs = [
        "SequenceDialect.h",
        "SequenceEnums.h.inc",
        "SequenceOpInterface.h.inc",
        "SequenceOps.h",
        "SequenceOps.h.inc",
        "SequenceTypes.h",
    ],
    deps = [
        ":SequenceEnumsGen",
        ":SequenceOpInterfaceGen",
        ":SequenceOpsGen",
        "//iree/compiler/Dialect/IREE/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

gentbl(
    name = "SequenceEnumsGen",
    tbl_outs = [
        ("-gen-enum-decls", "SequenceEnums.h.inc"),
        ("-gen-enum-defs", "SequenceEnums.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SequenceBase.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "SequenceOpInterfaceGen",
    tbl_outs = [
        ("-gen-op-interface-decls", "SequenceOpInterface.h.inc"),
        ("-gen-op-interface-defs", "SequenceOpInterface.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SequenceBase.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl(
    name = "SequenceOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "SequenceOps.h.inc"),
        ("-gen-op-defs", "SequenceOps.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SequenceOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/IR/SymbolInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
    ],
)

iree_tablegen_doc(
    name = "SequenceDialectDocGen",
    tbl_outs = [
        ("-gen-dialect-doc", "SequenceDialect.md"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "SequenceOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/IR/SymbolInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
    ],
)
