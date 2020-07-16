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

exports_files(["TensorListBase.td"])

filegroup(
    name = "td_files",
    srcs = glob(["*.td"]),
)

cc_library(
    name = "IR",
    srcs = [
        "TensorListOps.cpp",
        "TensorListTypes.cpp",
    ],
    hdrs = [
        "TensorListOps.h",
        "TensorListOps.h.inc",
        "TensorListTypes.h",
    ],
    textual_hdrs = [
        "TensorListOps.cpp.inc",
    ],
    deps = [
        ":TensorListOpsGen",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/IREE/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "TensorListDialect",
    srcs = ["TensorListDialect.cpp"],
    hdrs = ["TensorListDialect.h"],
    deps = [
        ":IR",
        "//iree/compiler/Dialect/HAL/Conversion",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Modules/TensorList:tensorlist_imports",
        "//iree/compiler/Dialect/Modules/TensorList/Conversion",
        "//iree/compiler/Dialect/VM/Conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl(
    name = "TensorListOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "TensorListOps.h.inc"),
        ("-gen-op-defs", "TensorListOps.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TensorListOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "//iree/compiler/Dialect/HAL/IR:td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

iree_tablegen_doc(
    name = "TensorListDialectDocGen",
    tbl_outs = [
        ("-gen-dialect-doc", "TensorListDialect.md"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "TensorListOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "//iree/compiler/Dialect/HAL/IR:td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)
