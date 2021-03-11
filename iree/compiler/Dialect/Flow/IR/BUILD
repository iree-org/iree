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
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "td_files",
    srcs = enforce_glob(
        [
            "FlowBase.td",
            "FlowOps.td",
        ],
        include = ["*.td"],
    ),
)

cc_library(
    name = "IR",
    srcs = [
        "FlowDialect.cpp",
        "FlowEnums.cpp.inc",
        "FlowInterfaces.cpp.inc",
        "FlowOpFolders.cpp",
        "FlowOpUtils.cpp",
        "FlowOps.cpp",
        "FlowOps.cpp.inc",
        "FlowTypes.cpp",
    ],
    hdrs = [
        "FlowDialect.h",
        "FlowEnums.h.inc",
        "FlowInterfaces.h.inc",
        "FlowOpUtils.h",
        "FlowOps.h",
        "FlowOps.h.inc",
        "FlowTypes.h",
    ],
    deps = [
        ":FlowEnumsGen",
        ":FlowInterfacesGen",
        ":FlowOpsGen",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Shape/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl(
    name = "FlowEnumsGen",
    tbl_outs = [
        ("-gen-enum-decls", "FlowEnums.h.inc"),
        ("-gen-enum-defs", "FlowEnums.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FlowBase.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "//iree/compiler/Dialect/Shape/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
    ],
)

gentbl(
    name = "FlowInterfacesGen",
    tbl_outs = [
        ("-gen-op-interface-decls", "FlowInterfaces.h.inc"),
        ("-gen-op-interface-defs", "FlowInterfaces.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FlowInterfaces.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "//iree/compiler/Dialect/Shape/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
    ],
)

gentbl(
    name = "FlowOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "FlowOps.h.inc"),
        ("-gen-op-defs", "FlowOps.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FlowOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "//iree/compiler/Dialect/Shape/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
    ],
)

iree_tablegen_doc(
    name = "FlowDialectDocGen",
    tbl_outs = [
        ("-gen-dialect-doc", "FlowDialect.md"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "FlowOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "//iree/compiler/Dialect/Shape/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
    ],
)
