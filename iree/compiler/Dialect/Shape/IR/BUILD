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

exports_files(["ShapeBase.td"])

filegroup(
    name = "td_files",
    srcs = glob(["*.td"]),
)

cc_library(
    name = "IR",
    srcs = [
        "Builders.cpp",
        "Folders.cpp",
        "ShapeDialect.cpp",
        "ShapeInterface.cpp",
        "ShapeOps.cpp",
        "ShapeOps.cpp.inc",
        "ShapeTypes.cpp",
    ],
    hdrs = [
        "Builders.h",
        "ShapeDialect.h",
        "ShapeInterface.h",
        "ShapeOps.h",
        "ShapeOps.h.inc",
        "ShapeTypes.h",
    ],
    deps = [
        ":ShapeOpsGen",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

gentbl(
    name = "ShapeOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "ShapeOps.h.inc"),
        ("-gen-op-defs", "ShapeOps.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ShapeOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/ViewLikeInterface.td",
        "@llvm-project//mlir:include/mlir/IR/OpAsmInterface.td",
    ],
)

iree_tablegen_doc(
    name = "ShapeDialectDocGen",
    tbl_outs = [
        ("-gen-dialect-doc", "ShapeDialect.md"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ShapeOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/ViewLikeInterface.td",
        "@llvm-project//mlir:include/mlir/IR/OpAsmInterface.td",
    ],
)
