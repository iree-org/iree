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
        "CheckOps.cpp",
        "CheckOps.cpp.inc",
    ],
    hdrs = [
        "CheckOps.h",
        "CheckOps.h.inc",
    ],
    deps = [
        ":check_ops_gen",
        "//iree/compiler/Dialect/HAL/IR",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
)

cc_library(
    name = "CheckDialect",
    srcs = [
        "CheckDialect.cpp",
    ],
    hdrs = [
        "CheckDialect.h",
    ],
    deps = [
        ":IR",
        ":check_ops_gen",
        "//iree/compiler/Dialect/HAL/Conversion",
        "//iree/compiler/Dialect/Modules/Check:check_imports",
        "//iree/compiler/Dialect/Modules/Check/Conversion",
        "//iree/compiler/Dialect/VM/Conversion",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl(
    name = "check_ops_gen",
    tbl_outs = [
        ("-gen-op-decls", "CheckOps.h.inc"),
        ("-gen-op-defs", "CheckOps.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CheckOps.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "//iree/compiler/Dialect/HAL/IR:td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
    ],
)

iree_tablegen_doc(
    name = "CheckDialectDocGen",
    tbl_outs = [
        ("-gen-dialect-doc", "CheckDialect.md"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CheckOps.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "//iree/compiler/Dialect/HAL/IR:td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
    ],
)
