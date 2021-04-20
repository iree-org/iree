# Copyright 2021 Google LLC
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
load("//build_tools/bazel:tblgen.bzl", "gentbl_cc_library")
load("//build_tools/bazel:enforce_glob.bzl", "enforce_glob")

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["VMLXOps.td"])

filegroup(
    name = "td_files",
    srcs = enforce_glob(
        [
            "VMVXBase.td",
            "VMVXOps.td",
        ],
        include = ["*.td"],
    ),
)

cc_library(
    name = "IR",
    srcs = [
        "VMVXEnums.cpp.inc",
        "VMVXOpInterface.cpp.inc",
        "VMVXOps.cpp",
        "VMVXTypes.cpp",
    ],
    hdrs = [
        "VMVXEnums.h.inc",
        "VMVXOpInterface.h.inc",
        "VMVXOps.h",
        "VMVXOps.h.inc",
        "VMVXTypes.h",
    ],
    textual_hdrs = [
        "VMVXOps.cpp.inc",
    ],
    deps = [
        ":VMVXEnumsGen",
        ":VMVXOpInterfaceGen",
        ":VMVXOpsGen",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/VM/IR",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Translation",
    ],
)

cc_library(
    name = "VMVXDialect",
    srcs = ["VMVXDialect.cpp"],
    hdrs = ["VMVXDialect.h"],
    deps = [
        ":IR",
        "//iree/compiler/Dialect/Modules/VMVX:vmvx_imports",
        "//iree/compiler/Dialect/Modules/VMVX/Conversion/VMVXToVM",
        "//iree/compiler/Dialect/VM/Conversion",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

gentbl_cc_library(
    name = "VMVXEnumsGen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "VMVXEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "VMVXEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMVXBase.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "//iree/compiler/Dialect/Shape/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "VMVXOpsGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "VMVXOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "VMVXOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMVXOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "//iree/compiler/Dialect/Shape/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

gentbl_cc_library(
    name = "VMVXOpInterfaceGen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "VMVXOpInterface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "VMVXOpInterface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMVXBase.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "//iree/compiler/Dialect/Shape/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

iree_tablegen_doc(
    name = "VMVXDialecDocGen",
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
            "VMVXDialect.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "VMVXOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "//iree/compiler/Dialect/Shape/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)
