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

load("@llvm-project//mlir:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["OpBase.td"])

filegroup(
    name = "td_files",
    srcs = glob(["*.td"]),
)

cc_library(
    name = "IR",
    srcs = [
        "ConfigOps.cpp",
        "ConfigOps.cpp.inc",
        "Dialect.cpp",
        "Enums.cpp.inc",
        "Ops.cpp",
        "Ops.cpp.inc",
        "StructureOps.cpp",
        "StructureOps.cpp.inc",
        "Types.cpp",
    ],
    hdrs = [
        "ConfigOps.h",
        "ConfigOps.h.inc",
        "Dialect.h",
        "Enums.h.inc",
        "Ops.h",
        "Ops.h.inc",
        "StructureOps.h",
        "StructureOps.h.inc",
        "Types.h",
    ],
    deps = [
        ":ConfigOpsGen",
        ":EnumsGen",
        ":OpsGen",
        ":StructureOpsGen",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

gentbl(
    name = "ConfigOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "ConfigOps.h.inc"),
        ("-gen-op-defs", "ConfigOps.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ConfigOps.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "EnumsGen",
    tbl_outs = [
        ("-gen-enum-decls", "Enums.h.inc"),
        ("-gen-enum-defs", "Enums.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "OpBase.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "OpsGen",
    tbl_outs = [
        ("-gen-op-decls", "Ops.h.inc"),
        ("-gen-op-defs", "Ops.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Ops.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "StructureOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "StructureOps.h.inc"),
        ("-gen-op-defs", "StructureOps.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "StructureOps.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)
