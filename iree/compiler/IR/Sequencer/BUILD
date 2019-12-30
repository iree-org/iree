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

filegroup(
    name = "td_files",
    srcs = glob(["*.td"]),
)

cc_library(
    name = "Sequencer",
    srcs = [
        "HLDialect.cpp",
        "HLOps.cpp",
        "HLOps.cpp.inc",
        "LLDialect.cpp",
        "LLOps.cpp",
        "LLOps.cpp.inc",
        "OpWriters.cpp",
    ],
    hdrs = [
        "HLDialect.h",
        "HLOps.h",
        "HLOps.h.inc",
        "LLDialect.h",
        "LLOps.h",
        "LLOps.h.inc",
        "OpWriters.h",
    ],
    deps = [
        ":HLOpsGen",
        ":LLOpsGen",
        "//iree/compiler/IR",
        "//iree/compiler/Serialization",
        "//iree/compiler/Utils",
        "//iree/schemas/bytecode:sequencer_bytecode_v0",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

gentbl(
    name = "HLOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "HLOps.h.inc"),
        ("-gen-op-defs", "HLOps.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "HLOps.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:include/mlir/IR/OpBase.td",
        "//iree/compiler/IR:OpBase.td",
    ],
)

gentbl(
    name = "LLOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "LLOps.h.inc"),
        ("-gen-op-defs", "LLOps.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "LLOps.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:include/mlir/IR/OpBase.td",
        "//iree/compiler/IR:OpBase.td",
    ],
)
