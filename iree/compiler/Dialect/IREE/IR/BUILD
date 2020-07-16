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

exports_files(["IREEBase.td"])

filegroup(
    name = "td_files",
    srcs = glob(["*.td"]),
)

cc_library(
    name = "IR",
    srcs = [
        "IREEDialect.cpp",
        "IREEOps.cpp",
        "IREEOps.cpp.inc",
        "IREETypes.cpp",
    ],
    hdrs = [
        "IREEAttributes.h",
        "IREEDialect.h",
        "IREEOps.h",
        "IREEOps.h.inc",
        "IREETraits.h",
        "IREETypes.h",
    ],
    deps = [
        ":IREEOpsGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
    ],
)

gentbl(
    name = "IREEOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "IREEOps.h.inc"),
        ("-gen-op-defs", "IREEOps.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IREEOps.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
    ],
)

iree_tablegen_doc(
    name = "IREEDialectDocGen",
    tbl_outs = [
        ("-gen-dialect-doc", "IREEDialect.md"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IREEOps.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffectInterfaces.td",
    ],
)
