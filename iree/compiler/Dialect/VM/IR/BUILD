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

load("@local_config_mlir//:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "td_files",
    srcs = glob(["*.td"]),
)

cc_library(
    name = "IR",
    srcs = [
        "VMDialect.cpp",
        "VMEnums.cpp.inc",
        "VMOpEncoder.cpp.inc",
        "VMOpFolders.cpp",
        "VMOpInterface.cpp.inc",
        "VMOps.cpp",
        "VMOps.cpp.inc",
        "VMTypes.cpp",
    ],
    hdrs = [
        "VMDialect.h",
        "VMEnums.h.inc",
        "VMFuncEncoder.h",
        "VMOpInterface.h.inc",
        "VMOps.h",
        "VMOps.h.inc",
        "VMTraits.h",
        "VMTypes.h",
    ],
    deps = [
        ":VMEnumsGen",
        ":VMOpEncoderGen",
        ":VMOpInterfaceGen",
        ":VMOpsGen",
        "//iree/compiler/Dialect",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TransformUtils",
    ],
    alwayslink = 1,
)

gentbl(
    name = "VMEnumsGen",
    tbl_outs = [
        ("-gen-enum-decls", "VMEnums.h.inc"),
        ("-gen-enum-defs", "VMEnums.cpp.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "VMBase.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect:td_files",
        "@local_config_mlir//:OpBaseTdFiles",
        "@local_config_mlir//:include/mlir/Analysis/CallInterfaces.td",
    ],
)

gentbl(
    name = "VMOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "VMOps.h.inc"),
        ("-gen-op-defs", "VMOps.cpp.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "VMOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect:td_files",
        "@local_config_mlir//:OpBaseTdFiles",
        "@local_config_mlir//:include/mlir/Analysis/CallInterfaces.td",
    ],
)

gentbl(
    name = "VMOpEncoderGen",
    tbl_outs = [
        ("-gen-iree-vm-op-encoder-defs", "VMOpEncoder.cpp.inc"),
    ],
    tblgen = "//iree/tools:iree-tblgen",
    td_file = "VMOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect:td_files",
        "@local_config_mlir//:OpBaseTdFiles",
        "@local_config_mlir//:include/mlir/Analysis/CallInterfaces.td",
    ],
)

gentbl(
    name = "VMOpInterfaceGen",
    tbl_outs = [
        ("-gen-op-interface-decls", "VMOpInterface.h.inc"),
        ("-gen-op-interface-defs", "VMOpInterface.cpp.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "VMBase.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect:td_files",
        "@local_config_mlir//:OpBaseTdFiles",
        "@local_config_mlir//:include/mlir/Analysis/CallInterfaces.td",
    ],
)
