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
        "FlowDialect.cpp",
        "FlowEnums.cpp.inc",
        "FlowOpFolders.cpp",
        "FlowOps.cpp",
        "FlowOps.cpp.inc",
        "FlowTypes.cpp",
    ],
    hdrs = [
        "FlowDialect.h",
        "FlowEnums.h.inc",
        "FlowOps.h",
        "FlowOps.h.inc",
        "FlowTypes.h",
    ],
    deps = [
        ":FlowEnumsGen",
        ":FlowOpsGen",
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
    name = "FlowEnumsGen",
    tbl_outs = [
        ("-gen-enum-decls", "FlowEnums.h.inc"),
        ("-gen-enum-defs", "FlowEnums.cpp.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "FlowBase.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect:td_files",
        "@local_config_mlir//:OpBaseTdFiles",
    ],
)

gentbl(
    name = "FlowOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "FlowOps.h.inc"),
        ("-gen-op-defs", "FlowOps.cpp.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "FlowOps.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect:td_files",
        "@local_config_mlir//:OpBaseTdFiles",
    ],
)
