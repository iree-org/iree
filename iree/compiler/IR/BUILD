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

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

load("@local_config_mlir//:tblgen.bzl", "gentbl")

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
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
    ],
    alwayslink = 1,
)

gentbl(
    name = "ConfigOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "ConfigOps.h.inc"),
        ("-gen-op-defs", "ConfigOps.cpp.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "ConfigOps.td",
    td_srcs = [
        ":td_files",
        "@local_config_mlir//:OpBaseTdFiles",
    ],
)

gentbl(
    name = "EnumsGen",
    tbl_outs = [
        ("-gen-enum-decls", "Enums.h.inc"),
        ("-gen-enum-defs", "Enums.cpp.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "OpBase.td",
    td_srcs = [
        ":td_files",
        "@local_config_mlir//:OpBaseTdFiles",
    ],
)

gentbl(
    name = "OpsGen",
    tbl_outs = [
        ("-gen-op-decls", "Ops.h.inc"),
        ("-gen-op-defs", "Ops.cpp.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "Ops.td",
    td_srcs = [
        ":td_files",
        "@local_config_mlir//:OpBaseTdFiles",
    ],
)

gentbl(
    name = "StructureOpsGen",
    tbl_outs = [
        ("-gen-op-decls", "StructureOps.h.inc"),
        ("-gen-op-defs", "StructureOps.cpp.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "StructureOps.td",
    td_srcs = [
        ":td_files",
        "@local_config_mlir//:OpBaseTdFiles",
    ],
)
