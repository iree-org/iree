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

gentbl(
    name = "ops_gen",
    tbl_outs = [
        ("-gen-op-decls", "ops.h.inc"),
        ("-gen-op-defs", "ops.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl(
    name = "op_interface_gen",
    tbl_outs = [
        ("-gen-op-interface-decls", "op_interface.h.inc"),
        ("-gen-op-interface-defs", "op_interface.cpp.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "base.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

cc_library(
    name = "ir",
    deps = [
        ":dialect",
        ":ops",
    ],
)

cc_library(
    name = "dialect",
    srcs = [
        "dialect.cpp",
    ],
    hdrs = [
        "dialect.h",
    ],
    deps = [
        ":op_interface_gen",
        ":ops_gen",
        "//integrations/tensorflow/compiler/dialect/tf_strings/conversion:convert_flow_to_hal",
        "//integrations/tensorflow/compiler/dialect/tf_strings/ir:ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ops",
    srcs = [
        "ops.cpp",
        "ops.cpp.inc",
    ],
    hdrs = [
        "ops.h",
        "ops.h.inc",
        "types.h",
    ],
    deps = [
        ":op_interface_gen",
        ":ops_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)
