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

load("//build_tools/embed_data:build_defs.bzl", "cc_embed_data")
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
    name = "dialect",
    srcs = [
        "conversion_patterns.cc",
        "custom_dialect.cc",
        "custom_ops.cc.inc",
    ],
    hdrs = [
        "conversion_patterns.h",
        "custom_dialect.h",
        "custom_ops.h.inc",
    ],
    deps = [
        ":custom_imports",
        ":custom_ops_gen",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/VM/Conversion",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

gentbl(
    name = "custom_ops_gen",
    tbl_outs = [
        ("-gen-op-decls", "custom_ops.h.inc"),
        ("-gen-op-defs", "custom_ops.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "custom_ops.td",
    td_srcs = [
        ":td_files",
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

cc_embed_data(
    name = "custom_imports",
    srcs = ["custom.imports.mlir"],
    cc_file_output = "custom.imports.cc",
    cpp_namespace = "mlir::iree_compiler::IREE::Custom",
    flatten = True,
    h_file_output = "custom.imports.h",
)

cc_binary(
    name = "custom-opt",
    deps = [
        ":dialect",
        "//iree/tools:iree_opt_library",
        "@llvm-project//mlir:MlirOptMain",
    ],
)

cc_binary(
    name = "custom-translate",
    deps = [
        ":dialect",
        "//iree/tools:iree_translate_library",
    ],
)
