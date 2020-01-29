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

load("//build_tools/embed_data:build_defs.bzl", "cc_embed_data")
load("//build_tools/bazel:tblgen.bzl", "gentbl")

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
        "check_dialect.cc",
        "check_ops.cc",
        "check_ops.cc.inc",
        "conversion_patterns.cc",
    ],
    hdrs = [
        "check_dialect.h",
        "check_ops.h",
        "check_ops.h.inc",
        "conversion_patterns.h",
    ],
    deps = [
        ":check_imports",
        ":check_ops_gen",
        "//iree/compiler/Dialect/VM/Conversion",
        "//iree/samples/custom_modules/dialect:custom_imports",
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
    name = "check_ops_gen",
    tbl_outs = [
        ("-gen-op-decls", "check_ops.h.inc"),
        ("-gen-op-defs", "check_ops.cc.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "check_ops.td",
    td_srcs = [
        ":td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

cc_embed_data(
    name = "check_imports",
    srcs = ["check.imports.mlir"],
    cc_file_output = "check.imports.cc",
    cpp_namespace = "mlir::iree_compiler::IREE::Check",
    flatten = True,
    h_file_output = "check.imports.h",
)

cc_binary(
    name = "check-opt",
    deps = [
        ":dialect",
        "//iree/tools:iree_opt_library",
        "@llvm-project//mlir:MlirOptMain",
    ],
)

cc_binary(
    name = "check-translate",
    deps = [
        ":dialect",
        "//iree/tools:iree_translate_library",
    ],
)
