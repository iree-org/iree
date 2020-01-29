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

load("//build_tools/bazel:tblgen.bzl", "gentbl")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

gentbl(
    name = "IndexComputationAttrGen",
    tbl_outs = [
        (
            "-gen-struct-attr-decls",
            "IndexComputationAttr.h.inc",
        ),
        (
            "-gen-struct-attr-defs",
            "IndexComputationAttr.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "IndexComputationAttr.td",
    td_srcs = [
        "//iree/compiler/Dialect/IREE/IR:td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:AffineOpsTdFiles",
    ],
)

cc_library(
    name = "IndexComputation",
    srcs = [
        "IREEIndexComputation.cpp",
        "IndexComputation.cpp",
        "IndexComputationAttr.cpp.inc",
        "IndexComputationAttribute.cpp",
        "IndexComputationPass.cpp",
        "XLAIndexPropagation.cpp",
    ],
    hdrs = [
        "IREEIndexComputation.h",
        "IndexComputation.h",
        "IndexComputationAttr.h.inc",
        "IndexComputationAttribute.h",
        "IndexComputationPass.h",
        "XLAIndexPropagation.h",
    ],
    deps = [
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Translation/SPIRV/IndexComputation:IndexComputationAttrGen",
        "//iree/compiler/Utils",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SPIRVDialect",
        "@llvm-project//mlir:StandardOps",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo",
    ],
    alwayslink = 1,
)
