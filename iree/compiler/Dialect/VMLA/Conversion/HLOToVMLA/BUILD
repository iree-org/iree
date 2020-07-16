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

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "HLOToVMLA",
    srcs = [
        "ConvertConvOps.cpp",
        "ConvertHLOToVMLA.cpp",
        "ConvertReductionOps.cpp",
    ],
    hdrs = [
        "ConvertHLOToVMLA.h",
    ],
    deps = [
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Shape/IR",
        "//iree/compiler/Dialect/VMLA/Conversion",
        "//iree/compiler/Dialect/VMLA/IR",
        "//iree/compiler/Dialect/VMLA/IR:VMLADialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo:legalize_to_linalg",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo:legalize_to_standard",
    ],
)
