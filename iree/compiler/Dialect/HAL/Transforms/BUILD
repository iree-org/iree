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
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Transforms",
    srcs = [
        "InlineDeviceSwitches.cpp",
        "LinkExecutables.cpp",
        "MaterializeInterfaces.cpp",
        "MaterializeResourceCaches.cpp",
        "MemoizeDeviceQueries.cpp",
        "Passes.cpp",
        "PublicAbiGeneration.cpp",
        "SerializeExecutables.cpp",
        "TranslateExecutables.cpp",
    ],
    hdrs = [
        "Passes.h",
    ],
    deps = [
        "//iree/base:signature_mangle",
        "//iree/compiler/Dialect/Flow/IR",
        "//iree/compiler/Dialect/HAL/Conversion/FlowToHAL",
        "//iree/compiler/Dialect/HAL/IR",
        "//iree/compiler/Dialect/HAL/Target",
        "//iree/compiler/Dialect/IREE/IR",
        "//iree/compiler/Dialect/Shape/Transforms",
        "//iree/compiler/Utils",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/hlo",
    ],
)
