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

cc_library(
    name = "tensorflow",
    srcs = select({
        "//iree:enable_tensorflow": [
            "TFSavedModelLowerExportedFunctions.cpp",
            "TFSavedModelLowerGlobalTensors.cpp",
            "Passes.cpp",
        ],
        "//conditions:default": [
        ],
    }),
    hdrs = [
        "Passes.h",
    ],
    defines = select({
        "//iree:enable_tensorflow": [
            "IREE_COMPILER_TENSORFLOW_ENABLED",
        ],
        "//conditions:default": [
        ],
    }),
    deps = select({
        "//iree:enable_tensorflow": [
            "@llvm-project//llvm:support",
            "@llvm-project//mlir:Support",
            "@llvm-project//mlir:TransformUtils",
            "@llvm-project//mlir:IR",
            "@org_tensorflow//tensorflow/compiler/mlir/tensorflow",
            "@org_tensorflow//tensorflow/compiler/mlir/tensorflow:tensorflow_dialect_registration",
            "//iree/compiler/Dialect/Flow/IR",
        ],
        "//conditions:default": [
        ],
    }) + [
        "//iree/base:signature_mangle",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "iree-tf-opt",
    deps = [
        ":tensorflow",
        "//iree/tools:iree_opt_library",
        "@llvm-project//mlir:MlirOptMain",
    ],
)

cc_binary(
    name = "iree-tf-translate",
    deps = [
        ":tensorflow",
        "//iree/tools:iree_translate_library",
    ],
)
