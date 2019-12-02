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

# Transforms specific to the IREE sequencer.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "Sequencer",
    srcs = [
        "AssignExecutableOrdinals.cpp",
        "AssignExecutableWorkloadAttrs.cpp",
        "FoldCompatibleDispatchRegions.cpp",
        "IdentifyDispatchRegions.cpp",
        "IdentifyReductionRegions.cpp",
        "LegalizeInputs.cpp",
        "LowerSequencerDialect.cpp",
        "LowerStdToSequencerDialect.cpp",
        "LowerToSequencerDialect.cpp",
        "LowerXLAToSequencerDialect.cpp",
        "OutlineDispatchRegions.cpp",
        "OutlineReductionRegions.cpp",
        "RematerializeDispatchConstants.cpp",
    ],
    hdrs = [
        "Passes.h",
        "Rewrites.h",
    ],
    deps = [
        "//iree/compiler/IR",
        "//iree/compiler/IR/Sequencer",
        "//iree/compiler/Transforms",
        "//iree/compiler/Utils",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardDialectRegistration",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TransformUtils",
        "@local_config_mlir//:Transforms",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:hlo",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_legalize_to_standard",
        "@org_tensorflow//tensorflow/compiler/mlir/xla:xla_lower",
    ],
    # To pick up registered passes.
    alwayslink = 1,
)
