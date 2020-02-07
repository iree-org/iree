#!/usr/bin/env python3
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

# Bazel to CMake target name conversions used by bazel_to_cmake.py.

ABSL_EXPLICIT_TARGET_MAPPING = {
    "@com_google_absl//absl/flags:flag": ["absl::flags"],
    "@com_google_absl//absl/flags:parse": ["absl::flags_parse"],
}


def _convert_absl_target(target):
  if target in ABSL_EXPLICIT_TARGET_MAPPING:
    return ABSL_EXPLICIT_TARGET_MAPPING[target]

  # Default to a pattern substitution approach.
  # Take "absl::" and append the name part of the full target identifier, e.g.
  #   "@com_google_absl//absl/memory"         -> "absl::memory"
  #   "@com_google_absl//absl/types:optional" -> "absl::optional"
  #   "@com_google_absl//absl/types:span"     -> "absl::span"
  if ":" in target:
    target_name = target.rsplit(":")[-1]
  else:
    target_name = target.rsplit("/")[-1]
  return ["absl::" + target_name]


DEAR_IMGUI_EXPLICIT_TARGET_MAPPING = {
    "@dear_imgui": ["dear_imgui::dear_imgui"],
    "@dear_imgui//:imgui_sdl_vulkan": [
        "dear_imgui::impl_sdl", "dear_imgui::impl_vulkan"
    ],
}

RENDERDOC_API_MAPPING = {
    "@renderdoc_api//:renderdoc_app": ["renderdoc_api::renderdoc_app"]
}

LLVM_TARGET_MAPPING = {
    "@llvm-project//llvm:core": ["LLVMCore"],
    "@llvm-project//llvm:support": ["LLVMSupport"],
    "@llvm-project//llvm:tablegen": ["LLVMTableGen"],
}

VULKAN_HEADERS_MAPPING = {
    # TODO(scotttodd): Set -DVK_NO_PROTOTYPES to COPTS for _no_prototypes.
    #   Maybe add a wrapper CMake lib within build_tools/third_party/?
    "@vulkan_headers//:vulkan_headers": ["Vulkan::Headers"],
    "@vulkan_headers//:vulkan_headers_no_prototypes": ["Vulkan::Headers"],
}

MLIR_EXPLICIT_TARGET_MAPPING = {
    "@llvm-project//mlir:AffineDialectRegistration": ["MLIRAffineOps"],
    "@llvm-project//mlir:AffineToStandardTransforms": ["MLIRAffineToStandard"],
    "@llvm-project//mlir:CFGTransforms": ["MLIRLoopToStandard"],
    "@llvm-project//mlir:GPUToSPIRVTransforms": ["MLIRGPUtoSPIRVTransforms"],
    "@llvm-project//mlir:GPUTransforms": ["MLIRGPU"],
    "@llvm-project//mlir:LinalgDialectRegistration": ["MLIRLinalgOps"],
    "@llvm-project//mlir:LLVMTransforms": ["MLIRStandardToLLVM"],
    "@llvm-project//mlir:LoopsToGPUPass": ["MLIRLoopsToGPU"],
    "@llvm-project//mlir:SPIRVDialect": ["MLIRSPIRV"],
    "@llvm-project//mlir:SPIRVDialectRegistration": ["MLIRSPIRV"],
    "@llvm-project//mlir:SPIRVLowering": ["MLIRSPIRV"],
    "@llvm-project//mlir:SPIRVTranslateRegistration": [
        "MLIRSPIRVSerialization"
    ],
    "@llvm-project//mlir:StandardDialectRegistration": ["MLIRStandardOps"],
    "@llvm-project//mlir:StandardToSPIRVConversions": [
        "MLIRStandardToSPIRVTransforms"
    ],
    "@llvm-project//mlir:TableGen": ["LLVMMLIRTableGen"],
    "@llvm-project//mlir:mlir-translate": ["mlir-translate"],
    "@llvm-project//mlir:MlirTableGenMain": ["MLIRTableGen"],
    "@llvm-project//mlir:MlirOptMain": ["MLIROptMain"],
}


def _convert_mlir_target(target):
  if target in MLIR_EXPLICIT_TARGET_MAPPING:
    return MLIR_EXPLICIT_TARGET_MAPPING[target]

  # Default to a pattern substitution approach.
  # Take "MLIR" and append the name part of the full target identifier, e.g.
  #   "@llvm-project//mlir:IR"   -> "MLIRIR"
  #   "@llvm-project//mlir:Pass" -> "MLIRPass"
  return ["MLIR" + target.rsplit(":")[-1]]


def convert_external_target(target):
  """Converts an external (doesn't start with //iree) Bazel target to Cmake.

  IREE targets are expected to follow a standard form between Bazel and CMake
  that facilitates conversion. External targets *may* have their own patterns,
  or they may be purely special cases.

  Multiple target in Bazel may map to a single target in CMake.
  A Bazel target may *not* map to multiple CMake targets.

  Returns:
    The converted target if it was successfully converted.

  Raises:
    KeyError: No conversion was found for the target.
  """
  if target.startswith("@com_google_absl"):
    return _convert_absl_target(target)
  if target == "@com_google_benchmark//:benchmark":
    return ["benchmark"]
  if target == "@com_github_google_flatbuffers//:flatbuffers":
    return ["flatbuffers"]
  if target.startswith("@dear_imgui"):
    return DEAR_IMGUI_EXPLICIT_TARGET_MAPPING[target]
  if target.startswith("@renderdoc_api"):
    return RENDERDOC_API_MAPPING[target]
  if target == "@com_google_googletest//:gtest":
    return ["gtest"]
  if target.startswith("@llvm-project//llvm"):
    return LLVM_TARGET_MAPPING[target]
  if target.startswith("@llvm-project//mlir"):
    return _convert_mlir_target(target)
  if target.startswith("@org_tensorflow//tensorflow/compiler/mlir"):
    # All Bazel targets map to a single CMake target.
    return ["tensorflow::mlir_xla"]
  if target.startswith("@org_tensorflow//tensorflow/lite/experimental/ruy"):
    # All Bazel targets map to a single CMake target.
    return ["ruy"]
  if target == "@sdl2//:SDL2":
    return ["SDL2-static"]
  if target.startswith("@vulkan_headers"):
    return VULKAN_HEADERS_MAPPING[target]
  if target == "@vulkan_sdk//:sdk":
    # The Bazel target maps to the IMPORTED target defined by FindVulkan().
    return ["Vulkan::Vulkan"]

  raise KeyError("No conversion found for target '%s'" % target)
