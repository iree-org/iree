# Lint as: python3
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

EXPLICIT_TARGET_MAPPING = {
    # absl
    "@com_google_absl//absl/flags:flag": ["absl::flags"],
    "@com_google_absl//absl/flags:parse": ["absl::flags_parse"],
    # dear_imgui
    "@dear_imgui": ["dear_imgui::dear_imgui"],
    "@dear_imgui//:imgui_sdl_vulkan": [
        "dear_imgui::impl_sdl", "dear_imgui::impl_vulkan"
    ],
    # MLIR
    "@llvm-project//mlir:AllPassesAndDialects": ["MLIRAllDialects"],
    "@llvm-project//mlir:AllPassesAndDialectsNoRegistration": [
        "MLIRAllDialects"
    ],
    "@llvm-project//mlir:Affine": ["MLIRAffineOps"],
    "@llvm-project//mlir:AffineToStandardTransforms": ["MLIRAffineToStandard"],
    "@llvm-project//mlir:CFGTransforms": ["MLIRSCFToStandard"],
    "@llvm-project//mlir:ExecutionEngineUtils": ["MLIRExecutionEngine"],
    "@llvm-project//mlir:GPUDialect": ["MLIRGPU"],
    "@llvm-project//mlir:GPUTransforms": ["MLIRGPU"],
    "@llvm-project//mlir:LLVMDialect": ["MLIRLLVMIR"],
    "@llvm-project//mlir:LLVMTransforms": ["MLIRStandardToLLVM"],
    "@llvm-project//mlir:SCFToGPUPass": ["MLIRSCFToGPU"],
    "@llvm-project//mlir:SCFDialect": ["MLIRSCF"],
    "@llvm-project//mlir:ShapeTransforms": ["MLIRShapeOpsTransforms"],
    "@llvm-project//mlir:SideEffects": ["MLIRSideEffectInterfaces"],
    "@llvm-project//mlir:SPIRVDialect": ["MLIRSPIRV"],
    "@llvm-project//mlir:SPIRVLowering": ["MLIRSPIRV", "MLIRSPIRVTransforms"],
    "@llvm-project//mlir:SPIRVTranslateRegistration": [
        "MLIRSPIRVSerialization"
    ],
    "@llvm-project//mlir:StandardToSPIRVConversions": [
        "MLIRStandardToSPIRVTransforms"
    ],
    "@llvm-project//mlir:mlir_c_runner_utils": ["MLIRExecutionEngine"],
    "@llvm-project//mlir:mlir-translate": ["mlir-translate"],
    "@llvm-project//mlir:MlirTableGenMain": ["MLIRTableGen"],
    "@llvm-project//mlir:MlirOptLib": ["MLIROptLib"],
    "@llvm-project//mlir:VectorOps": ["MLIRVector"],
    # Vulkan
    # TODO(scotttodd): Set -DVK_NO_PROTOTYPES to COPTS for _no_prototypes.
    #   Maybe add a wrapper CMake lib within build_tools/third_party/?
    "@iree_vulkan_headers//:vulkan_headers": ["Vulkan::Headers"],
    "@iree_vulkan_headers//:vulkan_headers_no_prototypes": ["Vulkan::Headers"],
    # The Bazel target maps to the IMPORTED target defined by FindVulkan().
    "@vulkan_sdk//:sdk": ["Vulkan::Vulkan"],
    # Misc single targets
    "@com_google_benchmark//:benchmark": ["benchmark"],
    "@com_github_google_flatbuffers//:flatbuffers": ["flatbuffers"],
    "@com_github_dvidelabs_flatcc//:flatcc": ["flatcc"],
    "@com_github_dvidelabs_flatcc//:runtime": ["flatcc::runtime"],
    "@com_google_googletest//:gtest": ["gmock", "gtest"],
    "@renderdoc_api//:renderdoc_app": ["renderdoc_api::renderdoc_app"],
    "@sdl2//:SDL2": ["SDL2-static"]
}


def _convert_absl_target(target):
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


def _convert_mlir_target(target):
  # Default to a pattern substitution approach.
  # Take "MLIR" and append the name part of the full target identifier, e.g.
  #   "@llvm-project//mlir:IR"   -> "MLIRIR"
  #   "@llvm-project//mlir:Pass" -> "MLIRPass"
  return ["MLIR" + target.rsplit(":")[-1]]


def _convert_llvm_target(target):
  # Default to a pattern substitution approach.
  # Prepend "LLVM" to the Bazel target name.
  #   "@llvm-project//llvm:AsmParser" -> "LLVMAsmParser"
  #   "@llvm-project//llvm:Core" -> "LLVMCore"
  return ["LLVM" + target.rsplit(":")[-1]]


def convert_external_target(target):
  """Converts an external (non-IREE) Bazel target to a list of CMake targets.

  IREE targets are expected to follow a standard form between Bazel and CMake
  that facilitates conversion. External targets *may* have their own patterns,
  or they may be purely special cases.

  Multiple target in Bazel may map to a single target in CMake and a Bazel
  target may map to multiple CMake targets.

  Returns:
    A list of converted targets if it was successfully converted.

  Raises:
    KeyError: No conversion was found for the target.
  """
  if target in EXPLICIT_TARGET_MAPPING:
    return EXPLICIT_TARGET_MAPPING[target]
  if target.startswith("@com_google_absl//absl"):
    return _convert_absl_target(target)
  if target.startswith("@llvm-project//llvm"):
    return _convert_llvm_target(target)
  if target.startswith("@llvm-project//mlir"):
    return _convert_mlir_target(target)
  if target.startswith("@org_tensorflow//tensorflow/compiler/mlir"):
    # All Bazel targets map to a single CMake target.
    return ["tensorflow::mlir_hlo"]
  if target.startswith("@com_google_ruy//ruy"):
    # All Bazel targets map to a single CMake target.
    return ["ruy"]

  raise KeyError("No conversion found for target '%s'" % target)
