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

# TODO(scotttodd): Trim using a pattern absl::[name], with special cases
ABSL_TARGET_MAPPING = {
    "@com_google_absl//absl/base:core_headers": "absl::base",
    "@com_google_absl//absl/container:inlined_vector": "absl::inlined_vector",
    "@com_google_absl//absl/memory": "absl::memory",
    "@com_google_absl//absl/strings": "absl::strings",
    "@com_google_absl//absl/synchronization": "absl::synchronization",
    "@com_google_absl//absl/time": "absl::time",
    "@com_google_absl//absl/types:optional": "absl::optional",
    "@com_google_absl//absl/types:span": "absl::span",
    "@com_google_absl//absl/types:variant": "absl::variant",
}

LLVM_TARGET_MAPPING = {
    "@llvm-project//llvm:support": "LLVMSupport",
}

# TODO(scotttodd): Trim using a pattern MLIR[Name], as long as all match
MLIR_TARGET_MAPPING = {
    "@llvm-project//mlir:IR": "MLIRIR",
    "@llvm-project//mlir:MlirOptMain": "MLIROptMain",
    "@llvm-project//mlir:Parser": "MLIRParser",
    "@llvm-project//mlir:Pass": "MLIRPass",
    "@llvm-project//mlir:StandardOps": "MLIRStandardOps",
    "@llvm-project//mlir:Support": "MLIRSupport",
    "@llvm-project//mlir:TransformUtils": "MLIRTransformUtils",
    "@llvm-project//mlir:Transforms": "MLIRTransforms",
}


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
    return ABSL_TARGET_MAPPING[target]
  if target.startswith("@llvm-project//llvm"):
    return LLVM_TARGET_MAPPING[target]
  if target.startswith("@llvm-project//mlir"):
    return MLIR_TARGET_MAPPING[target]
  if target.startswith("@org_tensorflow//tensorflow/compiler/mlir"):
    # All Bazel targets map to a single CMake target.
    return "tensorflow::mlir_xla"

  raise KeyError("No conversion found for target '%s'" % target)
