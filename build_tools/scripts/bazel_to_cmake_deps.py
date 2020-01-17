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

# Bazel to CMake dependency conversions used by bazel_to_cmake.py.

ABSL_DEP_MAPPING = {
    "@com_google_absl//absl/base:core_headers": "absl::base",
    "@com_google_absl//absl/container:inlined_vector": "absl::inlined_vector",
    "@com_google_absl//absl/memory": "absl::memory",
    "@com_google_absl//absl/synchronization": "absl::synchronization",
    "@com_google_absl//absl/time": "absl::time",
    "@com_google_absl//absl/types:span": "absl::span",
}

LLVM_DEP_MAPPING = {
    "@llvm-project//llvm:support": "LLVMSupport",
}

MLIR_DEP_MAPPING = {
    "@llvm-project//mlir:IR": "MLIRIR",
    "@llvm-project//mlir:Pass": "MLIRPass",
    "@llvm-project//mlir:StandardOps": "MLIRStandardOps",
    "@llvm-project//mlir:Support": "MLIRSupport",
    "@llvm-project//mlir:TransformUtils": "MLIRTransformUtils",
    "@llvm-project//mlir:Transforms": "MLIRTransforms",
}


def convert_external_dep(dep):
  """Converts an external (doesn't start with //iree) Bazel dep to Cmake.

  IREE deps are expected to follow a standard form between Bazel and CMake that
  facilitates conversion. External dependencies *may* have their own patterns,
  or they may be purely special cases.

  Multiple dependencies in Bazel may map to a single dependency in CMake.
  A Bazel dependency may *not* map to multiple CMake dependencies.

  Returns:
    The converted dep if it was successfully converted.

  Raises:
    KeyError: No conversion was found for the dependency.
  """
  if dep.startswith("@com_google_absl"):
    return ABSL_DEP_MAPPING[dep]
  if dep.startswith("@llvm-project//llvm"):
    return LLVM_DEP_MAPPING[dep]
  if dep.startswith("@llvm-project//mlir"):
    return MLIR_DEP_MAPPING[dep]
  if dep.startswith("@org_tensorflow//tensorflow/compiler/mlir"):
    # All Bazel dependencies map to a single CMake dependency.
    return "tensorflow::mlir_xla"

  raise KeyError("No conversion found for dep '%s'" % dep)
