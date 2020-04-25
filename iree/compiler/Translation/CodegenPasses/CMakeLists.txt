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

# bazel_to_cmake: DO NOT EDIT
# TODO: The follwing tablegen_library doesn't convert with bazel_to_cmake
# see https://github.com/google/iree/issues/1083

iree_add_all_subdirs()

iree_tablegen_library(
  NAME
    LinalgVectorTransformPatternsIncGen
  TD_FILE
    "LinalgVectorTransformPatterns.td"
  OUTS
    -gen-rewriters LinalgVectorTransformPatterns.h.inc
)

#TODO(GH-1673): Remove this once we can properly model dependencies between
# TableGen targets.
add_dependencies(iree_compiler_Translation_CodegenPasses_LinalgVectorTransformPatternsIncGen_target MLIRLinalgTransformPatternsIncGen)

iree_cc_library(
  NAME
    CodegenPasses
  HDRS
    "Passes.h"
    "LinalgVectorTransformPatterns.h.inc"
  SRCS
    "HALInterfaceToMemrefPass.cpp"
    "HLOToLinalgOnBuffers.cpp"
    "HLOToLinalgOnTensors.cpp"
    "LinalgFusion.cpp"
    "LinalgVectorTransform.cpp"
    "Passes.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRLinalgOps
    MLIRLinalgTransforms
    MLIRPass
    MLIRSPIRV
    MLIRStandardOps
    MLIRSupport
    MLIRTransforms
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Translation::CodegenUtils
    tensorflow::mlir_xla
  PUBLIC
)
