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

iree_add_all_subdirs()

iree_cc_library(
  NAME
    HLOToLinalgOnTensors
  HDRS
    "HLOToLinalgOnTensorPasses.h"
  SRCS
    "FusionOfTensorOps.cpp"
    "HLOToLinalgOnTensors.cpp"
  DEPS
    MLIRAffine
    MLIRIR
    MLIRLinalg
    MLIRLinalgTransforms
    MLIRPass
    MLIRStandard
    MLIRSupport
    MLIRTensor
    MLIRTransforms
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::HAL::IR::HALDialect
    iree::compiler::Dialect::Shape::IR
    tensorflow::mlir_hlo
  PUBLIC
)

iree_cc_library(
  NAME
    HLOToLinalg
  HDRS
    "Passes.h"
  SRCS
    "HLOToLinalgOnBuffers.cpp"
    "Passes.cpp"
    "ResolveShapeOps.cpp"
  DEPS
    ::HLOToLinalgOnTensors
    LLVMSupport
    MLIRAffine
    MLIRIR
    MLIRLinalg
    MLIRLinalgTransforms
    MLIRPass
    MLIRSPIRV
    MLIRStandard
    MLIRSupport
    MLIRTensor
    MLIRTransforms
    iree::compiler::Conversion::CodegenUtils
    iree::compiler::Conversion::HLOToHLO
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::HAL::IR::HALDialect
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::Shape::IR
    tensorflow::mlir_hlo
  PUBLIC
)
