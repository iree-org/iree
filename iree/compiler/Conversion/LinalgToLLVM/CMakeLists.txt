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
    LinalgToLLVM
  HDRS
    "KernelDispatch.h"
    "Passes.h"
  SRCS
    "ConvImg2ColMatmulConversion.cpp"
    "ConvertToLLVM.cpp"
    "KernelDispatch.cpp"
    "LinalgTileAndDistributeOnTensorsPass.cpp"
    "LinalgTileAndDistributePass.cpp"
    "LinalgTileAndVectorizePass.cpp"
    "LinalgVectorizePass.cpp"
    "Passes.cpp"
    "PlanConvLoopOrder.cpp"
  DEPS
    LLVMSupport
    MLIRAffineToStandard
    MLIRIR
    MLIRLLVMIR
    MLIRLinalg
    MLIRLinalgToLLVM
    MLIRLinalgTransforms
    MLIRPass
    MLIRSCF
    MLIRSCFToStandard
    MLIRStandard
    MLIRStandardOpsTransforms
    MLIRStandardToLLVM
    MLIRStandardToSPIRV
    MLIRSupport
    MLIRTransforms
    MLIRVector
    MLIRVectorToLLVM
    MLIRVectorToSCF
    iree::compiler::Conversion::CodegenUtils
    iree::compiler::Conversion::Common
    iree::compiler::Conversion::HLOToHLO
    iree::compiler::Conversion::HLOToLinalg
    iree::compiler::Conversion::LLVMToLLVM
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::Flow::Transforms
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::HAL::IR::HALDialect
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::Shape::IR
    iree::compiler::Dialect::Shape::Transforms
  PUBLIC
)
