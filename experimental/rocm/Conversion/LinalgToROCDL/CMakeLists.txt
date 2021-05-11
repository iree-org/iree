# Copyright 2021 Nod Labs
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
    LinalgToROCDL
  HDRS
    "KernelConfig.h"
    "Passes.h"
  SRCS
    "ConvertToROCDL.cpp"
    "KernelConfig.cpp"
    "Passes.cpp"
    "RemoveTrivialLoops.cpp"
    "TileAndDistribute.cpp"
    "VectorizationPass.cpp"
  DEPS
    MLIRAffine
    MLIRGPU
    MLIRGPUToROCDLTransforms
    MLIRIR
    MLIRLinalg
    MLIRLinalgTransforms
    MLIRMemRefTransforms
    MLIRROCDLIR
    MLIRPass
    MLIRSCFToStandard
    MLIRStandard
    MLIRStandardToLLVM
    MLIRStandardToSPIRV
    MLIRTransforms
    MLIRVector
    MLIRVectorToLLVM
    MLIRVectorToSCF
    iree::compiler::Conversion::CodegenUtils
    iree::compiler::Conversion::Common
    iree::compiler::Conversion::HLOToHLO
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::Shape::Transforms
    tensorflow::mlir_hlo
  PUBLIC
)
