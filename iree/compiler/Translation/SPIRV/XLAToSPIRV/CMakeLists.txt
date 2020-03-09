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

add_subdirectory(test)

iree_cc_library(
  NAME
    XLAToSPIRV
  HDRS
    "IREEToSPIRV.h"
    "IREEToSPIRVPass.h"
    "SPIRVLowering.h"
    "TensorIndexToScalarValueMap.h"
    "XLAToSPIRV.h"
  SRCS
    "IREEToSPIRV.cpp"
    "IREEToSPIRVPass.cpp"
    "SPIRVLowering.cpp"
    "XLAToSPIRV.cpp"
  DEPS
    LLVMSupport
    MLIRAffineOps
    MLIRAffineToStandard
    MLIRIR
    MLIRPass
    MLIRSPIRV
    MLIRSPIRVTransforms
    MLIRStandardOps
    MLIRStandardToSPIRVTransforms
    MLIRSupport
    MLIRTransforms
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Translation::CodegenUtils
    iree::compiler::Translation::SPIRV::IndexComputation
    iree::compiler::Translation::SPIRV::Passes
    tensorflow::mlir_xla
  ALWAYSLINK
  PUBLIC
)
