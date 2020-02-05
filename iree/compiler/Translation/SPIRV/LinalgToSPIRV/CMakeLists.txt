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
    LinalgToSPIRV
  HDRS
    "LowerToSPIRV.h"
  SRCS
    "LowerToSPIRV.cpp"
  DEPS
    iree::compiler::Translation::XLAToLinalg::IREELinalgTensorToBuffer
    iree::compiler::Utils
    LLVMSupport
    MLIRAffineOps
    MLIRAffineToStandard
    MLIREDSC
    MLIRGPUtoSPIRVTransforms
    MLIRGPU
    MLIRIR
    MLIRLinalgOps
    MLIRLinalgTransforms
    MLIRLoopOps
    MLIRLoopsToGPU
    MLIRPass
    MLIRSPIRV
    MLIRStandardOps
    MLIRStandardToSPIRVTransforms
    MLIRSupport
    MLIRTransforms
    tensorflow::mlir_xla
  ALWAYSLINK
  PUBLIC
)
