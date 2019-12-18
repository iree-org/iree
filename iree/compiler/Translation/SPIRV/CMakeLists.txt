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

add_subdirectory(Kernels)

iree_cc_library(
  NAME
    SPIRV
  HDRS
    "EmbeddedKernels.h"
    "IndexComputation.h"
    "IndexComputationAttribute.h"
    "IREECodegenUtils.h"
    "IREEIndexComputation.h"
    "IREEToSPIRV.h"
    "IREEToSPIRVPass.h"
    "SPIRVExecutableTranslation.h"
    "SPIRVLowering.h"
    "TensorIndexToScalarValueMap.h"
    "XLAIndexPropagation.h"
    "XLAToSPIRV.h"
  SRCS
    "AdjustIntegerWidthPass.cpp"
    "EmbeddedKernels.cpp"
    "IndexComputation.cpp"
    "IndexComputationAttribute.cpp"
    "IndexComputationPass.cpp"
    "IREEIndexComputation.cpp"
    "IREEToSPIRV.cpp"
    "IREEToSPIRVPass.cpp"
    "PrepareReductionDispatch.cpp"
    "ReductionFnLowering.cpp"
    "SPIRVExecutableTranslation.cpp"
    "SPIRVLowering.cpp"
    "XLAIndexPropagation.cpp"
    "XLAToSPIRV.cpp"
  DEPS
    flatbuffers
    iree::compiler::IR
    iree::compiler::IR::Interpreter
    iree::compiler::Serialization
    iree::compiler::Transforms
    iree::compiler::Transforms::Sequencer
    # TODO(marbre): Use instead of Kernels target, as soon as build with iree_glsl_cc_library
    #iree::compiler::Translation::SPIRV::Kernels
    iree_compiler_Translation_SPIRV_Kernels
    iree::compiler::Utils
    iree::schemas
    iree::schemas::spirv_executable_def_cc_fbs
    tensorflow::mlir_xla
    LLVMSupport
    MLIRIR
    MLIRSPIRV
    MLIRSPIRVSerialization
    MLIRTransforms
    MLIRTranslation
  ALWAYSLINK
  PUBLIC
)
