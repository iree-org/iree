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

iree_add_all_subdirs()

iree_cc_library(
  NAME
    Transforms
  HDRS
    "DestructiveUpdateUtils.h"
    "DispatchConfig.h"
    "Passes.h"
  SRCS
    "CreateBenchmarkFuncs.cpp"
    "DeduplicateExecutables.cpp"
    "DestructiveUpdateUtils.cpp"
    "DispatchConfig.cpp"
    "DispatchLinalgOnTensors.cpp"
    "DispatchabilityAnalysis.cpp"
    "ExpandVariableDynamicDims.cpp"
    "FlattenTuplesInCFG.cpp"
    "FoldCompatibleDispatchRegions.cpp"
    "FormStreams.cpp"
    "HLOToHLOPreprocessing.cpp"
    "HoistUnstreamableOps.cpp"
    "IdentifyDispatchRegions.cpp"
    "IdentifyDispatchRegions2.cpp"
    "InjectDispatchTracing.cpp"
    "LegalizeInputTypes.cpp"
    "MaterializeExportedReflection.cpp"
    "MergeExportedReflection.cpp"
    "OutlineDispatchRegions.cpp"
    "OutlineDispatchRegions2.cpp"
    "OutlineLargeConstants.cpp"
    "Passes.cpp"
    "PrePostPartitioningConversion.cpp"
    "RematerializeDispatchConstants.cpp"
    "StripAndSplatConstantVariables.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRLinalg
    MLIRLinalgTransforms
    MLIRPass
    MLIRSCF
    MLIRShape
    MLIRShapeOpsTransforms
    MLIRStandard
    MLIRSupport
    MLIRTensor
    MLIRTosaToLinalg
    MLIRTransformUtils
    MLIRTransforms
    iree::base::signature_mangle
    iree::compiler::Conversion::HLOToHLO
    iree::compiler::Conversion::HLOToLinalg
    iree::compiler::Conversion::HLOToLinalg::HLOToLinalgOnTensors
    iree::compiler::Dialect::Flow::Analysis
    iree::compiler::Dialect::Flow::Conversion
    iree::compiler::Dialect::Flow::Conversion::HLOToFlow
    iree::compiler::Dialect::Flow::Conversion::StandardToFlow
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::Flow::Utils
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::Shape::Conversion
    iree::compiler::Dialect::Shape::IR
    iree::compiler::Dialect::Shape::Transforms
    iree::compiler::Dialect::Shape::Utils::TypeConversion
    iree::compiler::Utils
    tensorflow::mlir_hlo
  PUBLIC
)
