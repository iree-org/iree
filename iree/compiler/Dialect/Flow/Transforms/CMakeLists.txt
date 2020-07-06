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
    "DispatchConfig.h"
    "Passes.h"
  SRCS
    "DispatchConfig.cpp"
    "DispatchabilityAnalysis.cpp"
    "FlattenTuplesInCFG.cpp"
    "FoldCompatibleDispatchRegions.cpp"
    "FormStreams.cpp"
    "HLOToHLOPreprocessing.cpp"
    "HoistUnstreamableOps.cpp"
    "IdentifyDispatchRegions.cpp"
    "IdentifyDispatchRegions2.cpp"
    "LegalizeInputTypes.cpp"
    "MaterializeExportedReflection.cpp"
    "MergeExportedReflection.cpp"
    "OutlineDispatchRegions.cpp"
    "Passes.cpp"
    "PrePostPartitioningConversion.cpp"
    "RematerializeDispatchConstants.cpp"
  DEPS
    LLVMSupport
    MLIRAnalysis
    MLIRIR
    MLIRPass
    MLIRShapeOpsTransforms
    MLIRStandardOps
    MLIRSupport
    MLIRTransformUtils
    MLIRTransforms
    iree::base::signature_mangle
    iree::compiler::Dialect::Flow::Analysis
    iree::compiler::Dialect::Flow::Conversion
    iree::compiler::Dialect::Flow::Conversion::HLOToFlow
    iree::compiler::Dialect::Flow::Conversion::StandardToFlow
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::Flow::Utils
    iree::compiler::Dialect::Shape::Conversion
    iree::compiler::Dialect::Shape::IR
    iree::compiler::Dialect::Shape::Transforms
    iree::compiler::Dialect::Shape::Utils::TypeConversion
    iree::compiler::Utils
    tensorflow::mlir_hlo
  PUBLIC
)
