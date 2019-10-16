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

iree_cc_library(
  NAME
    Sequencer
  HDRS
    "Passes.h"
    "Rewrites.h"
  SRCS
    "AssignExecutableOrdinals.cpp"
    "AssignExecutableWorkloadAttrs.cpp"
    "FoldCompatibleDispatchRegions.cpp"
    "IdentifyDispatchRegions.cpp"
    "IdentifyReductionRegions.cpp"
    "LegalizeInputs.cpp"
    "LowerSequencerDialect.cpp"
    "LowerStdToSequencerDialect.cpp"
    "LowerToSequencerDialect.cpp"
    "LowerXLAToSequencerDialect.cpp"
    "OutlineDispatchRegions.cpp"
    "OutlineReductionRegions.cpp"
    "RematerializeDispatchConstants.cpp"
  DEPS
    iree::compiler::IR
    iree::compiler::IR::Sequencer
    iree::compiler::Utils
    tensorflow::mlir_xla
    LLVMSupport
    MLIRIR
    MLIRPass
    MLIRTransformUtils
  PUBLIC
)
