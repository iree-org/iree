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
    Transforms
  HDRS
    "ConversionUtils.h"
    "Passes.h"
    "Rewrites.h"
  SRCS
    "AggressiveOpElimination.cpp"
    "AssignFunctionOrdinals.cpp"
    "ConvertFromTupleCallingConvention.cpp"
    "ConvertToMemRefCallingConvention.cpp"
    "DropUnreachableFunctions.cpp"
    "ExpandReductionsToOps.cpp"
    "LegalizeTypeStorage.cpp"
    "LowerInterpreterDialect.cpp"
    "LowerStdToInterpreterDialect.cpp"
    "LowerStdToIreeDialect.cpp"
    "LowerToInterpreterDialect.cpp"
    "LowerXLAToInterpreterDialect.cpp"
    "LowerXLAToIreeDialect.cpp"
    "MakeExecutableABI.cpp"
  DEPS
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Translation::Interpreter::IR
    iree::compiler::Translation::Interpreter::IR::Common
    iree::compiler::Translation::Interpreter::Serialization
    iree::compiler::Translation::Interpreter::Utils
    iree::compiler::Utils
    iree::schemas::bytecode::interpreter_bytecode_v0
    LLVMSupport
    MLIRAnalysis
    MLIRIR
    MLIRPass
    MLIRStandardOps
    MLIRSupport
    MLIRTransformUtils
    MLIRTransforms
    tensorflow::mlir_xla
  ALWAYSLINK
  PUBLIC
)
