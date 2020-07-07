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
    Transforms
  HDRS
    "Passes.h"
    "Patterns.h"
  SRCS
    "CleanupPlaceholdersPass.cpp"
    "ConvertHLOToShapeDialectPass.cpp"
    "FunctionSignatureExpansionPass.cpp"
    "HoistShapeCalculationsPass.cpp"
    "MaterializeShapeCalculations.cpp"
    "MaterializeShapeCalculationsPass.cpp"
    "TieDynamicShapesPass.cpp"
  DEPS
    LLVMSupport
    MLIRAnalysis
    MLIRIR
    MLIRPass
    MLIRStandardOps
    MLIRSupport
    MLIRTransforms
    iree::compiler::Dialect::Shape::IR
    iree::compiler::Dialect::Shape::Plugins::VMLA::VMLAShapeBuilder
    iree::compiler::Dialect::Shape::Plugins::XLA::XlaHloShapeBuilder
    iree::compiler::Dialect::Shape::Utils::TypeConversion
    iree::compiler::Utils
    tensorflow::mlir_hlo
  PUBLIC
)
