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

iree_tablegen_library(
  NAME
    IndexComputationAttrGen
  TD_FILE
    "IndexComputationAttr.td"
  OUTS
    -gen-struct-attr-decls IndexComputationAttr.h.inc
    -gen-struct-attr-defs IndexComputationAttr.cpp.inc
)

iree_cc_library(
  NAME
    IndexComputation
  HDRS
    "IREEIndexComputation.h"
    "IndexComputation.h"
    "IndexComputationAttr.h.inc"
    "IndexComputationAttribute.h"
    "IndexComputationPass.h"
    "XLAIndexPropagation.h"
  SRCS
    "IREEIndexComputation.cpp"
    "IndexComputation.cpp"
    "IndexComputationAttr.cpp.inc"
    "IndexComputationAttribute.cpp"
    "IndexComputationPass.cpp"
    "XLAIndexPropagation.cpp"
  DEPS
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Translation::SPIRV::IndexComputation::IndexComputationAttrGen
    iree::compiler::Utils
    LLVMSupport
    MLIRIR
    MLIRPass
    MLIRSPIRV
    MLIRStandardOps
    tensorflow::mlir_xla
  ALWAYSLINK
  PUBLIC
)
