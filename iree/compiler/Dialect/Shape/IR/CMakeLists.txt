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
    IR
  HDRS
    "Builders.h"
    "ShapeDialect.h"
    "ShapeInterface.h"
    "ShapeOps.h"
    "ShapeOps.h.inc"
    "ShapeTypes.h"
  SRCS
    "Builders.cpp"
    "ShapeDialect.cpp"
    "ShapeInterface.cpp"
    "ShapeOps.cpp"
    "ShapeOps.cpp.inc"
    "ShapeTypes.cpp"
  DEPS
    iree::compiler::Dialect::IREE::IR
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRStandardOps
    MLIRSupport
  ALWAYSLINK
  PUBLIC
)


iree_tablegen_library(
  NAME
    ShapeOpsGen
  TD_FILE
    ShapeOps.td
  OUTS
    -gen-op-decls ShapeOps.h.inc
    -gen-op-defs ShapeOps.cpp.inc
)
