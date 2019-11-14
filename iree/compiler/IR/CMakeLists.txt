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

add_subdirectory(Interpreter)
add_subdirectory(Sequencer)

iree_cc_library(
  NAME
    IR
  HDRS
    "ConfigOps.h"
    "Dialect.h"
    "Ops.h"
    "StructureOps.h"
    "Types.h"
  SRCS
    "ConfigOps.cpp"
    "ConfigOps.cpp.inc"
    "ConfigOps.h.inc"
    "Dialect.cpp"
    "Ops.cpp"
    "Ops.cpp.inc"
    "Ops.h.inc"
    "StructureOps.cpp"
    "StructureOps.cpp.inc"
    "StructureOps.h.inc"
    "Types.cpp"
  DEPS
    iree::compiler::IR::ConfigOpsGen
    iree::compiler::IR::EnumsGen
    iree::compiler::IR::OpsGen
    iree::compiler::IR::StructureOpsGen
    iree::schemas::executable_def_cc_fbs
    LLVMSupport
    MLIRIR
    MLIRStandardOps
  ALWAYSLINK
  PUBLIC
)

iree_tablegen_library(
  NAME
    ConfigOpsGen
  SRCS
    ConfigOps.td
  OUTS
    -gen-op-decls ConfigOps.h.inc
    -gen-op-defs ConfigOps.cpp.inc
)

iree_tablegen_library(
  NAME
    EnumsGen
  SRCS
    OpBase.td
  OUTS
    -gen-enum-decls Enums.h.inc
    -gen-enum-defs Enums.cpp.inc
)

iree_tablegen_library(
  NAME
    OpsGen
  SRCS
    Ops.td
  OUTS
    -gen-op-decls Ops.h.inc
    -gen-op-defs Ops.cpp.inc
)

iree_tablegen_library(
  NAME
    StructureOpsGen
  SRCS
    StructureOps.td
  OUTS
    -gen-op-decls StructureOps.h.inc
    -gen-op-defs StructureOps.cpp.inc
)
