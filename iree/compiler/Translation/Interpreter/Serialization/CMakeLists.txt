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
    Serialization
  HDRS
    "BytecodeTables.h"
    "BytecodeWriter.h"
    "VMFunctionBuilder.h"
    "VMFunctionTableBuilder.h"
    "VMModuleBuilder.h"
  SRCS
    "BytecodeTables.cpp"
    "BytecodeWriter.cpp"
    "VMFunctionBuilder.cpp"
    "VMFunctionTableBuilder.cpp"
    "VMModuleBuilder.cpp"
  DEPS
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Translation::Interpreter::Utils
    iree::schemas::interpreter_module_def_cc_fbs
    iree::schemas::bytecode::interpreter_bytecode_v0
    flatbuffers
    LLVMSupport
    MLIRIR
    MLIRStandardOps
    MLIRSupport
  PUBLIC
)
