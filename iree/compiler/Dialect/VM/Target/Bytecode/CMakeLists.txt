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
    Bytecode
  HDRS
    "BytecodeModuleTarget.h"
    "TranslationFlags.h"
  SRCS
    "BytecodeEncoder.cpp"
    "BytecodeEncoder.h"
    "BytecodeModuleTarget.cpp"
    "ConstantEncoder.cpp"
    "ConstantEncoder.h"
    "TranslationFlags.cpp"
    "TranslationRegistration.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRTranslation
    flatbuffers
    iree::compiler::Dialect::IREE::IR
    iree::compiler::Dialect::IREE::Transforms
    iree::compiler::Dialect::VM::Analysis
    iree::compiler::Dialect::VM::IR
    iree::compiler::Dialect::VM::Transforms
    iree::schemas::bytecode_module_def_cc_fbs
  PUBLIC
)
