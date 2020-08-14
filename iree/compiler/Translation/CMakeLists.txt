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

# bazel_to_cmake: DO NOT EDIT (EmitC is only buildable with CMake)

iree_add_all_subdirs()

if(IREE_ENABLE_EMITC)
  set(IREE_VM_CONDITIONAL_TARGETS
    iree::compiler::Dialect::VM::Target::C
  )
endif()

iree_cc_library(
  NAME
    IREEVM
  HDRS
    "IREEVM.h"
  SRCS
    "IREEVM.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTranslation
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::Flow::Transforms
    iree::compiler::Dialect::HAL::Conversion::HALToVM
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Dialect::HAL::Transforms
    iree::compiler::Dialect::IREE::Transforms
    iree::compiler::Dialect::VM::Conversion
    iree::compiler::Dialect::VM::Conversion::StandardToVM
    iree::compiler::Dialect::VM::Target::Bytecode
    iree::compiler::Dialect::VM::Transforms
    ${IREE_VM_CONDITIONAL_TARGETS}
  PUBLIC
)
