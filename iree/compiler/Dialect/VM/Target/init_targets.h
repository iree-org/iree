// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_COMPILER_DIALECT_VM_TARGET_INIT_TARGETS_H_
#define IREE_COMPILER_DIALECT_VM_TARGET_INIT_TARGETS_H_

namespace mlir {
namespace iree_compiler {

namespace IREE {
namespace VM {
void registerToVMBytecodeTranslation();
#ifdef IREE_HAVE_EMITC_DIALECT
void registerToCTranslation();
#endif  // IREE_HAVE_EMITC_DIALECT
}  // namespace VM
}  // namespace IREE

// This function should be called before creating any MLIRContext if one
// expects all the possible target backends to be available. Custom tools can
// select which targets they want to support by only registering those they
// need.
inline void registerVMTargets() {
  static bool init_once = []() {
    IREE::VM::registerToVMBytecodeTranslation();
#ifdef IREE_HAVE_EMITC_DIALECT
    IREE::VM::registerToCTranslation();
#endif  // IREE_HAVE_EMITC_DIALECT

    return true;
  }();
  (void)init_once;
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_TARGET_INIT_TARGETS_H_
