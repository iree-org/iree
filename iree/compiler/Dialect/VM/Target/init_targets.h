// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_TARGET_INIT_TARGETS_H_
#define IREE_COMPILER_DIALECT_VM_TARGET_INIT_TARGETS_H_

namespace mlir {
namespace iree_compiler {

namespace IREE {
namespace VM {
void registerToVMBytecodeTranslation();
void registerToCTranslation();
}  // namespace VM
}  // namespace IREE

// This function should be called before creating any MLIRContext if one
// expects all the possible target backends to be available. Custom tools can
// select which targets they want to support by only registering those they
// need.
inline void registerVMTargets() {
  static bool init_once = []() {
    IREE::VM::registerToVMBytecodeTranslation();
    IREE::VM::registerToCTranslation();
    return true;
  }();
  (void)init_once;
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_TARGET_INIT_TARGETS_H_
