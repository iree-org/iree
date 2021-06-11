// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_TRANSLATIONFLAGS_H_
#define IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_TRANSLATIONFLAGS_H_

#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

// Returns a BytecodeTargetOptions struct initialized with the
// --iree-vm-bytecode-* flags.
BytecodeTargetOptions getBytecodeTargetOptionsFromFlags();

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_TARGET_BYTECODE_TRANSLATIONFLAGS_H_
