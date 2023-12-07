// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_TARGET_C_TRANSLATIONFLAGS_H_
#define IREE_COMPILER_DIALECT_VM_TARGET_C_TRANSLATIONFLAGS_H_

#include "iree/compiler/Dialect/VM/Target/C/CModuleTarget.h"

namespace mlir::iree_compiler::IREE::VM {

// Returns a CTargetOptions struct initialized with the
// --iree-vm-c-* flags.
CTargetOptions getCTargetOptionsFromFlags();

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_TARGET_C_TRANSLATIONFLAGS_H_
