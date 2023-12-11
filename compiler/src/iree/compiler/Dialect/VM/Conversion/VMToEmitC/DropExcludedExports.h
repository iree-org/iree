// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_DROPEXCLUDEDEXPORTS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_DROPEXCLUDEDEXPORTS_H_

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::VM {

// TODO(marbre): Switch pass registration to tablegen.
std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createDropExcludedExportsPass();

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_DROPEXCLUDEDEXPORTS_H_
