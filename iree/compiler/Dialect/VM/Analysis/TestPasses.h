// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_ANALYSIS_TESTPASSES_H_
#define IREE_COMPILER_DIALECT_VM_ANALYSIS_TESTPASSES_H_

#include <memory>

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

//===----------------------------------------------------------------------===//
// Test passes
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<IREE::VM::FuncOp>> createValueLivenessTestPass();

std::unique_ptr<OperationPass<IREE::VM::FuncOp>>
createRegisterAllocationTestPass();

//===----------------------------------------------------------------------===//
// Register all analysis passes
//===----------------------------------------------------------------------===//

inline void registerVMAnalysisTestPasses() {
  createValueLivenessTestPass();
  createRegisterAllocationTestPass();
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_ANALYSIS_TESTPASSES_H_