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

#ifndef IREE_COMPILER_DIALECT_VM_ANALYSIS_TESTPASSES_H_
#define IREE_COMPILER_DIALECT_VM_ANALYSIS_TESTPASSES_H_

#include <memory>

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/Module.h"
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