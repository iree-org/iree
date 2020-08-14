// Copyright 2019 Google LLC
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

#ifndef IREE_COMPILER_DIALECT_VM_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_VM_TRANSFORMS_PASSES_H_

#include <memory>

#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that run the required VM
// transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   <run conversion to HAL/etc>
//   buildVMTransformPassPipeline & run
//   <run target serialization/etc>
void buildVMTransformPassPipeline(OpPassManager &passManager,
                                  TargetOptions targetOptions);

void registerVMTransformPassPipeline();

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

// Marks all symbols with public visibility as being exported with the
// `iree.module.export` attribute. This is only required until we fully support
// symbol visibility.
// TODO(#614): remove this when iree.module.export is gone.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createMarkPublicSymbolsExportedPass();

// Converts from various dialects (standard, HAL, etc) to the VM dialect.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createConversionPass(
    TargetOptions targetOptions);

//===----------------------------------------------------------------------===//
// Module Analysis and Assignment
//===----------------------------------------------------------------------===//

// Gathers all module-level global init/deinit functions into single locations
// such that the runtime can init/deinit everything at once.
std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createGlobalInitializationPass();

// Assigns module-unique ordinals to function/global/etc symbols within the
// module.
std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createOrdinalAllocationPass();

//===----------------------------------------------------------------------===//
// Test passes
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConvertStandardToVMTestPass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

inline void registerVMPasses() {
  auto targetOptions = getTargetOptionsFromFlags();
  registerVMTransformPassPipeline();
  createConversionPass(targetOptions);
  createGlobalInitializationPass();
  createOrdinalAllocationPass();
}

inline void registerVMTestPasses() {
  getTargetOptionsFromFlags();
  createConvertStandardToVMTestPass();
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_TRANSFORMS_PASSES_H_
