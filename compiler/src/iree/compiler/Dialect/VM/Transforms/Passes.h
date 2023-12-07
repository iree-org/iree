// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_VM_TRANSFORMS_PASSES_H_

#include <memory>

#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::IREE::VM {

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

// Converts from various dialects (standard, HAL, etc) to the VM dialect.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConversionPass(TargetOptions targetOptions);

//===----------------------------------------------------------------------===//
// Module layout
//===----------------------------------------------------------------------===//

// Hoists inline vm.rodata.inline values to module-level constant storage.
std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createHoistInlinedRodataPass();

// Deduplicates vm.rodata ops in the module.
std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createDeduplicateRodataPass();

// Resolves global loads of rodata ops to direct rodata references.
std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createResolveRodataLoadsPass();

//===----------------------------------------------------------------------===//
// Module analysis and ordinal assignment
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
// Optimization passes
//===----------------------------------------------------------------------===//

// Drops __init/__deinit functions that have no ops.
std::unique_ptr<OperationPass<IREE::VM::ModuleOp>>
createDropEmptyModuleInitializersPass();

// Sinks defining ops with few uses to their use-sites to reduce the total
// number of live registers at the cost of additional storage requirements.
std::unique_ptr<OperationPass<IREE::VM::ModuleOp>> createSinkDefiningOpsPass();

//===----------------------------------------------------------------------===//
// Test passes
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createConvertStandardToVMTestPass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

inline void registerVMPasses() {
  auto targetOptions = TargetOptions::FromFlags::get();
  registerVMTransformPassPipeline();
  createConversionPass(targetOptions);
  createHoistInlinedRodataPass();
  createDeduplicateRodataPass();
  createDropEmptyModuleInitializersPass();
  createGlobalInitializationPass();
  createOrdinalAllocationPass();
  createResolveRodataLoadsPass();
  createSinkDefiningOpsPass();
}

inline void registerVMTestPasses() {
  TargetOptions::FromFlags::get();
  createConvertStandardToVMTestPass();
}

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_TRANSFORMS_PASSES_H_
