// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_MODULES_HAL_INLINE_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_MODULES_HAL_INLINE_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineOps.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace Inline {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that run the required
// HALInline transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   <run conversion from TF/HLO/etc to flow>
//   buildHALInlineTransformPassPipeline & run
//   <serialize VM module>
void buildHALInlineStaticTransformPassPipeline(
    OpPassManager &passManager, const TargetBackendRegistry &targetRegistry,
    const TargetOptions &targetOptions);

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

// Inlines translated executable functions into the host program.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createInlineExecutablesPass();

// Converts from the stream dialect into the hal_inline dialect.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createConversionPass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerHALInlinePasses();

} // namespace Inline
} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_MODULES_HAL_INLINE_TRANSFORMS_PASSES_H_
