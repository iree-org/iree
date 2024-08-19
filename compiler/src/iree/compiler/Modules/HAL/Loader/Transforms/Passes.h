// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_MODULES_HAL_LOADER_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_MODULES_HAL_LOADER_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetOptions.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderOps.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::HAL::Loader {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that run the required
// HALLoader transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   <run conversion from TF/HLO/etc -> flow -> stream>
//   buildHALInlineDynamicTransformPassPipeline & run
//   <serialize VM module>
void buildHALInlineDynamicTransformPassPipeline(
    OpPassManager &passManager, const TargetRegistry &targetRegistry,
    const TargetOptions &targetOptions);

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "iree/compiler/Modules/HAL/Loader/Transforms/Passes.h.inc"

void registerHALLoaderPasses();

} // namespace mlir::iree_compiler::IREE::HAL::Loader

#endif // IREE_COMPILER_MODULES_HAL_LOADER_TRANSFORMS_PASSES_H_
