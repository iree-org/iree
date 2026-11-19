// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_SHARDY_INPUTCONVERSION_PASSES_H_
#define IREE_COMPILER_PLUGINS_INPUT_SHARDY_INPUTCONVERSION_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::shardy {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// Build the Shardy input conversion pipeline.
// For single-device execution, this strips sdy dialect ops and attributes.
void buildShardyInputConversionPassPipeline(OpPassManager &passManager);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

// Register all Shardy input conversion passes.
void registerShardyInputConversionPasses();

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

// Create pass to strip sdy dialect ops and attributes for single-device
// execution. Sdy ops are metadata-only sharding annotations that can be
// safely removed when targeting a single device.
std::unique_ptr<Pass> createStripShardyDialectPass();

} // namespace mlir::iree_compiler::shardy

#endif // IREE_COMPILER_PLUGINS_INPUT_SHARDY_INPUTCONVERSION_PASSES_H_
