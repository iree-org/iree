// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_INDEXING_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_IREE_INDEXING_TRANSFORMS_PASSES_H_

#include <optional>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::Indexing {

std::unique_ptr<OperationPass<void>> createTestIndexRangeAnalysisPass();

// Register all Passes
void registerIndexingPasses();

} // namespace mlir::iree_compiler::IREE::Indexing

#endif // IREE_COMPILER_DIALECT_IREE_INDEXING_TRANSFORMS_PASSES_H_
