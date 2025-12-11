// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_TRANSFORMS_TRANSFORMS_H_

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Codegen {

//===----------------------------------------------------------------------===//
// Populate functions.
//===----------------------------------------------------------------------===//

/// Populate patterns to propagate reshapes by expansion. This folds
/// tensor.expand_shape and tensor.collapse_shape ops with their producer
/// and consumer operations respectively.
void populateFoldReshapeOpsByExpansionPatterns(
    RewritePatternSet &patterns,
    const linalg::ControlFusionFn &controlFoldingReshapes);

} // namespace mlir::iree_compiler::IREE::Codegen

#endif // IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_TRANSFORMS_TRANSFORMS_H_
