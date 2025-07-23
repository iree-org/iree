// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_REGIONOPUTILS_H_
#define IREE_COMPILER_UTILS_REGIONOPUTILS_H_

#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler {

/// Move the definition of operands of `operations` before `insertionPoint`,
/// ensuring that dominance properties are maintained. Operations listed in
/// `ignoreOperations` are excluded from the slice of defining operations of the
/// operands.
///
/// TODO(#21451): Generalize upstream `moveOperationDependencies`.
LogicalResult moveOperandDefs(RewriterBase &rewriter,
                              ArrayRef<Operation *> operations,
                              Operation *insertionPoint,
                              DominanceInfo &dominanceInfo,
                              ArrayRef<Operation *> ignoreOperations = {});

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_REGIONOPUTILS_H_
