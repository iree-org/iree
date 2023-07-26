// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PATTERNS_H_
#define IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PATTERNS_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

// Populates |patterns| with some risky/IREE-specific canonicalization patterns.
// Some of these apply to other dialects (such as std/builtin) and could be
// upstreamed after some more exhaustive investigation.
void populateCommonPatterns(MLIRContext *context, RewritePatternSet &patterns);

// Populates patterns that fold DAGs of linalg-based operations into
// util.constexpr where the folding is guaranteed to be valid and
// deemed cost-neutral. This is less sophisticated than a global
// analysis/transform which can make trickier trade-offs, but it
// should be sufficient/beneficial for iterative IR modification
// and simplification as is done during fusion and other places.
void populateTrivialLinalgConstexprFoldingOperations(
    RewritePatternSet &patterns);

} // namespace Util
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PATTERNS_H_
