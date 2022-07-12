// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;

/// The only purpose of this class is to enable creation of PatternRewriter
/// instances as the base class doesn't have a public constructor.
/// The op-based constructor sets the insertion point before the `op`.
class SimplePatternRewriter : public PatternRewriter {
public:
  SimplePatternRewriter(MLIRContext *context) : PatternRewriter(context) {}

  SimplePatternRewriter(Operation *op) : PatternRewriter(op->getContext()) {
    setInsertionPoint(op);
  }
};
} // namespace mlir
