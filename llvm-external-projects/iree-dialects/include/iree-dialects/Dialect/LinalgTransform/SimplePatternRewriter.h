//===-- SimplePatternRewriter.h - Utility for IR rewrites -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"

namespace mlir {

class MLIRContext;

/// The only purpose of this class is to enable creation of PatternRewriter
/// instances as the base class doesn't have a public constructor.
class SimplePatternRewriter : public PatternRewriter {
public:
  explicit SimplePatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
};

} // namespace mlir
