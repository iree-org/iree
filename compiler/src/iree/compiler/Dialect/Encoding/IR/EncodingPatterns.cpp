// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingPatterns.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

namespace mlir::iree_compiler::IREE::Encoding {
namespace {

//===----------------------------------------------------------------------===//
// Interface-based reification pattern
//===----------------------------------------------------------------------===//

/// Resolves iree_encoding.dim using EncodingDimReificationInterface.
///
/// This pattern handles operations that implement the interface in two ways:
/// 1. Operations that directly provide encoding dims (like set_encoding):
///    The pattern calls reifyEncodingDim() and replaces with the result.
/// 2. Operations that forward encoding dims from a source (like tensor.cast):
///    The pattern calls getEncodingDimSource() and creates a new dim op on
///    that source.
///
struct ReifyEncodingDimFromInterface : public OpRewritePattern<DimOp> {
  using OpRewritePattern::OpRewritePattern;

  // This pattern may create new DimOp, so we need to bound recursion.
  void initialize() { setHasBoundedRewriteRecursion(); }

  LogicalResult matchAndRewrite(DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    OpResult result = dyn_cast<OpResult>(dimOp.getSource());
    if (!result) {
      return failure();
    }

    auto reificationOp =
        dyn_cast<EncodingDimReificationInterface>(result.getOwner());
    if (!reificationOp) {
      return failure();
    }

    int64_t dimIndex = dimOp.getConstantIndex();
    unsigned resultIndex = result.getResultNumber();

    // First, try direct reification (for ops like set_encoding).
    FailureOr<Value> directValue =
        reificationOp.reifyEncodingDim(rewriter, resultIndex, dimIndex);
    if (succeeded(directValue)) {
      rewriter.replaceOp(dimOp, *directValue);
      return success();
    }

    // Fall back to source tracing (for pass-through ops like tensor.cast).
    Value source = reificationOp.getEncodingDimSource(resultIndex);
    if (!source) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<DimOp>(dimOp, source, dimOp.getConstantIndex());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Explicit pattern: Trace through DPS operations (linalg, etc.)
//===----------------------------------------------------------------------===//

/// For DPS operations where the result encoding comes from an init operand,
/// forward the dim query to that init operand.
///
/// This is an explicit pattern because DPS ops don't have a common base class
/// we can attach an external model to - they implement
/// DestinationStyleOpInterface which is an interface, not a concrete op.
///
/// Before:
///   %result = linalg.generic {outs(%init : tensor<?x?xf32, #enc>)} ...
///   %dim = iree_encoding.dim %result[0] : tensor<?x?xf32, #enc>
///
/// After:
///   %dim = iree_encoding.dim %init[0] : tensor<?x?xf32, #enc>
///
struct ReifyEncodingDimThroughDPS : public OpRewritePattern<DimOp> {
  using OpRewritePattern::OpRewritePattern;

  // This pattern creates a new DimOp, so we need to bound recursion.
  void initialize() { setHasBoundedRewriteRecursion(); }

  LogicalResult matchAndRewrite(DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    OpResult result = dyn_cast<OpResult>(dimOp.getSource());
    if (!result) {
      return failure();
    }

    // Skip if already handled by interface.
    if (isa<EncodingDimReificationInterface>(result.getOwner())) {
      return failure();
    }

    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(result.getOwner());
    if (!dpsOp) {
      return failure();
    }

    // For DPS ops, the result encoding comes from the corresponding init.
    OpOperand *tiedInit = dpsOp.getTiedOpOperand(result);
    if (!tiedInit) {
      return failure();
    }

    // Verify encodings match.
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    auto initType = dyn_cast<RankedTensorType>(tiedInit->get().getType());
    if (!resultType || !initType) {
      return failure();
    }

    if (resultType.getEncoding() != initType.getEncoding()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<DimOp>(dimOp, tiedInit->get(),
                                       dimOp.getConstantIndex());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void populateEncodingDimReificationPatterns(RewritePatternSet &patterns) {
  // Interface-based pattern (handles set_encoding, tensor.cast via external
  // model).
  patterns.add<ReifyEncodingDimFromInterface>(patterns.getContext());

  // Explicit pattern for DPS ops (interface-based approach not feasible for
  // DestinationStyleOpInterface since it's an interface, not a concrete op).
  patterns.add<ReifyEncodingDimThroughDPS>(patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::Encoding
