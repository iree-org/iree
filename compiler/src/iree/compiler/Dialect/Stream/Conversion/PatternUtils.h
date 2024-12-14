// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_CONVERSION_PATTERN_UTILS_H_
#define IREE_COMPILER_DIALECT_STREAM_CONVERSION_PATTERN_UTILS_H_

#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::Stream {
class AffinityAnalysis;
} // namespace mlir::iree_compiler::IREE::Stream

namespace mlir::iree_compiler {

// Converts a supported attribute type to the corresponding stream dialect
// value. Returns the provided value if it is natively supported.
TypedAttr convertAttributeToStream(TypedAttr attr);

IREE::Stream::AffinityAttr
tryLookupGlobalAffinity(Operation *op,
                        IREE::Stream::AffinityAnalysis *affinityAnalysis);
IREE::Stream::AffinityAttr
tryLookupExecutionAffinity(Operation *op,
                           IREE::Stream::AffinityAnalysis *affinityAnalysis);
IREE::Stream::AffinityAttr
tryLookupResultAffinity(Value value,
                        IREE::Stream::AffinityAnalysis *affinityAnalysis);

struct ConvertedTensor {
  // Optional affinity of the resource at the time it is consumed.
  // May be nullptr if the affinity could not be determined.
  IREE::Stream::AffinityAttr affinity;
  // Resource storing the tensor.
  Value resource;
  // Size of the resource in bytes.
  Value resourceSize;
};

ConvertedTensor resolveTensorOperands(
    Location loc, Value originalOperand, ValueRange convertedOperand,
    IREE::Stream::AffinityAnalysis *affinityAnalysis, OpBuilder &builder);
ConvertedTensor transferTensorOperands(
    Location loc, Value originalOperand, ValueRange convertedOperand,
    IREE::Stream::AffinityAttr requiredAffinityAttr,
    IREE::Stream::AffinityAnalysis *affinityAnalysis, OpBuilder &builder);

template <typename OpT>
struct AffinityAwareConversionPattern : public OpConversionPattern<OpT> {
public:
  AffinityAwareConversionPattern(
      const TypeConverter &typeConverter, MLIRContext *context,
      IREE::Stream::AffinityAnalysis *affinityAnalysis,
      PatternBenefit benefit = 1)
      : OpConversionPattern<OpT>(typeConverter, context, benefit),
        affinityAnalysis(affinityAnalysis) {}

  IREE::Stream::AffinityAnalysis *getAffinityAnalysis() const {
    return affinityAnalysis;
  }

protected:
  ConvertedTensor resolveTensorOperands(Location loc, Value originalOperand,
                                        ValueRange convertedOperand,
                                        OpBuilder &builder) const {
    return mlir::iree_compiler::resolveTensorOperands(
        loc, originalOperand, convertedOperand, affinityAnalysis, builder);
  }

  ConvertedTensor
  transferTensorOperands(Location loc, Value originalOperand,
                         ValueRange convertedOperand,
                         IREE::Stream::AffinityAttr requiredAffinityAttr,
                         OpBuilder &builder) const {
    return mlir::iree_compiler::transferTensorOperands(
        loc, originalOperand, convertedOperand, requiredAffinityAttr,
        affinityAnalysis, builder);
  }

  IREE::Stream::AffinityAttr lookupResultAffinity(Value originalResult) const {
    return mlir::iree_compiler::tryLookupResultAffinity(originalResult,
                                                        affinityAnalysis);
  }

  IREE::Stream::AffinityAnalysis *affinityAnalysis = nullptr;
};

template <typename OpT>
struct AffinityOpConversionPattern
    : public AffinityAwareConversionPattern<OpT> {
public:
  AffinityOpConversionPattern(const TypeConverter &typeConverter,
                              MLIRContext *context,
                              IREE::Stream::AffinityAnalysis *affinityAnalysis,
                              PatternBenefit benefit = 1)
      : AffinityAwareConversionPattern<OpT>(typeConverter, context,
                                            affinityAnalysis, benefit) {}

protected:
  virtual LogicalResult matchAndRewriteOnAffinity(
      OpT op, typename OpConversionPattern<OpT>::OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const = 0;

private:
  LogicalResult
  matchAndRewrite(OpT op,
                  typename OpConversionPattern<OpT>::OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override final {
    auto executionAffinityAttr =
        tryLookupExecutionAffinity(op, this->getAffinityAnalysis());
    return matchAndRewriteOnAffinity(op, adaptor, executionAffinityAttr,
                                     rewriter);
  }
};

void replaceOpWithMultiple(Operation *op,
                           ArrayRef<SmallVector<Value>> replacements,
                           ConversionPatternRewriter &rewriter);
void replaceOpWithMultiple(Operation *op, ValueRange resources,
                           ValueRange sizes,
                           ConversionPatternRewriter &rewriter);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_STREAM_CONVERSION_PATTERN_UTILS_H_
