// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/FlowToStream/Patterns.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Stream/Conversion/PatternUtils.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

namespace {

static SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> vec;
  for (auto v : values) {
    vec.append(v.begin(), v.end());
  }
  return vec;
}

// Inserts a sizeof calculation for the given tensor value type and dims.
// This should only be used to produce sizes for values produced by an op; the
// size of operands must be queried from the input resource.
static Value buildResultSizeOf(Location loc, Value tensorValue,
                               ValueRange dynamicDims,
                               IREE::Stream::AffinityAttr affinityAttr,
                               ConversionPatternRewriter &rewriter) {
  // TODO(benvanik): see if we can stash this on the side to avoid expensive
  // materialization of a bunch of redundant IR.
  return rewriter.create<IREE::Stream::TensorSizeOfOp>(
      loc, rewriter.getIndexType(), TypeAttr::get(tensorValue.getType()),
      dynamicDims, affinityAttr);
}

struct ConvertTensorConstantOp
    : public AffinityOpConversionPattern<IREE::Flow::TensorConstantOp> {
public:
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::TensorConstantOp constantOp, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    // Capture the tensor constant strongly typed with constant lifetime.
    auto constantType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::Constant);
    auto newOp = rewriter.create<IREE::Stream::TensorConstantOp>(
        constantOp.getLoc(), constantType,
        convertAttributeToStream(constantOp.getValue()),
        TypeAttr::get(constantOp.getType()), ValueRange{},
        executionAffinityAttr);

    // Transfer to unknown lifetime.
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto constantSize = rewriter.createOrFold<IREE::Stream::ResourceSizeOp>(
        constantOp.getLoc(), rewriter.getIndexType(), newOp.getResult());
    auto transferOp = rewriter.create<IREE::Stream::AsyncTransferOp>(
        constantOp.getLoc(), unknownType, newOp.getResult(), constantSize,
        constantSize,
        /*source_affinity=*/executionAffinityAttr,
        /*result_affinity=*/executionAffinityAttr);
    rewriter.replaceOpWithMultiple(constantOp,
                                   {{transferOp.getResult(), constantSize}});
    return success();
  }
};

struct ConvertTensorDynamicConstantOp
    : public AffinityOpConversionPattern<IREE::Flow::TensorDynamicConstantOp> {
public:
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::TensorDynamicConstantOp constantOp, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto attrType = dyn_cast<RankedTensorType>(constantOp.getValue().getType());
    if (!attrType)
      return failure();
    auto resultType = constantOp.getType();

    // If the op is acting as a dynamic value then preserve that behavior by
    // calculating the shape through optimization barriers.
    SmallVector<Value> dynamicDims;
    for (unsigned i = 0; i < resultType.getRank(); ++i) {
      if (resultType.isDynamicDim(i)) {
        Value staticDim = rewriter.create<arith::ConstantIndexOp>(
            constantOp.getLoc(), attrType.getDimSize(i));
        Value dynamicDim = rewriter
                               .create<IREE::Util::OptimizationBarrierOp>(
                                   constantOp.getLoc(), staticDim)
                               .getResult(0);
        dynamicDims.push_back(dynamicDim);
      }
    }

    // Capture the tensor constant strongly typed with constant lifetime.
    auto constantType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::Constant);
    auto newOp = rewriter.create<IREE::Stream::TensorConstantOp>(
        constantOp.getLoc(), constantType,
        convertAttributeToStream(constantOp.getValue()),
        TypeAttr::get(resultType), dynamicDims, executionAffinityAttr);

    // Transfer to unknown lifetime.
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto constantSize = rewriter.createOrFold<IREE::Stream::ResourceSizeOp>(
        constantOp.getLoc(), rewriter.getIndexType(), newOp.getResult());
    auto transferOp = rewriter.create<IREE::Stream::AsyncTransferOp>(
        constantOp.getLoc(), unknownType, newOp.getResult(), constantSize,
        constantSize,
        /*source_affinity=*/executionAffinityAttr,
        /*result_affinity=*/executionAffinityAttr);
    rewriter.replaceOpWithMultiple(constantOp, {{transferOp, constantSize}});
    return success();
  }
};

// Reshapes and bitcasts become clones here to preserve shape/element type
// information (which may become actual transfers depending on source/target
// shape) - they'll be elided if not needed.
//
// NOTE: we transfer to the target before cloning. This may not be optimal
// as the clone may otherwise have been able to be elided on the producer
// side but we leave that for future copy elision to determine.
template <typename CastOpTy>
struct ConvertTensorCastLikeOp
    : public AffinityAwareConversionPattern<CastOpTy> {
  using AffinityAwareConversionPattern<
      CastOpTy>::AffinityAwareConversionPattern;
  LogicalResult matchAndRewrite(
      CastOpTy op,
      typename OpConversionPattern<CastOpTy>::OneToNOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultAffinityAttr = this->lookupResultAffinity(op.getResult());
    auto source = this->transferTensorOperands(op.getLoc(), op.getSource(),
                                               adaptor.getSource(),
                                               resultAffinityAttr, rewriter);
    auto resultSize =
        buildResultSizeOf(op.getLoc(), op.getResult(), op.getResultDims(),
                          resultAffinityAttr, rewriter);
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    Value cloneOp = rewriter.create<IREE::Stream::TensorCloneOp>(
        op.getLoc(), unknownType, source.resource, op.getSource().getType(),
        op.getSourceDims(), source.resourceSize, op.getResult().getType(),
        flattenValues(adaptor.getResultDims()), resultSize, resultAffinityAttr);
    rewriter.replaceOpWithMultiple(op, {{cloneOp, resultSize}});
    return success();
  }
};

struct ConvertTensorAllocaOp
    : public AffinityOpConversionPattern<IREE::Flow::TensorAllocaOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::TensorAllocaOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto resultSize =
        buildResultSizeOf(op.getLoc(), op.getResult(), op.getResultDims(),
                          executionAffinityAttr, rewriter);
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto allocaOp = rewriter.create<IREE::Stream::AsyncAllocaOp>(
        op.getLoc(), unknownType, resultSize, executionAffinityAttr);
    rewriter.replaceOpWithMultiple(op, {{allocaOp.getResult(), resultSize}});
    return success();
  }
};

struct ConvertTensorEmptyOp
    : public AffinityOpConversionPattern<IREE::Flow::TensorEmptyOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::TensorEmptyOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto resultSize =
        buildResultSizeOf(op.getLoc(), op.getResult(), op.getResultDims(),
                          executionAffinityAttr, rewriter);
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto emptyOp = rewriter.create<IREE::Stream::TensorEmptyOp>(
        op.getLoc(), unknownType, op.getResult().getType(),
        flattenValues(adaptor.getResultDims()), resultSize,
        executionAffinityAttr);
    rewriter.replaceOpWithMultiple(op, {{emptyOp.getResult(), resultSize}});
    return success();
  }
};

struct ConvertTensorSplatOp
    : public AffinityOpConversionPattern<IREE::Flow::TensorSplatOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::TensorSplatOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto resultSize =
        buildResultSizeOf(op.getLoc(), op.getResult(), op.getResultDims(),
                          executionAffinityAttr, rewriter);
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto splatOp = rewriter.create<IREE::Stream::TensorSplatOp>(
        op.getLoc(), unknownType, adaptor.getValue().front(),
        op.getResult().getType(), flattenValues(adaptor.getResultDims()),
        resultSize, executionAffinityAttr);
    rewriter.replaceOpWithMultiple(op, {{splatOp, resultSize}});
    return success();
  }
};

struct ConvertTensorCloneOp
    : public AffinityOpConversionPattern<IREE::Flow::TensorCloneOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::TensorCloneOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto operand = transferTensorOperands(op.getLoc(), op.getOperand(),
                                          adaptor.getOperand(),
                                          executionAffinityAttr, rewriter);
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto cloneOp = rewriter.create<IREE::Stream::TensorCloneOp>(
        op.getLoc(), unknownType, operand.resource, op.getOperand().getType(),
        op.getOperandDims(), operand.resourceSize, op.getResult().getType(),
        flattenValues(adaptor.getOperandDims()), operand.resourceSize,
        executionAffinityAttr);
    rewriter.replaceOpWithMultiple(op, {{cloneOp, operand.resourceSize}});
    return success();
  }
};

struct ConvertTensorBarrierOp
    : public AffinityOpConversionPattern<IREE::Flow::TensorBarrierOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::TensorBarrierOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto operand = resolveTensorOperands(op.getLoc(), op.getOperand(),
                                         adaptor.getOperand(), rewriter);
    auto barrierOp = rewriter.create<IREE::Stream::AsyncBarrierOp>(
        op.getLoc(), operand.resource.getType(), operand.resource,
        operand.resourceSize,
        /*affinity=*/executionAffinityAttr);
    rewriter.replaceOpWithMultiple(op, {{barrierOp, operand.resourceSize}});
    return success();
  }
};

struct ConvertTensorTransferOp
    : public AffinityOpConversionPattern<IREE::Flow::TensorTransferOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::TensorTransferOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    if (!executionAffinityAttr) {
      return rewriter.notifyMatchFailure(op, "invalid stream affinity attr");
    }
    auto operand = resolveTensorOperands(op.getLoc(), op.getOperand(),
                                         adaptor.getOperand(), rewriter);
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto transferOp = rewriter.create<IREE::Stream::AsyncTransferOp>(
        op.getLoc(), unknownType, operand.resource, operand.resourceSize,
        operand.resourceSize,
        /*source_affinity=*/operand.affinity,
        /*result_affinity=*/executionAffinityAttr);
    rewriter.replaceOpWithMultiple(op, {{transferOp, operand.resourceSize}});
    return success();
  }
};

struct ConvertTensorSliceOp
    : public AffinityOpConversionPattern<IREE::Flow::TensorSliceOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::TensorSliceOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto source =
        transferTensorOperands(op.getLoc(), op.getSource(), adaptor.getSource(),
                               executionAffinityAttr, rewriter);
    auto resultSize =
        buildResultSizeOf(op.getLoc(), op.getResult(), op.getResultDims(),
                          executionAffinityAttr, rewriter);
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto sliceOp = rewriter.create<IREE::Stream::TensorSliceOp>(
        op.getLoc(), unknownType, source.resource, op.getSource().getType(),
        op.getSourceDims(), source.resourceSize,
        flattenValues(adaptor.getStartIndices()),
        flattenValues(adaptor.getLengths()), op.getResult().getType(),
        flattenValues(adaptor.getResultDims()), resultSize,
        executionAffinityAttr);
    rewriter.replaceOpWithMultiple(op, {{sliceOp, resultSize}});
    return success();
  }
};

struct ConvertTensorUpdateOp
    : public AffinityOpConversionPattern<IREE::Flow::TensorUpdateOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::TensorUpdateOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto target =
        transferTensorOperands(op.getLoc(), op.getTarget(), adaptor.getTarget(),
                               executionAffinityAttr, rewriter);
    auto update =
        transferTensorOperands(op.getLoc(), op.getUpdate(), adaptor.getUpdate(),
                               executionAffinityAttr, rewriter);
    auto updateOp = rewriter.create<IREE::Stream::TensorUpdateOp>(
        op.getLoc(), target.resource.getType(), target.resource,
        op.getTarget().getType(), flattenValues(adaptor.getTargetDims()),
        target.resourceSize, flattenValues(adaptor.getStartIndices()),
        update.resource, op.getUpdate().getType(), op.getUpdateDims(),
        update.resourceSize, executionAffinityAttr);
    rewriter.replaceOpWithMultiple(
        op, {{updateOp.getResult(), target.resourceSize}});
    return success();
  }
};

static bool isScalarTensor(RankedTensorType type) {
  if (type.getRank() == 0)
    return true; // tensor<i32>
  if (!type.hasStaticShape())
    return false; // tensor<...?...xi32>
  int64_t elementCount = 1;
  for (int64_t dim : type.getShape())
    elementCount *= dim;
  return elementCount == 1; // tensor<1xi32> or tensor<1x1x1xi32>
}

struct ConvertTensorLoadOp
    : public AffinityAwareConversionPattern<IREE::Flow::TensorLoadOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::TensorLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto source = resolveTensorOperands(op.getLoc(), op.getSource(),
                                        adaptor.getSource(), rewriter);

    // If the source is not a staging resource then we need to transfer it to
    // a staging resource. We slice out just what is being loaded so that we
    // don't transfer the entire tensor. If loading multiple values from the
    // same tensor we'll either want to have batched that before this point
    // by loading an entire buffer or after by coalescing the slices.
    //
    // If already a staging resource then we can fast-path load the value.
    auto stagingType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::Staging);
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    SmallVector<Value> convertedSourceDims =
        flattenValues(adaptor.getSourceDims());
    SmallVector<Value> convertedIndices = flattenValues(adaptor.getIndices());
    if (source.resource.getType() == stagingType) {
      rewriter.replaceOpWithNewOp<IREE::Stream::TensorLoadOp>(
          op, resultType, source.resource, op.getSource().getType(),
          convertedSourceDims, source.resourceSize, convertedIndices);
      return success();
    }

    // Scalar tensors get transferred without slicing.
    auto sourceEncoding = op.getSource().getType();
    if (isScalarTensor(sourceEncoding)) {
      auto transferOp = rewriter.create<IREE::Stream::AsyncTransferOp>(
          op.getLoc(), stagingType, source.resource, source.resourceSize,
          source.resourceSize,
          /*source_affinity=*/source.affinity,
          /*result_affinity=*/source.affinity);
      rewriter.replaceOpWithNewOp<IREE::Stream::TensorLoadOp>(
          op, resultType, transferOp.getResult(), sourceEncoding,
          convertedSourceDims, transferOp.getResultSize(), convertedIndices);
      return success();
    }

    // Slice out the individual element value.
    IndexSet indexSet(op.getLoc(), rewriter);
    indexSet.populate(convertedIndices);
    SmallVector<Value> sliceIndices;
    SmallVector<Value> sliceLengths;
    SmallVector<Value> loadIndices;
    SmallVector<int64_t> resultDims;
    for (auto index : convertedIndices) {
      // TODO(benvanik): support larger buffer slices.
      sliceIndices.push_back(index);
      sliceLengths.push_back(indexSet.get(1));
      loadIndices.push_back(indexSet.get(0));
      resultDims.push_back(1);
    }
    auto resultEncoding =
        RankedTensorType::get(resultDims, sourceEncoding.getElementType(),
                              sourceEncoding.getEncoding());
    Value resultSize = rewriter.create<IREE::Stream::TensorSizeOfOp>(
        op.getLoc(), resultEncoding, ValueRange{}, source.affinity);
    auto sliceOp = rewriter.create<IREE::Stream::TensorSliceOp>(
        op.getLoc(), source.resource.getType(), source.resource, sourceEncoding,
        convertedSourceDims, source.resourceSize, sliceIndices, sliceLengths,
        resultEncoding, ValueRange{}, resultSize, source.affinity);
    auto transferOp = rewriter.create<IREE::Stream::AsyncTransferOp>(
        op.getLoc(), stagingType, sliceOp.getResult(), sliceOp.getResultSize(),
        sliceOp.getResultSize(),
        /*source_affinity=*/source.affinity,
        /*result_affinity=*/source.affinity);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorLoadOp>(
        op, resultType, transferOp.getResult(), sliceOp.getResultEncoding(),
        sliceOp.getResultEncodingDims(), transferOp.getResultSize(),
        loadIndices);
    return success();
  }
};

struct ConvertTensorStoreOp
    : public AffinityAwareConversionPattern<IREE::Flow::TensorStoreOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::TensorStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto target = resolveTensorOperands(op.getLoc(), op.getTarget(),
                                        adaptor.getTarget(), rewriter);

    // If the target is a staging resource then we can directly store into it
    // with a fast-path. Otherwise we need to stage an upload.
    auto stagingType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::Staging);
    if (target.resource.getType() == stagingType) {
      auto storeOp = rewriter.create<IREE::Stream::TensorStoreOp>(
          op.getLoc(), target.resource.getType(), target.resource,
          op.getTarget().getType(), flattenValues(adaptor.getTargetDims()),
          target.resourceSize, flattenValues(adaptor.getIndices()),
          adaptor.getValue().front());
      rewriter.replaceOpWithMultiple(op, {{storeOp, target.resourceSize}});
      return success();
    }

    // Use fill to store the value.
    // TODO(benvanik): support larger buffer slices (stage + update).
    IndexSet indexSet(op.getLoc(), rewriter);
    SmallVector<Value> convertedIndices = flattenValues(adaptor.getIndices());
    indexSet.populate(convertedIndices);
    SmallVector<Value> lengths(convertedIndices.size(), indexSet.get(1));
    auto targetEncoding = op.getTarget().getType();
    auto fillOp = rewriter.create<IREE::Stream::TensorFillOp>(
        op.getLoc(), target.resource, targetEncoding,
        flattenValues(adaptor.getTargetDims()), target.resourceSize,
        convertedIndices, lengths, adaptor.getValue().front(), target.affinity);
    rewriter.replaceOpWithMultiple(op, {{fillOp, target.resourceSize}});
    return success();
  }
};

struct ConvertTensorTraceOp
    : public AffinityAwareConversionPattern<IREE::Flow::TensorTraceOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::TensorTraceOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> resources;
    SmallVector<Value> resourceSizes;
    SmallVector<Attribute> resourceEncodings;
    for (auto [tensorOperand, resourceOperand] :
         llvm::zip_equal(op.getValues(), adaptor.getValues())) {
      auto source = resolveTensorOperands(op.getLoc(), tensorOperand,
                                          resourceOperand, rewriter);
      auto stagingType = rewriter.getType<IREE::Stream::ResourceType>(
          IREE::Stream::Lifetime::Staging);
      auto traceSource = source.resource;
      if (source.resource.getType() != stagingType) {
        traceSource = rewriter.create<IREE::Stream::AsyncTransferOp>(
            op.getLoc(), stagingType, source.resource, source.resourceSize,
            source.resourceSize,
            /*source_affinity=*/source.affinity,
            /*result_affinity=*/source.affinity);
      }
      resources.push_back(traceSource);
      resourceSizes.push_back(source.resourceSize);
      resourceEncodings.push_back(TypeAttr::get(tensorOperand.getType()));
    }
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorTraceOp>(
        op, adaptor.getKey(), resources, resourceSizes,
        rewriter.getArrayAttr(resourceEncodings),
        flattenValues(adaptor.getValueDims()));
    return success();
  }
};

struct ConvertChannelDefaultOp
    : public AffinityOpConversionPattern<IREE::Flow::ChannelDefaultOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::ChannelDefaultOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::Stream::ChannelCreateOp>(
        op,
        /*id=*/Value{},
        /*group=*/adaptor.getGroupAttr(),
        /*rank=*/Value{},
        /*count=*/Value{}, executionAffinityAttr);
    return success();
  }
};

struct ConvertChannelSplitOp
    : public OpConversionPattern<IREE::Flow::ChannelSplitOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::ChannelSplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::Stream::ChannelSplitOp>(
        op, adaptor.getChannel(), adaptor.getColor(), adaptor.getKey());
    return success();
  }
};

struct ConvertChannelRankOp
    : public OpConversionPattern<IREE::Flow::ChannelRankOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::ChannelRankOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::Stream::ChannelRankOp>(
        op, adaptor.getOperands());
    return success();
  }
};

struct ConvertChannelCountOp
    : public OpConversionPattern<IREE::Flow::ChannelCountOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::ChannelCountOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::Stream::ChannelCountOp>(
        op, adaptor.getOperands());
    return success();
  }
};

struct ConvertAllGatherOp
    : public AffinityOpConversionPattern<IREE::Flow::CollectiveAllGatherOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::CollectiveAllGatherOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto collectiveAttr = rewriter.getAttr<IREE::Stream::CollectiveAttr>(
        IREE::Stream::CollectiveKind::AllGather,
        /*reduction=*/std::nullopt,
        static_cast<IREE::Stream::CollectiveElementType>(op.getElementType()));

    auto zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto elementCount = rewriter.create<arith::ConstantIndexOp>(
        op.getLoc(), op.getType().getNumElements());
    auto newTargetCast =
        transferTensorOperands(op.getLoc(), op.getTarget(), adaptor.getTarget(),
                               executionAffinityAttr, rewriter);
    auto newSourceCast =
        transferTensorOperands(op.getLoc(), op.getSource(), adaptor.getSource(),
                               executionAffinityAttr, rewriter);

    auto collectiveOp = rewriter.create<IREE::Stream::AsyncCollectiveOp>(
        op.getLoc(), collectiveAttr,
        /*target=*/newTargetCast.resource,
        /*target_size=*/newTargetCast.resourceSize,
        /*target_offset=*/zeroOffset,
        /*target_end=*/newTargetCast.resourceSize,
        /*target_length=*/newTargetCast.resourceSize,
        /*source=*/newSourceCast.resource,
        /*source_size=*/newSourceCast.resourceSize,
        /*source_offset=*/zeroOffset,
        /*source_end=*/newSourceCast.resourceSize,
        /*source_length=*/newSourceCast.resourceSize,
        /*element_count=*/elementCount,
        /*channel=*/adaptor.getChannel().front(),
        /*param=*/mlir::Value(), executionAffinityAttr);
    rewriter.replaceOpWithMultiple(
        op, {{collectiveOp, newTargetCast.resourceSize}});
    return success();
  }
};

struct ConvertAllReduceOp
    : public AffinityOpConversionPattern<IREE::Flow::CollectiveAllReduceOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::CollectiveAllReduceOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto collectiveAttr = rewriter.getAttr<IREE::Stream::CollectiveAttr>(
        IREE::Stream::CollectiveKind::AllReduce,
        static_cast<IREE::Stream::CollectiveReductionOp>(op.getReductionOp()),
        static_cast<IREE::Stream::CollectiveElementType>(op.getElementType()));

    auto zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto elementCount = rewriter.create<arith::ConstantIndexOp>(
        op.getLoc(), op.getType().getNumElements());
    auto newTargetCast =
        transferTensorOperands(op.getLoc(), op.getTarget(), adaptor.getTarget(),
                               executionAffinityAttr, rewriter);
    auto newSourceCast =
        transferTensorOperands(op.getLoc(), op.getSource(), adaptor.getSource(),
                               executionAffinityAttr, rewriter);

    auto collectiveOp = rewriter.create<IREE::Stream::AsyncCollectiveOp>(
        op.getLoc(), collectiveAttr,
        /*target=*/newTargetCast.resource,
        /*target_size=*/newTargetCast.resourceSize,
        /*target_offset=*/zeroOffset,
        /*target_end=*/newTargetCast.resourceSize,
        /*target_length=*/newTargetCast.resourceSize,
        /*source=*/newSourceCast.resource,
        /*source_size=*/newSourceCast.resourceSize,
        /*source_offset=*/zeroOffset,
        /*source_end=*/newSourceCast.resourceSize,
        /*source_length=*/newSourceCast.resourceSize,
        /*element_count=*/elementCount,
        /*channel=*/adaptor.getChannel().front(),
        /*param=*/mlir::Value(), executionAffinityAttr);
    rewriter.replaceOpWithMultiple(
        op, {{collectiveOp, newTargetCast.resourceSize}});
    return success();
  }
};

struct ConvertAllToAllOp
    : public AffinityOpConversionPattern<IREE::Flow::CollectiveAllToAllOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::CollectiveAllToAllOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto collectiveAttr = rewriter.getAttr<IREE::Stream::CollectiveAttr>(
        IREE::Stream::CollectiveKind::AllToAll,
        /*reduction=*/std::nullopt,
        static_cast<IREE::Stream::CollectiveElementType>(op.getElementType()));

    auto zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto elementCount = rewriter.create<arith::ConstantIndexOp>(
        op.getLoc(), op.getType().getNumElements());
    auto newTargetCast =
        transferTensorOperands(op.getLoc(), op.getTarget(), adaptor.getTarget(),
                               executionAffinityAttr, rewriter);
    auto newSourceCast =
        transferTensorOperands(op.getLoc(), op.getSource(), adaptor.getSource(),
                               executionAffinityAttr, rewriter);

    auto collectiveOp = rewriter.create<IREE::Stream::AsyncCollectiveOp>(
        op.getLoc(), collectiveAttr,
        /*target=*/newTargetCast.resource,
        /*target_size=*/newTargetCast.resourceSize,
        /*target_offset=*/zeroOffset,
        /*target_end=*/newTargetCast.resourceSize,
        /*target_length=*/newTargetCast.resourceSize,
        /*source=*/newSourceCast.resource,
        /*source_size=*/newSourceCast.resourceSize,
        /*source_offset=*/zeroOffset,
        /*source_end=*/newSourceCast.resourceSize,
        /*source_length=*/newSourceCast.resourceSize,
        /*element_count=*/elementCount,
        /*channel=*/adaptor.getChannel().front(),
        /*param=*/mlir::Value(), executionAffinityAttr);
    rewriter.replaceOpWithMultiple(
        op, {{collectiveOp, newTargetCast.resourceSize}});
    return success();
  }
};

struct ConvertReduceScatterOp : public AffinityOpConversionPattern<
                                    IREE::Flow::CollectiveReduceScatterOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::CollectiveReduceScatterOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto collectiveAttr = rewriter.getAttr<IREE::Stream::CollectiveAttr>(
        IREE::Stream::CollectiveKind::ReduceScatter,
        static_cast<IREE::Stream::CollectiveReductionOp>(op.getReductionOp()),
        static_cast<IREE::Stream::CollectiveElementType>(op.getElementType()));

    auto zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto elementCount = rewriter.create<arith::ConstantIndexOp>(
        op.getLoc(), op.getType().getNumElements());
    auto newTargetCast =
        transferTensorOperands(op.getLoc(), op.getTarget(), adaptor.getTarget(),
                               executionAffinityAttr, rewriter);
    auto newSourceCast =
        transferTensorOperands(op.getLoc(), op.getSource(), adaptor.getSource(),
                               executionAffinityAttr, rewriter);

    auto collectiveOp = rewriter.create<IREE::Stream::AsyncCollectiveOp>(
        op.getLoc(), collectiveAttr,
        /*target=*/newTargetCast.resource,
        /*target_size=*/newTargetCast.resourceSize,
        /*target_offset=*/zeroOffset,
        /*target_end=*/newTargetCast.resourceSize,
        /*target_length=*/newTargetCast.resourceSize,
        /*source=*/newSourceCast.resource,
        /*source_size=*/newSourceCast.resourceSize,
        /*source_offset=*/zeroOffset,
        /*source_end=*/newSourceCast.resourceSize,
        /*source_length=*/newSourceCast.resourceSize,
        /*element_count=*/elementCount,
        /*channel=*/adaptor.getChannel().front(),
        /*param=*/mlir::Value(), executionAffinityAttr);
    rewriter.replaceOpWithMultiple(
        op, {{collectiveOp, newTargetCast.resourceSize}});
    return success();
  }
};

struct ConvertCollectiveSendRecvOp
    : public AffinityOpConversionPattern<IREE::Flow::CollectiveSendRecvOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::CollectiveSendRecvOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    auto collectiveAttr = rewriter.getAttr<IREE::Stream::CollectiveAttr>(
        IREE::Stream::CollectiveKind::SendRecv,
        /*reduction=*/std::nullopt,
        static_cast<IREE::Stream::CollectiveElementType>(op.getElementType()));

    auto zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto elementCount = rewriter.create<arith::ConstantIndexOp>(
        op.getLoc(), op.getType().getNumElements());
    auto newTargetCast =
        transferTensorOperands(op.getLoc(), op.getTarget(), adaptor.getTarget(),
                               executionAffinityAttr, rewriter);
    auto newSourceCast =
        transferTensorOperands(op.getLoc(), op.getSource(), adaptor.getSource(),
                               executionAffinityAttr, rewriter);

    // Pack send, recv into param. The values are checked to be within the
    // 16-bit range during lowering to Flow dialect.
    auto send = rewriter.create<arith::IndexCastOp>(
        op.getLoc(), rewriter.getI32Type(), adaptor.getSend());
    auto lo = rewriter.create<arith::AndIOp>(
        op.getLoc(), send,
        rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0xFFFF, 32));
    auto recv = rewriter.create<arith::IndexCastOp>(
        op.getLoc(), rewriter.getI32Type(), adaptor.getRecv());
    auto hi = rewriter.create<arith::ShLIOp>(
        op.getLoc(), recv,
        rewriter.create<arith::ConstantIntOp>(op.getLoc(), 16, 32));
    auto param = rewriter.create<arith::OrIOp>(op.getLoc(), hi, lo);

    auto collectiveOp = rewriter.create<IREE::Stream::AsyncCollectiveOp>(
        op.getLoc(), collectiveAttr,
        /*target=*/newTargetCast.resource,
        /*target_size=*/newTargetCast.resourceSize,
        /*target_offset=*/zeroOffset,
        /*target_end=*/newTargetCast.resourceSize,
        /*target_length=*/newTargetCast.resourceSize,
        /*source=*/newSourceCast.resource,
        /*source_size=*/newSourceCast.resourceSize,
        /*source_offset=*/zeroOffset,
        /*source_end=*/newSourceCast.resourceSize,
        /*source_length=*/newSourceCast.resourceSize,
        /*element_count=*/elementCount,
        /*channel=*/adaptor.getChannel().front(),
        /*param=*/param, executionAffinityAttr);
    rewriter.replaceOpWithMultiple(
        op, {{collectiveOp, newTargetCast.resourceSize}});
    return success();
  }
};

struct ConvertDispatchOp
    : public AffinityOpConversionPattern<IREE::Flow::DispatchOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::DispatchOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    // Query and resolve all operands and their sizes.
    SmallVector<Value> operands;
    SmallVector<Value> operandSizes;
    SmallVector<Value> allOperandSizes;
    SmallVector<Type> operandEncodings;
    for (auto [oldOperand, convertedOperands] :
         llvm::zip_equal(op.getArguments(), adaptor.getArguments())) {
      Value newOperand;
      if (llvm::isa<ShapedType>(oldOperand.getType())) {
        auto newOperandCast =
            transferTensorOperands(op.getLoc(), oldOperand, convertedOperands,
                                   executionAffinityAttr, rewriter);
        newOperand = newOperandCast.resource;
        operandSizes.push_back(newOperandCast.resourceSize);
        allOperandSizes.push_back(newOperandCast.resourceSize);
        operandEncodings.push_back(oldOperand.getType());
      } else {
        allOperandSizes.push_back({});
        operandEncodings.push_back(rewriter.getType<IREE::Util::UnusedType>());
        newOperand = convertedOperands.front();
      }
      operands.push_back(newOperand);
    }

    // Construct result sizes or reuse tied operand sizes from above.
    SmallVector<Value> resultSizes;
    SmallVector<Type> resultTypes;
    SmallVector<Type> resultEncodings;
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto tiedOperandBase = op.getTiedOperandsIndexAndLength().first;
    for (auto result : llvm::enumerate(op.getResults())) {
      auto oldResultType = result.value().getType();
      if (!llvm::isa<ShapedType>(oldResultType)) {
        resultTypes.push_back(getTypeConverter()->convertType(oldResultType));
        resultEncodings.push_back(rewriter.getType<IREE::Util::UnusedType>());
        continue;
      }
      auto tiedOperand = op.getTiedResultOperandIndex(result.index());
      if (tiedOperand.has_value()) {
        auto operandIndex = tiedOperand.value() - tiedOperandBase;
        resultSizes.push_back(allOperandSizes[operandIndex]);
        resultTypes.push_back(operands[operandIndex].getType());
        resultEncodings.push_back(operandEncodings[operandIndex]);
      } else {
        auto resultDynamicDims = IREE::Util::buildDynamicDimsForValue(
            op.getLoc(), result.value(), rewriter);
        resultSizes.push_back(
            buildResultSizeOf(op.getLoc(), result.value(), resultDynamicDims,
                              executionAffinityAttr, rewriter));
        resultTypes.push_back(unknownType);
        resultEncodings.push_back(oldResultType);
      }
    }

    auto newOp = rewriter.create<IREE::Stream::TensorDispatchOp>(
        op.getLoc(), resultTypes, flattenValues(adaptor.getWorkload()),
        adaptor.getEntryPointsAttr(), operands, operandSizes,
        rewriter.getTypeArrayAttr(operandEncodings), op.getArgumentDims(),
        resultSizes, rewriter.getTypeArrayAttr(resultEncodings),
        op.getResultDims(), adaptor.getTiedOperandsAttr(),
        executionAffinityAttr);
    newOp->setDialectAttrs(
        llvm::make_filter_range(op->getDialectAttrs(), [](NamedAttribute attr) {
          return attr.getName() != "stream.affinity";
        }));
    SmallVector<SmallVector<Value>> replacementsVec = llvm::map_to_vector(
        llvm::zip_equal(newOp->getResults(), resultSizes), [](auto it) {
          return SmallVector<Value>{std::get<0>(it), std::get<1>(it)};
        });
    SmallVector<ValueRange> replacements = llvm::map_to_vector(
        replacementsVec, [](ArrayRef<Value> v) -> ValueRange { return v; });
    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};

struct ConvertFuncOp : public OpConversionPattern<IREE::Flow::FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto convertType = [&](Type type) -> Type {
      if (llvm::isa<TensorType>(type)) {
        // Tensors become resources without sizes. The default type converter
        // adds the size so we bypass that here. We may want to allow the user
        // to override the lifetime with attributes, too.
        return rewriter.getType<IREE::Stream::ResourceType>(
            IREE::Stream::Lifetime::Unknown);
      }
      return getTypeConverter()->convertType(type);
    };
    auto newArgTypes = llvm::map_to_vector(op.getArgumentTypes(), convertType);
    auto newResultTypes = llvm::map_to_vector(op.getResultTypes(), convertType);
    auto newType = FunctionType::get(getContext(), newArgTypes, newResultTypes);
    SmallVector<DictionaryAttr> argAttrs;
    if (auto argAttrsAttr = adaptor.getArgAttrsAttr()) {
      llvm::append_range(argAttrs, argAttrsAttr.getAsRange<DictionaryAttr>());
    }
    SmallVector<DictionaryAttr> resultAttrs;
    if (auto resAttrsAttr = adaptor.getResAttrsAttr()) {
      llvm::append_range(resultAttrs,
                         resAttrsAttr.getAsRange<DictionaryAttr>());
    }
    auto newOp = rewriter.replaceOpWithNewOp<IREE::Stream::AsyncFuncOp>(
        op, adaptor.getSymName(), newType, adaptor.getTiedOperandsAttr(),
        argAttrs, resultAttrs);
    newOp->setDialectAttrs(op->getDialectAttrs());
    return success();
  }
};

struct ConvertCallOp : public AffinityOpConversionPattern<IREE::Flow::CallOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      IREE::Flow::CallOp op, OneToNOpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    // Zero is going to be used for each operand to start.
    auto zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);

    // Query and resolve all operands and their sizes.
    SmallVector<Value> callOperands;
    SmallVector<Value> callOperandSizes;
    SmallVector<Value> callOperandOffsets;
    SmallVector<Value> callOperandEnds;
    SmallVector<Value> callOperandLengths;
    SmallVector<Value> operandSizes;
    for (auto [oldOperand, convertedOperand] :
         llvm::zip_equal(op.getArguments(), adaptor.getArguments())) {
      Value newOperand;
      if (llvm::isa<ShapedType>(oldOperand.getType())) {
        auto newOperandCast =
            transferTensorOperands(op.getLoc(), oldOperand, convertedOperand,
                                   executionAffinityAttr, rewriter);
        newOperand = newOperandCast.resource;
        callOperandSizes.push_back(newOperandCast.resourceSize);
        operandSizes.push_back(newOperandCast.resourceSize);
        callOperandOffsets.push_back(zeroOffset);
        callOperandEnds.push_back(newOperandCast.resourceSize);
        callOperandLengths.push_back(newOperandCast.resourceSize);
      } else {
        newOperand = convertedOperand.front();
        operandSizes.push_back({});
      }
      callOperands.push_back(newOperand);
    }

    // Construct result sizes or reuse tied operand sizes from above.
    SmallVector<Value> resultSizes;
    SmallVector<Type> resultTypes;
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto tiedOperandBase = op.getTiedOperandsIndexAndLength().first;
    for (auto result : llvm::enumerate(op.getResults())) {
      auto oldResultType = result.value().getType();
      if (!llvm::isa<ShapedType>(oldResultType)) {
        resultTypes.push_back(getTypeConverter()->convertType(oldResultType));
        resultSizes.push_back(nullptr);
        continue;
      }
      auto tiedOperand = op.getTiedResultOperandIndex(result.index());
      if (tiedOperand.has_value()) {
        auto operandIndex = tiedOperand.value() - tiedOperandBase;
        resultSizes.push_back(operandSizes[operandIndex]);
        resultTypes.push_back(callOperands[operandIndex].getType());
      } else {
        auto resultDynamicDims = IREE::Util::buildDynamicDimsForValue(
            op.getLoc(), result.value(), rewriter);
        resultSizes.push_back(
            buildResultSizeOf(op.getLoc(), result.value(), resultDynamicDims,
                              executionAffinityAttr, rewriter));
        resultTypes.push_back(unknownType);
      }
    }

    auto newOp = rewriter.create<IREE::Stream::AsyncCallOp>(
        op.getLoc(), resultTypes, adaptor.getCalleeAttr(), callOperands,
        callOperandSizes, callOperandOffsets, callOperandEnds,
        callOperandLengths, resultSizes, adaptor.getTiedOperandsAttr(),
        op.getArgAttrsAttr(), op.getResAttrsAttr(), executionAffinityAttr);
    newOp->setDialectAttrs(op->getDialectAttrs());
    replaceOpWithMultiple(op, newOp->getResults(), resultSizes, rewriter);
    return success();
  }
};

// Inserts a stream.binding.subspan op for |arg|.
// If the tensor result is statically shaped then the binding is inserted at the
// current |builder| location. If the result is dynamically shaped then the
// insertion point is set to the first use of the |arg| where all required
// dynamic dimension SSA values are present.
static bool insertBindingOp(BlockArgument arg,
                            IREE::Flow::DispatchTensorType tensorType,
                            Value zero, OpBuilder &builder) {
  // No uses: don't need a binding op.
  if (arg.use_empty())
    return true;

  // Find the dynamic dimension SSA values of the argument within the region.
  // If the flow dialect properly modeled dimension associations we wouldn't
  // need this.
  SmallVector<Value> dynamicDims;
  OpBuilder::InsertPoint ip;
  if (!tensorType.hasStaticShape()) {
    // Try first to find a flow.dispatch.tie_shape op. All args with dynamic
    // shapes should have one. We don't need to perform any analysis and don't
    // need to worry about insertion order if we do our work next to it as it
    // already carries all the SSA values we need.
    IREE::Flow::DispatchTieShapeOp tieShapeOp;
    for (auto user : arg.getUsers()) {
      tieShapeOp = dyn_cast<IREE::Flow::DispatchTieShapeOp>(user);
      if (tieShapeOp)
        break;
    }
    if (tieShapeOp) {
      // Found a tie shape op - we'll insert ourselves there.
      ip = builder.saveInsertionPoint();
      builder.setInsertionPoint(tieShapeOp);
      dynamicDims = tieShapeOp.getDynamicDims();
    } else {
      // The issue here is that at this point we no longer have the information
      // we need to reconstitute the dimensions. We expect that whoever created
      // the executable captured the shape dimensions they needed and we can
      // find them with the simple logic above. If we do hit this case we'll
      // need to add a new dispatch argument for the dimension and then go to
      // each dispatch site and insert the dimension query - if that feels like
      // a nasty hack it's because it is and it'd be better if we could avoid
      // needing it.
      mlir::emitError(arg.getLoc())
          << "dynamic dispatch dimensions not properly captured; the must be "
             "associated with a flow.dispatch.tie_shape op";
      return false;
    }
  }

  auto subspanOp = builder.create<IREE::Stream::BindingSubspanOp>(
      arg.getLoc(), tensorType, arg, zero, dynamicDims);
  arg.replaceAllUsesExcept(subspanOp.getResult(), subspanOp);

  // If we needed to insert at a special point restore back to the original
  // insertion point to keep the ops ordered with arguments.
  if (ip.isSet()) {
    builder.restoreInsertionPoint(ip);
  }

  return true;
}

// Replaces flow.return ops with stream.return ops.
// We do this outside of conversion to bypass dynamic recursive legality checks.
// If we remove the recursive legality check - which requires not passing
// through any flow ops inside of the executable - we'll be able to get rid of
// this.
static void convertReturnOps(Region &region) {
  region.walk([](IREE::Flow::ReturnOp oldOp) {
    OpBuilder(oldOp).create<IREE::Stream::ReturnOp>(oldOp.getLoc(),
                                                    oldOp.getOperands());
    oldOp.erase();
  });
}

template <typename FlowOpT, typename StreamOpT>
static void replaceDispatchWorkgroupInfoOp(FlowOpT op,
                                           PatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<StreamOpT>(op, op.getResult().getType(),
                                         op.getDimension());
}

template <typename FlowOpT, typename StreamOpT>
struct ConvertDispatchWorkgroupInfoOp : public OpConversionPattern<FlowOpT> {
  using OpConversionPattern<FlowOpT>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FlowOpT op, typename FlowOpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<StreamOpT>(op, op.getResult().getType(),
                                           adaptor.getDimension());
    return success();
  }
};

struct ConvertExecutableOp
    : public OpConversionPattern<IREE::Flow::ExecutableOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::ExecutableOp flowOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // flow.executable -> stream.executable
    auto streamOp = rewriter.create<IREE::Stream::ExecutableOp>(
        flowOp.getLoc(), flowOp.getSymName());
    streamOp.setVisibility(flowOp.getVisibility());
    streamOp->setDialectAttrs(flowOp->getDialectAttrs());
    rewriter.setInsertionPointToStart(&streamOp.getBody().front());

    // flow.executable.export -> stream.executable.export
    for (auto exportOp : flowOp.getOps<IREE::Flow::ExecutableExportOp>()) {
      auto newOp = rewriter.create<IREE::Stream::ExecutableExportOp>(
          exportOp.getLoc(), exportOp.getSymName(),
          exportOp.getFunctionRefAttr());
      newOp->setDialectAttrs(exportOp->getDialectAttrs());
      if (!exportOp.getWorkgroupCount().empty()) {
        mlir::IRMapping mapper;
        exportOp.getWorkgroupCount().cloneInto(&newOp.getWorkgroupCount(),
                                               mapper);
        convertReturnOps(newOp.getWorkgroupCount());
      }
    }

    // Move the original nested module body into the new executable directly.
    if (auto innerModuleOp = flowOp.getInnerModule()) {
      auto moduleOp = rewriter.cloneWithoutRegions(innerModuleOp);
      streamOp.getInnerModule().getBodyRegion().takeBody(
          flowOp.getInnerModule().getBodyRegion());

      // Update the entry point signatures in the module.
      // Dispatch tensor arguments become bindings and all others are preserved
      // as adaptor. Note that we only touch public (exported) functions.
      for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
        if (!funcOp.isPublic())
          continue;

        SmallVector<Type> newTypes;
        newTypes.reserve(funcOp.getNumArguments());
        assert(funcOp.getNumResults() == 0 &&
               "flow dispatches have no results");

        rewriter.setInsertionPointToStart(&funcOp.front());
        auto zero = rewriter.create<arith::ConstantIndexOp>(funcOp.getLoc(), 0);
        for (auto arg : funcOp.front().getArguments()) {
          auto oldType = arg.getType();
          if (auto tensorType =
                  llvm::dyn_cast<IREE::Flow::DispatchTensorType>(oldType)) {
            // Now a binding - insert the stream.binding.subspan op to slice it.
            auto newType = rewriter.getType<IREE::Stream::BindingType>();
            newTypes.push_back(newType);
            if (!insertBindingOp(arg, tensorType, zero, rewriter)) {
              return rewriter.notifyMatchFailure(
                  flowOp, "failed to query dynamic dimensions");
            }
            arg.setType(newType);
          } else {
            // Preserved - will eventually be a push constants.
            newTypes.push_back(oldType);
          }
        }

        // Strip any shape ties now that we've extracted the information.
        funcOp.walk([&](IREE::Flow::DispatchTieShapeOp tieOp) {
          rewriter.replaceOp(tieOp, tieOp.getOperand());
        });

        funcOp.setType(rewriter.getFunctionType(newTypes, {}));
      }

      // Walk the module and replace some ops that we don't rely on the pattern
      // rewriter for today. This is pretty shady and a side-effect of
      // recursively marking the stream executable contents as legal - if we
      // didn't do that (and converted all flow ops) we could drop this logic
      // and rely only the patterns.
      moduleOp.walk([&](Operation *op) {
        TypeSwitch<Operation *>(op)
            .Case<IREE::Flow::DispatchWorkgroupIDOp>([&](auto op) {
              replaceDispatchWorkgroupInfoOp<
                  IREE::Flow::DispatchWorkgroupIDOp,
                  IREE::Stream::DispatchWorkgroupIDOp>(op, rewriter);
            })
            .Case<IREE::Flow::DispatchWorkgroupCountOp>([&](auto op) {
              replaceDispatchWorkgroupInfoOp<
                  IREE::Flow::DispatchWorkgroupCountOp,
                  IREE::Stream::DispatchWorkgroupCountOp>(op, rewriter);
            })
            .Case<IREE::Flow::DispatchWorkgroupSizeOp>([&](auto op) {
              replaceDispatchWorkgroupInfoOp<
                  IREE::Flow::DispatchWorkgroupSizeOp,
                  IREE::Stream::DispatchWorkgroupSizeOp>(op, rewriter);
            })
            .Default([&](auto *op) {});
      });
    }

    rewriter.eraseOp(flowOp);
    return success();
  }
};

struct ConvertReturnOp : public OpConversionPattern<IREE::Flow::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::Stream::ReturnOp>(op,
                                                        adaptor.getOperands());
    return success();
  }
};

} // namespace

void populateFlowToStreamConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    IREE::Stream::AffinityAnalysis *affinityAnalysis,
    RewritePatternSet &patterns) {
  patterns.insert<
      ConvertTensorConstantOp, ConvertTensorDynamicConstantOp,
      ConvertTensorCastLikeOp<IREE::Flow::TensorReshapeOp>,
      ConvertTensorCastLikeOp<IREE::Flow::TensorBitCastOp>,
      ConvertTensorAllocaOp, ConvertTensorEmptyOp, ConvertTensorSplatOp,
      ConvertTensorCloneOp, ConvertTensorBarrierOp, ConvertTensorTransferOp,
      ConvertTensorSliceOp, ConvertTensorUpdateOp, ConvertTensorLoadOp,
      ConvertTensorStoreOp, ConvertTensorTraceOp>(typeConverter, context,
                                                  affinityAnalysis);
  patterns.insert<ConvertChannelDefaultOp>(typeConverter, context,
                                           affinityAnalysis);
  patterns.insert<ConvertChannelSplitOp, ConvertChannelRankOp,
                  ConvertChannelCountOp>(typeConverter, context);
  patterns
      .insert<ConvertAllGatherOp, ConvertAllReduceOp, ConvertReduceScatterOp,
              ConvertAllToAllOp, ConvertCollectiveSendRecvOp>(
          typeConverter, context, affinityAnalysis);
  patterns.insert<ConvertDispatchOp>(typeConverter, context, affinityAnalysis);
  patterns.insert<ConvertFuncOp>(typeConverter, context);
  patterns.insert<ConvertCallOp>(typeConverter, context, affinityAnalysis);
  patterns.insert<ConvertExecutableOp>(typeConverter, context);
  patterns.insert<
      ConvertDispatchWorkgroupInfoOp<IREE::Flow::DispatchWorkgroupIDOp,
                                     IREE::Stream::DispatchWorkgroupIDOp>,
      ConvertDispatchWorkgroupInfoOp<IREE::Flow::DispatchWorkgroupCountOp,
                                     IREE::Stream::DispatchWorkgroupCountOp>,
      ConvertDispatchWorkgroupInfoOp<IREE::Flow::DispatchWorkgroupSizeOp,
                                     IREE::Stream::DispatchWorkgroupSizeOp>>(
      typeConverter, context);
  patterns.insert<ConvertReturnOp>(typeConverter, context);
}

void populateFlowToStreamConversionPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter,
    IREE::Stream::AffinityAnalysis *affinityAnalysis,
    RewritePatternSet &patterns) {
  // Disallow all flow ops besides the ones we pass through (today).
  // We don't have a stream-equivalent of several of the dispatch-level flow
  // ops as the codegen backends directly touch them and so long as we have both
  // paths we can't cut over. Once we convert the flow.executable to a
  // stream.executable we ignore the contents and cross our fingers.
  conversionTarget.addIllegalDialect<IREE::Flow::FlowDialect>();
  conversionTarget.addLegalOp<IREE::Stream::ExecutableOp>();
  conversionTarget.markOpRecursivelyLegal<IREE::Stream::ExecutableOp>();

  populateFlowToStreamConversionPatterns(context, typeConverter,
                                         affinityAnalysis, patterns);
}

} // namespace mlir::iree_compiler
