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
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

namespace {

// Inserts a sizeof calculation for the given tensor value type and dims.
// This should only be used to produce sizes for values produced by an op; the
// size of operands must be queried from the input resource.
static Value buildResultSizeOf(Location loc, Value tensorValue,
                               ValueRange dynamicDims,
                               ConversionPatternRewriter &rewriter) {
  // TODO(benvanik): see if we can stash this on the side to avoid expensive
  // materialization of a bunch of redundant IR.
  return rewriter.createOrFold<IREE::Stream::TensorSizeOfOp>(
      loc, rewriter.getIndexType(), TypeAttr::get(tensorValue.getType()),
      dynamicDims,
      IREE::Stream::AffinityAttr::lookup(tensorValue.getDefiningOp()));
}

// Reshapes and bitcasts become clones here to preserve shape/element type
// information (which may become actual transfers depending on source/target
// shape) - they'll be elided if not needed.
template <typename CastOpTy>
struct ConvertTensorCastLikeOp : public OpConversionPattern<CastOpTy> {
  using OpConversionPattern<CastOpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CastOpTy op, typename CastOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto source =
        consumeTensorOperand(op.getLoc(), adaptor.getSource(), rewriter);
    auto resultSize = buildResultSizeOf(op.getLoc(), op.getResult(),
                                        op.getResultDims(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorCloneOp>(
        op, unknownType, source.resource, op.getSource().getType(),
        op.getSourceDims(), source.resourceSize, op.getResult().getType(),
        adaptor.getResultDims(), resultSize,
        IREE::Stream::AffinityAttr::lookup(op));
    return success();
  }
};

struct ConvertTensorAllocaOp
    : public OpConversionPattern<IREE::Flow::TensorAllocaOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::TensorAllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type unknownType = IREE::Stream::ResourceType::get(getContext());
    auto resultSize = buildResultSizeOf(op.getLoc(), op.getResult(),
                                        op.getResultDims(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncAllocaOp>(
        op, unknownType, resultSize, IREE::Stream::AffinityAttr::lookup(op));
    return success();
  }
};

struct ConvertTensorEmptyOp
    : public OpConversionPattern<IREE::Flow::TensorEmptyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::TensorEmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type unknownType = IREE::Stream::ResourceType::get(getContext());
    auto resultSize = buildResultSizeOf(op.getLoc(), op.getResult(),
                                        op.getResultDims(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorEmptyOp>(
        op, unknownType, op.getResult().getType(), adaptor.getResultDims(),
        resultSize, IREE::Stream::AffinityAttr::lookup(op));
    return success();
  }
};

struct ConvertTensorSplatOp
    : public OpConversionPattern<IREE::Flow::TensorSplatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::TensorSplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto resultSize = buildResultSizeOf(op.getLoc(), op.getResult(),
                                        op.getResultDims(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorSplatOp>(
        op, unknownType, adaptor.getValue(), op.getResult().getType(),
        adaptor.getResultDims(), resultSize,
        IREE::Stream::AffinityAttr::lookup(op));
    return success();
  }
};

struct ConvertTensorCloneOp
    : public OpConversionPattern<IREE::Flow::TensorCloneOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::TensorCloneOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto operand =
        consumeTensorOperand(op.getLoc(), adaptor.getOperand(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorCloneOp>(
        op, unknownType, operand.resource, op.getOperand().getType(),
        op.getArgumentDims(), operand.resourceSize, op.getResult().getType(),
        adaptor.getArgumentDims(), operand.resourceSize,
        IREE::Stream::AffinityAttr::lookup(op));
    return success();
  }
};

struct ConvertTensorSliceOp
    : public OpConversionPattern<IREE::Flow::TensorSliceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::TensorSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto source =
        consumeTensorOperand(op.getLoc(), adaptor.getSource(), rewriter);
    auto resultSize = buildResultSizeOf(op.getLoc(), op.getResult(),
                                        op.getResultDims(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorSliceOp>(
        op, unknownType, source.resource, op.getSource().getType(),
        op.getSourceDims(), source.resourceSize, adaptor.getStartIndices(),
        adaptor.getLengths(), op.getResult().getType(), adaptor.getResultDims(),
        resultSize, IREE::Stream::AffinityAttr::lookup(op));
    return success();
  }
};

struct ConvertTensorUpdateOp
    : public OpConversionPattern<IREE::Flow::TensorUpdateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::TensorUpdateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto update =
        consumeTensorOperand(op.getLoc(), adaptor.getUpdate(), rewriter);
    auto target =
        consumeTensorOperand(op.getLoc(), adaptor.getTarget(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorUpdateOp>(
        op, target.resource.getType(), target.resource,
        op.getTarget().getType(), adaptor.getTargetDims(), target.resourceSize,
        adaptor.getStartIndices(), update.resource, op.getUpdate().getType(),
        op.getUpdateDims(), update.resourceSize,
        IREE::Stream::AffinityAttr::lookup(op));
    return success();
  }
};

struct ConvertTensorLoadOp
    : public OpConversionPattern<IREE::Flow::TensorLoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::TensorLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto source =
        consumeTensorOperand(op.getLoc(), adaptor.getSource(), rewriter);

    auto stagingType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::Staging);
    auto loadSource = source.resource;
    if (source.resource.getType() != stagingType) {
      loadSource = rewriter.createOrFold<IREE::Stream::AsyncTransferOp>(
          op.getLoc(), stagingType, source.resource, source.resourceSize,
          source.resourceSize,
          /*source_affinity=*/IREE::Stream::AffinityAttr::lookup(op),
          /*result_affinity=*/nullptr);
    }

    rewriter.replaceOpWithNewOp<IREE::Stream::TensorLoadOp>(
        op, resultType, loadSource, op.getSource().getType(),
        op.getSourceDims(), source.resourceSize, adaptor.getIndices());
    return success();
  }
};

struct ConvertTensorStoreOp
    : public OpConversionPattern<IREE::Flow::TensorStoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::TensorStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto target =
        consumeTensorOperand(op.getLoc(), adaptor.getTarget(), rewriter);

    auto stagingType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::Staging);
    auto storeTarget = target.resource;
    if (target.resource.getType() != stagingType) {
      storeTarget = rewriter.createOrFold<IREE::Stream::AsyncTransferOp>(
          op.getLoc(), stagingType, storeTarget, target.resourceSize,
          target.resourceSize,
          /*source_affinity=*/IREE::Stream::AffinityAttr::lookup(op),
          /*result_affinity=*/nullptr);
    }

    auto newOp = rewriter.create<IREE::Stream::TensorStoreOp>(
        op.getLoc(), storeTarget.getType(), storeTarget,
        op.getTarget().getType(), adaptor.getTargetDims(), target.resourceSize,
        adaptor.getIndices(), adaptor.getValue());

    Value newResult = newOp.getResult();
    if (target.resource.getType() != stagingType) {
      newResult = rewriter.createOrFold<IREE::Stream::AsyncTransferOp>(
          op.getLoc(), target.resource.getType(), newResult,
          target.resourceSize, target.resourceSize,
          /*source_affinity=*/nullptr,
          /*result_affinity=*/IREE::Stream::AffinityAttr::lookup(op));
    }
    rewriter.replaceOp(op, {newResult});

    return success();
  }
};

struct ConvertTensorTraceOp
    : public OpConversionPattern<IREE::Flow::TensorTraceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::TensorTraceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> resources;
    SmallVector<Value> resourceSizes;
    SmallVector<Attribute> resourceEncodings;
    for (auto [tensorOperand, resourceOperand] :
         llvm::zip_equal(op.getValues(), adaptor.getValues())) {
      auto source =
          consumeTensorOperand(op.getLoc(), resourceOperand, rewriter);
      auto stagingType = rewriter.getType<IREE::Stream::ResourceType>(
          IREE::Stream::Lifetime::Staging);
      auto traceSource = source.resource;
      if (source.resource.getType() != stagingType) {
        traceSource = rewriter.create<IREE::Stream::AsyncTransferOp>(
            op.getLoc(), stagingType, source.resource, source.resourceSize,
            source.resourceSize,
            /*source_affinity=*/IREE::Stream::AffinityAttr::lookup(op),
            /*result_affinity=*/nullptr);
      }
      resources.push_back(traceSource);
      resourceSizes.push_back(source.resourceSize);
      resourceEncodings.push_back(TypeAttr::get(tensorOperand.getType()));
    }
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorTraceOp>(
        op, adaptor.getKey(), resources, resourceSizes,
        rewriter.getArrayAttr(resourceEncodings), adaptor.getValueDims());
    return success();
  }
};

struct ConvertChannelDefaultOp
    : public OpConversionPattern<IREE::Flow::ChannelDefaultOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::ChannelDefaultOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::Stream::ChannelCreateOp>(
        op, /*id=*/Value{},
        /*group=*/adaptor.getGroupAttr(),
        /*rank=*/Value{},
        /*count=*/Value{}, IREE::Stream::AffinityAttr::lookup(op));
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
    : public OpConversionPattern<IREE::Flow::CollectiveAllGatherOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::CollectiveAllGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto shape = llvm::cast<ShapedType>(op.getSource().getType());
    auto collectiveAttr = IREE::Stream::CollectiveAttr::get(
        op.getContext(), IREE::Stream::CollectiveKind::AllGather,
        /*reduction=*/std::nullopt,
        static_cast<IREE::Stream::CollectiveElementType>(op.getElementType()));

    auto zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto elementCount = rewriter.create<arith::ConstantIndexOp>(
        op.getLoc(), shape.getNumElements());
    auto newTargetCast =
        consumeTensorOperand(op.getLoc(), adaptor.getTarget(), rewriter);
    auto newSourceCast =
        consumeTensorOperand(op.getLoc(), adaptor.getSource(), rewriter);

    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncCollectiveOp>(
        op, collectiveAttr, adaptor.getTarget(),
        /*target_size=*/newTargetCast.resourceSize,
        /*target_offset=*/zeroOffset,
        /*target_end=*/newTargetCast.resourceSize,
        /*target_length=*/newTargetCast.resourceSize, adaptor.getSource(),
        /*source_size=*/newSourceCast.resourceSize,
        /*source_offset=*/zeroOffset, /*source_end=*/newSourceCast.resourceSize,
        /*source_length=*/newSourceCast.resourceSize, elementCount,
        adaptor.getChannel(),
        /*param=*/mlir::Value(), IREE::Stream::AffinityAttr::lookup(op));
    return success();
  }
};

struct ConvertAllReduceOp
    : public OpConversionPattern<IREE::Flow::CollectiveAllReduceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::CollectiveAllReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto shape = llvm::cast<ShapedType>(op.getType());
    auto collectiveAttr = IREE::Stream::CollectiveAttr::get(
        op.getContext(), IREE::Stream::CollectiveKind::AllReduce,
        static_cast<IREE::Stream::CollectiveReductionOp>(op.getReductionOp()),
        static_cast<IREE::Stream::CollectiveElementType>(op.getElementType()));

    auto zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto elementCount = rewriter.create<arith::ConstantIndexOp>(
        op.getLoc(), shape.getNumElements());
    auto newTargetCast =
        consumeTensorOperand(op.getLoc(), adaptor.getTarget(), rewriter);
    auto newSourceCast =
        consumeTensorOperand(op.getLoc(), adaptor.getSource(), rewriter);

    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncCollectiveOp>(
        op, collectiveAttr, adaptor.getTarget(),
        /*target_size=*/newTargetCast.resourceSize,
        /*target_offset=*/zeroOffset,
        /*target_end=*/newTargetCast.resourceSize,
        /*target_length=*/newTargetCast.resourceSize, adaptor.getSource(),
        /*source_size=*/newSourceCast.resourceSize,
        /*source_offset=*/zeroOffset, /*source_end=*/newSourceCast.resourceSize,
        /*source_length=*/newSourceCast.resourceSize, elementCount,
        adaptor.getChannel(),
        /*param=*/mlir::Value(), IREE::Stream::AffinityAttr::lookup(op));
    return success();
  }
};

struct ConvertAllToAllOp
    : public OpConversionPattern<IREE::Flow::CollectiveAllToAllOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::CollectiveAllToAllOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto shape = llvm::cast<ShapedType>(op.getSource().getType());
    auto collectiveAttr = IREE::Stream::CollectiveAttr::get(
        op.getContext(), IREE::Stream::CollectiveKind::AllToAll,
        /*reduction=*/std::nullopt,
        static_cast<IREE::Stream::CollectiveElementType>(op.getElementType()));

    auto zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto elementCount = rewriter.create<arith::ConstantIndexOp>(
        op.getLoc(), shape.getNumElements());
    auto newTargetCast =
        consumeTensorOperand(op.getLoc(), adaptor.getTarget(), rewriter);
    auto newSourceCast =
        consumeTensorOperand(op.getLoc(), adaptor.getSource(), rewriter);

    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncCollectiveOp>(
        op, collectiveAttr, adaptor.getTarget(),
        /*target_size=*/newTargetCast.resourceSize,
        /*target_offset=*/zeroOffset,
        /*target_end=*/newTargetCast.resourceSize,
        /*target_length=*/newTargetCast.resourceSize, adaptor.getSource(),
        /*source_size=*/newSourceCast.resourceSize,
        /*source_offset=*/zeroOffset, /*source_end=*/newSourceCast.resourceSize,
        /*source_length=*/newSourceCast.resourceSize, elementCount,
        adaptor.getChannel(),
        /*param=*/mlir::Value(), IREE::Stream::AffinityAttr::lookup(op));
    return success();
  }
};

struct ConvertReduceScatterOp
    : public OpConversionPattern<IREE::Flow::CollectiveReduceScatterOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::CollectiveReduceScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto shape = llvm::cast<ShapedType>(op.getType());
    auto collectiveAttr = IREE::Stream::CollectiveAttr::get(
        op.getContext(), IREE::Stream::CollectiveKind::ReduceScatter,
        static_cast<IREE::Stream::CollectiveReductionOp>(op.getReductionOp()),
        static_cast<IREE::Stream::CollectiveElementType>(op.getElementType()));

    auto zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto elementCount = rewriter.create<arith::ConstantIndexOp>(
        op.getLoc(), shape.getNumElements());
    auto newTargetCast =
        consumeTensorOperand(op.getLoc(), adaptor.getTarget(), rewriter);
    auto newSourceCast =
        consumeTensorOperand(op.getLoc(), adaptor.getSource(), rewriter);

    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncCollectiveOp>(
        op, collectiveAttr, adaptor.getTarget(),
        /*target_size=*/newTargetCast.resourceSize,
        /*target_offset=*/zeroOffset,
        /*target_end=*/newTargetCast.resourceSize,
        /*target_length=*/newTargetCast.resourceSize, adaptor.getSource(),
        /*source_size=*/newSourceCast.resourceSize,
        /*source_offset=*/zeroOffset, /*source_end=*/newSourceCast.resourceSize,
        /*source_length=*/newSourceCast.resourceSize, elementCount,
        adaptor.getChannel(),
        /*param=*/mlir::Value(), IREE::Stream::AffinityAttr::lookup(op));
    return success();
  }
};

struct ConvertCollectiveSendRecvOp
    : public OpConversionPattern<IREE::Flow::CollectiveSendRecvOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::CollectiveSendRecvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto shape = llvm::cast<ShapedType>(op.getType());
    auto collectiveAttr = IREE::Stream::CollectiveAttr::get(
        op.getContext(), IREE::Stream::CollectiveKind::SendRecv,
        /*reduction=*/std::nullopt,
        static_cast<IREE::Stream::CollectiveElementType>(op.getElementType()));

    auto zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto elementCount = rewriter.create<arith::ConstantIndexOp>(
        op.getLoc(), shape.getNumElements());
    auto newTargetCast =
        consumeTensorOperand(op.getLoc(), adaptor.getTarget(), rewriter);
    auto newSourceCast =
        consumeTensorOperand(op.getLoc(), adaptor.getSource(), rewriter);

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

    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncCollectiveOp>(
        op, collectiveAttr, adaptor.getTarget(),
        /*target_size=*/newTargetCast.resourceSize,
        /*target_offset=*/zeroOffset,
        /*target_end=*/newTargetCast.resourceSize,
        /*target_length=*/newTargetCast.resourceSize, adaptor.getSource(),
        /*source_size=*/newSourceCast.resourceSize,
        /*source_offset=*/zeroOffset, /*source_end=*/newSourceCast.resourceSize,
        /*source_length=*/newSourceCast.resourceSize, elementCount,
        adaptor.getChannel(),
        /*param=*/param, IREE::Stream::AffinityAttr::lookup(op));
    return success();
  }
};

struct ConvertDispatchOp : public OpConversionPattern<IREE::Flow::DispatchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::DispatchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Zero is going to be used for each operand to start.
    auto zeroOffset = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);

    // Query and resolve all operands and their sizes.
    SmallVector<Value> dispatchOperands;
    SmallVector<Value> dispatchOperandSizes;
    SmallVector<Value> dispatchOperandOffsets;
    SmallVector<Value> dispatchOperandEnds;
    SmallVector<Value> dispatchOperandLengths;
    SmallVector<Value> operandSizes;
    for (auto [oldOperand, newOperand] :
         llvm::zip_equal(op.getArguments(), adaptor.getArguments())) {
      if (llvm::isa<ShapedType>(oldOperand.getType())) {
        auto newOperandCast =
            consumeTensorOperand(op.getLoc(), newOperand, rewriter);
        newOperand = newOperandCast.resource;
        dispatchOperandSizes.push_back(newOperandCast.resourceSize);
        operandSizes.push_back(newOperandCast.resourceSize);
        dispatchOperandOffsets.push_back(zeroOffset);
        dispatchOperandEnds.push_back(newOperandCast.resourceSize);
        dispatchOperandLengths.push_back(newOperandCast.resourceSize);
      } else {
        operandSizes.push_back({});
      }
      dispatchOperands.push_back(newOperand);
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
        continue;
      }
      auto tiedOperand = op.getTiedResultOperandIndex(result.index());
      if (tiedOperand.has_value()) {
        auto operandIndex = tiedOperand.value() - tiedOperandBase;
        resultSizes.push_back(operandSizes[operandIndex]);
        resultTypes.push_back(dispatchOperands[operandIndex].getType());
      } else {
        auto resultDynamicDims = IREE::Util::buildDynamicDimsForValue(
            op.getLoc(), result.value(), rewriter);
        resultSizes.push_back(buildResultSizeOf(op.getLoc(), result.value(),
                                                resultDynamicDims, rewriter));
        resultTypes.push_back(unknownType);
      }
    }

    auto newOp = rewriter.replaceOpWithNewOp<IREE::Stream::AsyncDispatchOp>(
        op, resultTypes, adaptor.getWorkload(), adaptor.getEntryPointsAttr(),
        dispatchOperands, dispatchOperandSizes, dispatchOperandOffsets,
        dispatchOperandEnds, dispatchOperandLengths, resultSizes,
        adaptor.getTiedOperandsAttr(), IREE::Stream::AffinityAttr::lookup(op));
    newOp->setDialectAttrs(op->getDialectAttrs());
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
        return IREE::Stream::ResourceType::get(type.getContext(),
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

struct ConvertCallOp : public OpConversionPattern<IREE::Flow::CallOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(IREE::Flow::CallOp op, OpAdaptor adaptor,
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
    for (auto [oldOperand, newOperand] :
         llvm::zip_equal(op.getArguments(), adaptor.getArguments())) {
      if (llvm::isa<ShapedType>(oldOperand.getType())) {
        auto newOperandCast =
            consumeTensorOperand(op.getLoc(), newOperand, rewriter);
        newOperand = newOperandCast.resource;
        callOperandSizes.push_back(newOperandCast.resourceSize);
        operandSizes.push_back(newOperandCast.resourceSize);
        callOperandOffsets.push_back(zeroOffset);
        callOperandEnds.push_back(newOperandCast.resourceSize);
        callOperandLengths.push_back(newOperandCast.resourceSize);
      } else {
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
        resultSizes.push_back(buildResultSizeOf(op.getLoc(), result.value(),
                                                resultDynamicDims, rewriter));
        resultTypes.push_back(unknownType);
      }
    }

    auto newOp = rewriter.replaceOpWithNewOp<IREE::Stream::AsyncCallOp>(
        op, resultTypes, adaptor.getCalleeAttr(), callOperands,
        callOperandSizes, callOperandOffsets, callOperandEnds,
        callOperandLengths, resultSizes, adaptor.getTiedOperandsAttr(),
        IREE::Stream::AffinityAttr::lookup(op));
    newOp->setDialectAttrs(op->getDialectAttrs());
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

void populateFlowToStreamConversionPatterns(MLIRContext *context,
                                            TypeConverter &typeConverter,
                                            RewritePatternSet &patterns) {
  patterns
      .insert<ConvertTensorCastLikeOp<IREE::Flow::TensorReshapeOp>,
              ConvertTensorCastLikeOp<IREE::Flow::TensorBitCastOp>,
              ConvertTensorAllocaOp, ConvertTensorEmptyOp, ConvertTensorSplatOp,
              ConvertTensorCloneOp, ConvertTensorSliceOp, ConvertTensorUpdateOp,
              ConvertTensorLoadOp, ConvertTensorStoreOp, ConvertTensorTraceOp>(
          typeConverter, context);
  patterns.insert<ConvertChannelDefaultOp, ConvertChannelSplitOp,
                  ConvertChannelRankOp, ConvertChannelCountOp>(typeConverter,
                                                               context);
  patterns
      .insert<ConvertAllGatherOp, ConvertAllReduceOp, ConvertReduceScatterOp,
              ConvertAllToAllOp, ConvertCollectiveSendRecvOp>(typeConverter,
                                                              context);
  patterns.insert<ConvertDispatchOp>(typeConverter, context);
  patterns.insert<ConvertFuncOp, ConvertCallOp>(typeConverter, context);
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

void populateFlowToStreamConversionPatterns(MLIRContext *context,
                                            ConversionTarget &conversionTarget,
                                            TypeConverter &typeConverter,
                                            RewritePatternSet &patterns) {
  // Disallow all flow ops besides the ones we pass through (today).
  // We don't have a stream-equivalent of several of the dispatch-level flow
  // ops as the codegen backends directly touch them and so long as we have both
  // paths we can't cut over. Once we convert the flow.executable to a
  // stream.executable we ignore the contents and cross our fingers.
  conversionTarget.addIllegalDialect<IREE::Flow::FlowDialect>();
  conversionTarget.addLegalOp<IREE::Stream::ExecutableOp>();
  conversionTarget.markOpRecursivelyLegal<IREE::Stream::ExecutableOp>();

  populateFlowToStreamConversionPatterns(context, typeConverter, patterns);
}

} // namespace mlir::iree_compiler
