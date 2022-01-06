// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/FlowToStream/ConvertFlowToStream.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Stream/Conversion/PatternUtils.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace iree_compiler {

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
      dynamicDims, /*affinity=*/nullptr);
}

// Reshapes become clones here to preserve shape information (which may become
// actual transfers depending on source/target shape) - they'll be elided if not
// needed.
struct ConvertTensorReshapeOp
    : public OpConversionPattern<IREE::Flow::TensorReshapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::TensorReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto source = consumeTensorOperand(op.getLoc(), adaptor.source(), rewriter);
    auto resultSize =
        buildResultSizeOf(op.getLoc(), op.result(), op.result_dims(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorCloneOp>(
        op, unknownType, source.resource, op.source().getType(),
        op.source_dims(), source.resourceSize, op.result().getType(),
        adaptor.result_dims(), resultSize,
        /*affinity=*/nullptr);
    return success();
  }
};

struct ConvertTensorSplatOp
    : public OpConversionPattern<IREE::Flow::TensorSplatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::TensorSplatOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto resultSize =
        buildResultSizeOf(op.getLoc(), op.result(), op.result_dims(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorSplatOp>(
        op, unknownType, adaptor.value(), op.result().getType(),
        adaptor.result_dims(), resultSize,
        /*affinity=*/nullptr);
    return success();
  }
};

struct ConvertTensorCloneOp
    : public OpConversionPattern<IREE::Flow::TensorCloneOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::TensorCloneOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto operand =
        consumeTensorOperand(op.getLoc(), adaptor.operand(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorCloneOp>(
        op, unknownType, operand.resource, op.operand().getType(),
        op.operand_dims(), operand.resourceSize, op.result().getType(),
        adaptor.operand_dims(), operand.resourceSize,
        /*affinity=*/nullptr);
    return success();
  }
};

struct ConvertTensorSliceOp
    : public OpConversionPattern<IREE::Flow::TensorSliceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::TensorSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto source = consumeTensorOperand(op.getLoc(), adaptor.source(), rewriter);
    auto resultSize =
        buildResultSizeOf(op.getLoc(), op.result(), op.result_dims(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorSliceOp>(
        op, unknownType, source.resource, op.source().getType(),
        op.source_dims(), source.resourceSize, adaptor.start_indices(),
        adaptor.lengths(), op.result().getType(), adaptor.result_dims(),
        resultSize,
        /*affinity=*/nullptr);
    return success();
  }
};

struct ConvertTensorUpdateOp
    : public OpConversionPattern<IREE::Flow::TensorUpdateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::TensorUpdateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto update = consumeTensorOperand(op.getLoc(), adaptor.update(), rewriter);
    auto target = consumeTensorOperand(op.getLoc(), adaptor.target(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorUpdateOp>(
        op, target.resource.getType(), target.resource, op.target().getType(),
        adaptor.target_dims(), target.resourceSize, adaptor.start_indices(),
        update.resource, op.update().getType(), op.update_dims(),
        update.resourceSize,
        /*affinity=*/nullptr);
    return success();
  }
};

struct ConvertTensorLoadOp
    : public OpConversionPattern<IREE::Flow::TensorLoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::TensorLoadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = getTypeConverter()->convertType(op.result().getType());
    auto source = consumeTensorOperand(op.getLoc(), adaptor.source(), rewriter);

    auto stagingType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::Staging);
    auto loadSource = source.resource;
    if (source.resource.getType() != stagingType) {
      loadSource = rewriter.createOrFold<IREE::Stream::AsyncTransferOp>(
          op.getLoc(), stagingType, source.resource, source.resourceSize,
          source.resourceSize,
          /*source_affinity=*/nullptr,
          /*result_affinity=*/nullptr);
    }

    rewriter.replaceOpWithNewOp<IREE::Stream::TensorLoadOp>(
        op, resultType, loadSource, op.source().getType(), op.source_dims(),
        source.resourceSize, adaptor.indices());
    return success();
  }
};

struct ConvertTensorStoreOp
    : public OpConversionPattern<IREE::Flow::TensorStoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::TensorStoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto target = consumeTensorOperand(op.getLoc(), adaptor.target(), rewriter);

    auto stagingType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::Staging);
    auto storeTarget = target.resource;
    if (target.resource.getType() != stagingType) {
      storeTarget = rewriter.createOrFold<IREE::Stream::AsyncTransferOp>(
          op.getLoc(), stagingType, storeTarget, target.resourceSize,
          target.resourceSize,
          /*source_affinity=*/nullptr,
          /*result_affinity=*/nullptr);
    }

    auto newOp = rewriter.create<IREE::Stream::TensorStoreOp>(
        op.getLoc(), storeTarget.getType(), storeTarget, op.target().getType(),
        adaptor.target_dims(), target.resourceSize, adaptor.indices(),
        adaptor.value());

    Value newResult = newOp.result();
    if (target.resource.getType() != stagingType) {
      newResult = rewriter.createOrFold<IREE::Stream::AsyncTransferOp>(
          op.getLoc(), target.resource.getType(), newResult,
          target.resourceSize, target.resourceSize,
          /*source_affinity=*/nullptr,
          /*result_affinity=*/nullptr);
    }
    rewriter.replaceOp(op, {newResult});

    return success();
  }
};

struct ConvertTensorTraceOp
    : public OpConversionPattern<IREE::Flow::TensorTraceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::TensorTraceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> exportedTensors;
    for (auto it : llvm::zip(op.operands(), adaptor.operands())) {
      auto tensorOperand = std::get<0>(it);
      auto resourceOperand = std::get<1>(it);
      auto source =
          consumeTensorOperand(op.getLoc(), resourceOperand, rewriter);
      auto externalType = rewriter.getType<IREE::Stream::ResourceType>(
          IREE::Stream::Lifetime::External);
      auto exportSource = resourceOperand;
      if (source.resource.getType() != externalType) {
        exportSource = rewriter.create<IREE::Stream::AsyncTransferOp>(
            op.getLoc(), externalType, source.resource, source.resourceSize,
            source.resourceSize,
            /*source_affinity=*/nullptr,
            /*result_affinity=*/nullptr);
      }
      auto dynamicDims = IREE::Util::buildDynamicDimsForValue(
          op.getLoc(), tensorOperand, rewriter);
      exportedTensors.push_back(rewriter.create<IREE::Stream::TensorExportOp>(
          op.getLoc(), tensorOperand.getType(), exportSource,
          TypeAttr::get(tensorOperand.getType()), dynamicDims,
          source.resourceSize,
          /*affinity=*/nullptr));
    }
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorTraceOp>(op, adaptor.key(),
                                                             exportedTensors);
    return success();
  }
};

struct ConvertDispatchOp : public OpConversionPattern<IREE::Flow::DispatchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::DispatchOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Query and resolve all operands and their sizes.
    SmallVector<Value> dispatchOperands;
    SmallVector<Value> dispatchOperandSizes;
    SmallVector<Value> operandSizes;
    for (auto oldNewOperand : llvm::zip(op.operands(), adaptor.operands())) {
      auto oldOperand = std::get<0>(oldNewOperand);
      auto newOperand = std::get<1>(oldNewOperand);
      if (oldOperand.getType().isa<ShapedType>()) {
        auto newOperandCast =
            consumeTensorOperand(op.getLoc(), newOperand, rewriter);
        newOperand = newOperandCast.resource;
        dispatchOperandSizes.push_back(newOperandCast.resourceSize);
        operandSizes.push_back(newOperandCast.resourceSize);
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
    for (auto result : llvm::enumerate(op.results())) {
      auto oldResultType = result.value().getType();
      if (!oldResultType.isa<ShapedType>()) {
        resultTypes.push_back(getTypeConverter()->convertType(oldResultType));
        continue;
      }
      auto tiedOperand = op.getTiedResultOperandIndex(result.index());
      if (tiedOperand.hasValue()) {
        auto operandIndex = tiedOperand.getValue() - tiedOperandBase;
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

    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncDispatchOp>(
        op, resultTypes, adaptor.workgroup_count(), adaptor.entry_point(),
        dispatchOperands, dispatchOperandSizes, resultSizes,
        adaptor.tied_operandsAttr(),
        /*affinity=*/nullptr);
    return success();
  }
};

static SmallVector<Value> makeBindingDynamicDims(
    Location loc, IREE::Flow::DispatchTensorType tensorType, BlockArgument arg,
    OpBuilder &builder) {
  assert(!arg.use_empty() && "must have uses to query dims");
  if (tensorType.hasStaticShape()) return {};

  // 95% of the time the dims come right from the arguments so look there first.
  auto foundDims = IREE::Util::findDynamicDims(arg, builder.getInsertionBlock(),
                                               builder.getInsertionPoint());
  if (foundDims.hasValue()) return llvm::to_vector<4>(foundDims.getValue());

  // The issue here is that at this point we no longer have the information we
  // need to reconstitute the dimensions. We expect that whoever created the
  // executable captured the shape dimensions they needed and we can find them
  // with the simple logic above. If we do hit this case we'll need to add a new
  // dispatch argument for the dimension and then go to each dispatch site and
  // insert the dimension query - if that feels like a nasty hack it's because
  // it is and it'd be better if we could avoid needing it.
  llvm_unreachable("TODO: synthesize dynamic dimensions/hoist logic");
  return {};
}

struct ConvertExecutableOp
    : public OpConversionPattern<IREE::Flow::ExecutableOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::ExecutableOp flowOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // flow.executable -> stream.executable
    auto streamOp = rewriter.create<IREE::Stream::ExecutableOp>(
        flowOp.getLoc(), flowOp.sym_name());
    streamOp.setVisibility(flowOp.getVisibility());
    streamOp->setDialectAttrs(flowOp->getDialectAttrs());
    rewriter.setInsertionPointToStart(&streamOp.body().front());

    // flow.dispatch.entry -> stream.executable.entry_point
    for (auto entryOp : flowOp.getOps<IREE::Flow::DispatchEntryOp>()) {
      auto newOp = rewriter.create<IREE::Stream::ExecutableExportOp>(
          entryOp.getLoc(), entryOp.sym_name(), entryOp.function_refAttr());
      newOp->setDialectAttrs(entryOp->getDialectAttrs());
    }

    // Move the original nested module body into the new executable directly.
    auto moduleOp = rewriter.cloneWithoutRegions(flowOp.getInnerModule());
    streamOp.getInnerModule().body().takeBody(flowOp.getInnerModule().body());

    // Update the entry point signatures in the module.
    // Dispatch tensor arguments become bindings and all others are preserved as
    // adaptor. Note that we only touch public (exported) functions.
    for (auto funcOp : moduleOp.getOps<mlir::FuncOp>()) {
      if (!funcOp.isPublic()) continue;

      SmallVector<Type> newTypes;
      newTypes.reserve(funcOp.getNumArguments());
      assert(funcOp.getNumResults() == 0 && "flow dispatches have no results");

      rewriter.setInsertionPointToStart(&funcOp.front());
      auto zero = rewriter.create<arith::ConstantIndexOp>(funcOp.getLoc(), 0);
      for (auto arg : funcOp.front().getArguments()) {
        auto oldType = arg.getType();
        if (auto tensorType =
                oldType.dyn_cast<IREE::Flow::DispatchTensorType>()) {
          // Now a binding.
          auto newType = rewriter.getType<IREE::Stream::BindingType>();
          newTypes.push_back(newType);
          if (!arg.use_empty()) {
            auto dynamicDims =
                makeBindingDynamicDims(arg.getLoc(), tensorType, arg, rewriter);
            auto subspanOp = rewriter.create<IREE::Stream::BindingSubspanOp>(
                arg.getLoc(), tensorType, arg, zero, dynamicDims);
            arg.replaceAllUsesExcept(subspanOp.result(), subspanOp);
          }
          arg.setType(newType);
        } else {
          // Preserved - will eventually be a push constants.
          newTypes.push_back(oldType);
        }
      }

      funcOp.setType(rewriter.getFunctionType(newTypes, {}));
    }

    rewriter.replaceOp(flowOp, {});
    return success();
  }
};

}  // namespace

void populateFlowToStreamConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    OwningRewritePatternList &patterns) {
  patterns
      .insert<ConvertTensorReshapeOp, ConvertTensorSplatOp,
              ConvertTensorCloneOp, ConvertTensorSliceOp, ConvertTensorUpdateOp,
              ConvertTensorLoadOp, ConvertTensorStoreOp, ConvertTensorTraceOp>(
          typeConverter, context);
  patterns.insert<ConvertDispatchOp>(typeConverter, context);
  patterns.insert<ConvertExecutableOp>(typeConverter, context);
}

void populateFlowToStreamConversionPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter, OwningRewritePatternList &patterns) {
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

}  // namespace iree_compiler
}  // namespace mlir
