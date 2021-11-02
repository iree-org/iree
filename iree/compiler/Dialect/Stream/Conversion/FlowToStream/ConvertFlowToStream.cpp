// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/FlowToStream/ConvertFlowToStream.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Queries the size of the source tensor value.
// This may reuse existing IR or insert a new size calculation.
static Value querySizeOf(Location loc, Value streamValue, Value tensorValue,
                         ValueRange dynamicDims,
                         ConversionPatternRewriter &rewriter) {
  return rewriter.createOrFold<IREE::Stream::ResourceSizeOp>(
      loc, rewriter.getIndexType(), streamValue);
}

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

// hal.tensor.cast is inserted by frontends to ensure that ABI types are HAL
// buffer views. We need to map those to the stream import/export equivalents as
// the cast has special meaning when we are dealing with asynchronous values.
//
// %1 = hal.tensor.cast %0 : !hal.buffer_view -> tensor<4xf32>
// ->
// %1 = stream.tensor.import %0 : !hal.buffer_view ->
//                                tensor<4xf32> in !stream.resource<*>
struct ConvertHALTensorCastOp
    : public OpConversionPattern<IREE::HAL::TensorCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::HAL::TensorCastOp op, ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::HAL::TensorCastOpAdaptor operands(newOperands,
                                            op->getAttrDictionary());
    if (op.source().getType().isa<IREE::HAL::BufferViewType>()) {
      // Import (buffer view to stream resource).
      Type resultType = IREE::Stream::ResourceType::get(
          getContext(), IREE::Stream::Lifetime::External);
      auto resultSize = buildResultSizeOf(op.getLoc(), op.target(),
                                          operands.target_dims(), rewriter);
      auto newOp = rewriter.create<IREE::Stream::TensorImportOp>(
          op.getLoc(), resultType, operands.source(),
          TypeAttr::get(op.target().getType()), operands.target_dims(),
          resultSize,
          /*affinity=*/nullptr);

      auto unknownType = IREE::Stream::ResourceType::get(getContext());
      rewriter.replaceOpWithNewOp<IREE::Stream::AsyncTransferOp>(
          op, unknownType, newOp.result(), resultSize, resultSize,
          /*source_affinity=*/nullptr,
          /*result_affinity=*/nullptr);

    } else if (op.target().getType().isa<IREE::HAL::BufferViewType>()) {
      auto sourceSize = querySizeOf(op.getLoc(), operands.source(), op.source(),
                                    operands.source_dims(), rewriter);
      auto externalType = IREE::Stream::ResourceType::get(
          getContext(), IREE::Stream::Lifetime::External);
      Value exportSource = operands.source();
      if (operands.source().getType() != externalType) {
        exportSource = rewriter.create<IREE::Stream::AsyncTransferOp>(
            op.getLoc(), externalType, operands.source(), sourceSize,
            sourceSize,
            /*source_affinity=*/nullptr,
            /*result_affinity=*/nullptr);
      }

      // Export (stream resource to buffer view).
      rewriter.replaceOpWithNewOp<IREE::Stream::TensorExportOp>(
          op, op.target().getType(), exportSource,
          TypeAttr::get(op.source().getType()), operands.source_dims(),
          sourceSize,
          /*affinity=*/nullptr);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported HAL cast conversion");
    }
    return success();
  }
};

// Reshapes become clones here to preserve shape information (which may become
// actual transfers depending on source/target shape) - they'll be elided if not
// needed.
struct ConvertTensorReshapeOp
    : public OpConversionPattern<IREE::Flow::TensorReshapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::TensorReshapeOp op, ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::TensorReshapeOpAdaptor operands(newOperands,
                                                op->getAttrDictionary());
    Type resultType = IREE::Stream::ResourceType::get(getContext());
    Value sourceSize = querySizeOf(op.getLoc(), operands.source(), op.source(),
                                   op.source_dims(), rewriter);
    Value resultSize =
        buildResultSizeOf(op.getLoc(), op.result(), op.result_dims(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorCloneOp>(
        op, resultType, operands.source(), op.source().getType(),
        op.source_dims(), sourceSize, op.result().getType(),
        operands.result_dims(), resultSize,
        /*affinity=*/nullptr);
    return success();
  }
};

struct ConvertTensorSplatOp
    : public OpConversionPattern<IREE::Flow::TensorSplatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::TensorSplatOp op, ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::TensorSplatOpAdaptor operands(newOperands);
    Type resultType = IREE::Stream::ResourceType::get(getContext());
    Value resultSize =
        buildResultSizeOf(op.getLoc(), op.result(), op.result_dims(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorSplatOp>(
        op, resultType, operands.value(), op.result().getType(),
        operands.result_dims(), resultSize,
        /*affinity=*/nullptr);
    return success();
  }
};

struct ConvertTensorCloneOp
    : public OpConversionPattern<IREE::Flow::TensorCloneOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::TensorCloneOp op, ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::TensorCloneOpAdaptor operands(newOperands);
    Type resultType = IREE::Stream::ResourceType::get(getContext());
    Value size = querySizeOf(op.getLoc(), operands.operand(), op.operand(),
                             op.operand_dims(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorCloneOp>(
        op, resultType, operands.operand(), op.operand().getType(),
        op.operand_dims(), size, op.result().getType(), operands.operand_dims(),
        size,
        /*affinity=*/nullptr);
    return success();
  }
};

struct ConvertTensorSliceOp
    : public OpConversionPattern<IREE::Flow::TensorSliceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::TensorSliceOp op, ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::TensorSliceOpAdaptor operands(newOperands,
                                              op->getAttrDictionary());
    Type resultType = IREE::Stream::ResourceType::get(getContext());
    Value sourceSize = querySizeOf(op.getLoc(), operands.source(), op.source(),
                                   op.source_dims(), rewriter);
    Value resultSize =
        buildResultSizeOf(op.getLoc(), op.result(), op.result_dims(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorSliceOp>(
        op, resultType, operands.source(), op.source().getType(),
        op.source_dims(), sourceSize, operands.start_indices(),
        operands.lengths(), op.result().getType(), operands.result_dims(),
        resultSize,
        /*affinity=*/nullptr);
    return success();
  }
};

struct ConvertTensorUpdateOp
    : public OpConversionPattern<IREE::Flow::TensorUpdateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::TensorUpdateOp op, ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::TensorUpdateOpAdaptor operands(newOperands,
                                               op->getAttrDictionary());
    Value updateSize = querySizeOf(op.getLoc(), operands.update(), op.update(),
                                   op.update_dims(), rewriter);
    Value targetSize = querySizeOf(op.getLoc(), operands.target(), op.target(),
                                   op.target_dims(), rewriter);
    rewriter.replaceOpWithNewOp<IREE::Stream::TensorUpdateOp>(
        op, operands.target().getType(), operands.target(),
        op.target().getType(), operands.target_dims(), targetSize,
        operands.start_indices(), operands.update(), op.update().getType(),
        op.update_dims(), updateSize,
        /*affinity=*/nullptr);
    return success();
  }
};

struct ConvertTensorLoadOp
    : public OpConversionPattern<IREE::Flow::TensorLoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::TensorLoadOp op, ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::TensorLoadOpAdaptor operands(newOperands,
                                             op->getAttrDictionary());
    auto resultType = getTypeConverter()->convertType(op.result().getType());
    Value sourceSize = querySizeOf(op.getLoc(), operands.source(), op.source(),
                                   op.source_dims(), rewriter);

    auto stagingType = IREE::Stream::ResourceType::get(
        rewriter.getContext(), IREE::Stream::Lifetime::Staging);
    Value loadSource = operands.source();
    if (operands.source().getType() != stagingType) {
      loadSource = rewriter.createOrFold<IREE::Stream::AsyncTransferOp>(
          op.getLoc(), stagingType, operands.source(), sourceSize, sourceSize,
          /*source_affinity=*/nullptr,
          /*result_affinity=*/nullptr);
    }

    rewriter.replaceOpWithNewOp<IREE::Stream::TensorLoadOp>(
        op, resultType, loadSource, op.source().getType(), op.source_dims(),
        sourceSize, operands.indices());
    return success();
  }
};

struct ConvertTensorStoreOp
    : public OpConversionPattern<IREE::Flow::TensorStoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::TensorStoreOp op, ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::TensorStoreOpAdaptor operands(newOperands,
                                              op->getAttrDictionary());
    Value targetSize = querySizeOf(op.getLoc(), operands.target(), op.target(),
                                   op.target_dims(), rewriter);

    auto stagingType = IREE::Stream::ResourceType::get(
        rewriter.getContext(), IREE::Stream::Lifetime::Staging);
    Value storeTarget = operands.target();
    if (operands.target().getType() != stagingType) {
      storeTarget = rewriter.createOrFold<IREE::Stream::AsyncTransferOp>(
          op.getLoc(), stagingType, storeTarget, targetSize, targetSize,
          /*source_affinity=*/nullptr,
          /*result_affinity=*/nullptr);
    }

    auto newOp = rewriter.create<IREE::Stream::TensorStoreOp>(
        op.getLoc(), storeTarget.getType(), storeTarget, op.target().getType(),
        operands.target_dims(), targetSize, operands.indices(),
        operands.value());

    Value newResult = newOp.result();
    if (operands.target().getType() != stagingType) {
      newResult = rewriter.createOrFold<IREE::Stream::AsyncTransferOp>(
          op.getLoc(), operands.target().getType(), newResult, targetSize,
          targetSize,
          /*source_affinity=*/nullptr,
          /*result_affinity=*/nullptr);
    }
    rewriter.replaceOp(op, {newResult});

    return success();
  }
};

struct ConvertDispatchOp : public OpConversionPattern<IREE::Flow::DispatchOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::DispatchOp op, ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    IREE::Flow::DispatchOpAdaptor operands(newOperands,
                                           op->getAttrDictionary());

    // Query all operand sizes.
    SmallVector<Value> operandSizes;
    for (auto oldNewOperand : llvm::zip(op.operands(), operands.operands())) {
      auto oldOperand = std::get<0>(oldNewOperand);
      auto newOperand = std::get<1>(oldNewOperand);
      if (oldOperand.getType().isa<ShapedType>()) {
        auto operandDynamicDims = Shape::buildOrFindDynamicDimsForValue(
            op.getLoc(), oldOperand, rewriter);
        operandSizes.push_back(querySizeOf(op.getLoc(), newOperand, oldOperand,
                                           operandDynamicDims, rewriter));
      }
    }

    // Construct result sizes or reuse tied operand sizes from above.
    SmallVector<Value> resultSizes;
    SmallVector<Type> resultTypes;
    auto unknownType = IREE::Stream::ResourceType::get(getContext());
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
        resultTypes.push_back(operands.operands()[operandIndex].getType());
      } else {
        auto resultDynamicDims = Shape::buildOrFindDynamicDimsForValue(
            op.getLoc(), result.value(), rewriter);
        resultSizes.push_back(buildResultSizeOf(op.getLoc(), result.value(),
                                                resultDynamicDims, rewriter));
        resultTypes.push_back(unknownType);
      }
    }

    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncDispatchOp>(
        op, resultTypes, operands.workgroup_count(), operands.entry_point(),
        operands.operands(), operandSizes, resultSizes,
        operands.tied_operands(),
        /*affinity=*/nullptr);
    return success();
  }
};

static SmallVector<Value> makeBindingDynamicDims(
    Location loc, IREE::Flow::DispatchTensorType tensorType, BlockArgument arg,
    OpBuilder &builder) {
  if (tensorType.hasStaticShape()) return {};

  // We can expect its first user to be a tie_shape op to associate
  // concrete dimension values. Originally we have such information
  // maintained in the flow ops handling dynamic tensors. But during
  // flow executable outlining, such information is transfered to
  // tie_shape ops.
  //
  // HACK: this is disgusting - we should carry this from the flow level in the
  // right way such that we don't need to make this assumption.
  auto tieShapeOp = dyn_cast<IREE::Flow::DispatchTieShapeOp>(*arg.user_begin());
  assert(tieShapeOp && "missing flow tie shape for dynamic value");
  builder.setInsertionPointAfter(tieShapeOp.shape().getDefiningOp());

  // Get the SSA values for all dynamic dimensions.
  SmallVector<Value> dynamicDims;
  dynamicDims.reserve(tensorType.getNumDynamicDims());
  for (int i = 0; i < tensorType.getRank(); ++i) {
    if (!tensorType.isDynamicDim(i)) continue;
    dynamicDims.push_back(builder.create<Shape::RankedDimOp>(
        tieShapeOp.getLoc(), tieShapeOp.shape(), i));
  }
  assert(dynamicDims.size() == tensorType.getNumDynamicDims());

  return dynamicDims;
}

struct ConvertExecutableOp
    : public OpConversionPattern<IREE::Flow::ExecutableOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::Flow::ExecutableOp flowOp, ArrayRef<Value> newOperands,
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
    // operands. Note that we only touch public (exported) functions.
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
          auto dynamicDims =
              makeBindingDynamicDims(arg.getLoc(), tensorType, arg, rewriter);
          auto subspanOp = rewriter.create<IREE::Stream::BindingSubspanOp>(
              arg.getLoc(), tensorType, arg, zero, dynamicDims);
          arg.replaceAllUsesExcept(subspanOp.result(), subspanOp);
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
  typeConverter.addConversion(
      [](IREE::HAL::BufferViewType type) { return type; });
  patterns.insert<ConvertHALTensorCastOp>(typeConverter, context);

  patterns
      .insert<ConvertTensorReshapeOp, ConvertTensorSplatOp,
              ConvertTensorCloneOp, ConvertTensorSliceOp, ConvertTensorUpdateOp,
              ConvertTensorLoadOp, ConvertTensorStoreOp>(typeConverter,
                                                         context);
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

  conversionTarget.addDynamicallyLegalOp<IREE::HAL::TensorCastOp>(
      [&](IREE::HAL::TensorCastOp op) {
        return typeConverter.isLegal(op.source().getType()) &&
               typeConverter.isLegal(op.target().getType());
      });

  populateFlowToStreamConversionPatterns(context, typeConverter, patterns);
}

}  // namespace iree_compiler
}  // namespace mlir
