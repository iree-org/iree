// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Shape/IR/Builders.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Stream/Conversion/FlowToStream/ConvertFlowToStream.h"
#include "iree/compiler/Dialect/Stream/Conversion/StandardToStream/ConvertStandardToStream.h"
#include "iree/compiler/Dialect/Stream/Conversion/UtilToStream/ConvertUtilToStream.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Patterns.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {
namespace {

// Builds a stream.tensor.import op that imports an external tensor value into
// a stream resource.
static Value buildTensorImportOp(Location loc, Value sourceTensor,
                                 Type targetType, OpBuilder &builder) {
  // Gather dynamic dimensions from the input value.
  auto dynamicDims =
      Shape::buildOrFindDynamicDimsForValue(loc, sourceTensor, builder);

  // Compute the size of the tensor once in the stream resource.
  // This may differ from the external encoding of the tensor as imports are
  // a transfer operation that may need to reformat the tensor.
  auto encodingAttr = TypeAttr::get(sourceTensor.getType());
  auto resultSize = builder.createOrFold<IREE::Stream::TensorSizeOfOp>(
      loc, builder.getIndexType(), encodingAttr, dynamicDims,
      /*affinity=*/nullptr);

  // Associate the external SSA value, encoding, and shape information with the
  // stream resource. When lowering we'll then have all the metadata required
  // even after erasing it all on the resource.
  auto externalType = IREE::Stream::ResourceType::get(
      builder.getContext(), IREE::Stream::Lifetime::External);
  auto importOp = builder.create<IREE::Stream::TensorImportOp>(
      loc, externalType, sourceTensor, encodingAttr, dynamicDims, resultSize,
      /*affinity=*/nullptr);

  // If needed insert a transfer to the target lifetime.
  Value result = importOp.result();
  if (targetType != externalType) {
    result = builder
                 .create<IREE::Stream::AsyncTransferOp>(
                     loc, externalType, result, resultSize, resultSize,
                     /*source_affinity=*/nullptr,
                     /*result_affinity=*/nullptr)
                 .result();
  }

  return result;
}

// Builds a stream.tensor.export op that exports a stream resource into an
// external tensor value.
static Value buildTensorExportOp(Location loc, Value sourceResource,
                                 TensorType targetType, ValueRange dynamicDims,
                                 OpBuilder &builder) {
  // Query the size of the resource - which may differ from the target external
  // value if we changed the encoding.
  auto sourceSize = builder.createOrFold<IREE::Stream::ResourceSizeOp>(
      loc, builder.getIndexType(), sourceResource);

  // If needed insert a transfer to external resource lifetime.
  auto externalType = IREE::Stream::ResourceType::get(
      builder.getContext(), IREE::Stream::Lifetime::External);
  if (sourceResource.getType() != externalType) {
    sourceResource = builder.create<IREE::Stream::AsyncTransferOp>(
        loc, externalType, sourceResource, sourceSize, sourceSize,
        /*source_affinity=*/nullptr,
        /*result_affinity=*/nullptr);
  }

  // Associate the stream resource and external encoding and shape information.
  auto newOp = builder.create<IREE::Stream::TensorExportOp>(
      loc, targetType, sourceResource, TypeAttr::get(targetType), dynamicDims,
      sourceSize,
      /*affinity=*/nullptr);
  return newOp.result();
}

// Returns true if |op| has tensor I/O that is not yet imported/exported using
// the stream ops that capture encodings and shapes.
static bool doesOperationNeedWrapping(Operation *op) {
  return llvm::any_of(
             op->getOperands(),
             [&](Value operand) {
               if (!operand.getType().isa<TensorType>()) return false;
               return !isa_and_nonnull<TensorExportOp>(operand.getDefiningOp());
             }) ||
         llvm::any_of(op->getResults(), [&](Value result) {
           if (!result.getType().isa<TensorType>()) return false;
           return !llvm::all_of(result.getUsers(), [&](Operation *user) {
             return isa<TensorImportOp>(user);
           });
         });
}

// Fallback handler for unknown ops taking/returning tensors that need to be
// marshaled into/outof stream resource types.
struct GenericResourcePattern : public ConversionPattern {
  GenericResourcePattern(MLIRContext *context, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 0, context) {}
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!doesOperationNeedWrapping(op)) return failure();

    // Export resources into tensor operands for the op to consume.
    SmallVector<Value> newOperands;
    newOperands.reserve(op->getNumOperands());
    rewriter.setInsertionPoint(op);
    for (auto it : llvm::zip(op->getOperands(), operands)) {
      auto oldOperand = std::get<0>(it);
      auto newOperand = std::get<1>(it);
      if (!newOperand.getType().isa<IREE::Stream::ResourceType>()) {
        newOperands.push_back(newOperand);
        continue;
      }
      auto tensorType = oldOperand.getType().dyn_cast<TensorType>();
      assert(tensorType && "must have a tensor type to map to a resource");

      auto dynamicDims = Shape::buildOrFindDynamicDimsForValue(
          op->getLoc(), oldOperand, rewriter);
      newOperands.push_back(buildTensorExportOp(
          op->getLoc(), newOperand, tensorType, dynamicDims, rewriter));
    }
    rewriter.updateRootInPlace(op, [&]() { op->setOperands(newOperands); });

    // Import into resources from tensor results produced by the op.
    rewriter.setInsertionPointAfter(op);
    for (auto result : op->getResults()) {
      auto tensorType = result.getType().dyn_cast<TensorType>();
      if (!tensorType) continue;

      auto dynamicDims =
          Shape::buildOrFindDynamicDimsForValue(op->getLoc(), result, rewriter);
      auto importedValue = buildTensorImportOp(
          op->getLoc(), result, IREE::Stream::ResourceType::get(getContext()),
          rewriter);
      result.replaceAllUsesExcept(importedValue, importedValue.getDefiningOp());
    }

    return success();
  }
};

class ConvertToStreamPass : public ConvertToStreamBase<ConvertToStreamPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ShapeDialect>();
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<mlir::arith::ArithmeticDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();

    TypeConverter typeConverter;
    ConversionTarget conversionTarget(getContext());
    OwningRewritePatternList patterns(&getContext());

    // Always allow lowerering target dialects and reasonable types.
    conversionTarget.addLegalDialect<IREE::Stream::StreamDialect>();
    typeConverter.addConversion(
        [](IREE::Stream::ResourceType type) { return type; });

    // Disallow tensor dialects; the goal here is to remove all tensors and
    // turn them into stream resource ops.
    auto indexType = IndexType::get(context);
    conversionTarget.addIllegalDialect<tensor::TensorDialect>();
    typeConverter.addConversion(
        [=](TensorType type, SmallVectorImpl<Type> &resultTypes) {
          // Expand tensors to [resource, sizeof resource].
          resultTypes.push_back(IREE::Stream::ResourceType::get(context));
          resultTypes.push_back(indexType);
          return success();
        });
    typeConverter.addArgumentMaterialization(
        [](OpBuilder &builder, TensorType resultType, ValueRange inputs,
           Location loc) -> Optional<Value> {
          assert(inputs.size() >= 2);
          auto resourceValue = inputs[0];
          auto resourceSize = inputs[1];
          assert(inputs.size() == 2 &&
                 "expecting 2 operands (resource + size)");
          return builder
              .create<IREE::Stream::AsyncTransferOp>(
                  loc, resourceValue.getType(), resourceValue, resourceSize,
                  resourceSize,
                  /*source_affinity=*/nullptr,
                  /*result_affinity=*/nullptr)
              .result();
        });

    populateUtilConversionPatterns(context, conversionTarget, typeConverter,
                                   patterns);
    populateUtilToStreamConversionPatterns(context, conversionTarget,
                                           typeConverter, patterns);

    populateStandardToStreamConversionPatterns(context, conversionTarget,
                                               typeConverter, patterns);
    populateFlowToStreamConversionPatterns(context, conversionTarget,
                                           typeConverter, patterns);

    conversionTarget.markUnknownOpDynamicallyLegal(
        [&](Operation *op) -> bool { return !doesOperationNeedWrapping(op); });
    patterns.insert<GenericResourcePattern>(context, typeConverter);

    // NOTE: we allow ops that we don't know about to allow custom dialects
    // that don't need anything Stream-specific to pass through.
    conversionTarget.addLegalOp<UnrealizedConversionCastOp>();
    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertToStreamPass() {
  return std::make_unique<ConvertToStreamPass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
