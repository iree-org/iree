// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Stream/Conversion/FlowToStream/Patterns.h"
#include "iree/compiler/Dialect/Stream/Conversion/HALToStream/Patterns.h"
#include "iree/compiler/Dialect/Stream/Conversion/PatternUtils.h"
#include "iree/compiler/Dialect/Stream/Conversion/StandardToStream/Patterns.h"
#include "iree/compiler/Dialect/Stream/Conversion/UtilToStream/Patterns.h"
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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
// a stream resource. |consumingOps| will be populated with all ops that consume
// the original |sourceTensor| and that should not be replaced with the returned
// value.
static Value buildTensorImportOp(Location loc, Value sourceTensor,
                                 Type targetType,
                                 SmallPtrSetImpl<Operation *> &consumingOps,
                                 OpBuilder &builder) {
  // Gather dynamic dimensions from the input value.
  auto dynamicDims =
      IREE::Util::buildDynamicDimsForValue(loc, sourceTensor, builder);

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
  auto externalType = builder.getType<IREE::Stream::ResourceType>(
      IREE::Stream::Lifetime::External);
  auto importOp = builder.create<IREE::Stream::TensorImportOp>(
      loc, externalType, sourceTensor, encodingAttr, dynamicDims, resultSize,
      /*affinity=*/nullptr);
  consumingOps.insert(importOp);

  // If needed insert a transfer to the target lifetime.
  Value result = importOp.getResult();
  if (targetType != externalType) {
    result = builder
                 .create<IREE::Stream::AsyncTransferOp>(
                     loc, targetType, result, resultSize, resultSize,
                     /*source_affinity=*/nullptr,
                     /*result_affinity=*/nullptr)
                 .getResult();
  }

  auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(
      loc, sourceTensor.getType(), ValueRange{result, resultSize});
  consumingOps.insert(castOp);
  return castOp.getResult(0);
}

// Builds a stream.tensor.export op that exports a stream resource into an
// external tensor value.
static Value buildTensorExportOp(Location loc, Value sourceValue,
                                 TensorType targetType, ValueRange dynamicDims,
                                 OpBuilder &builder) {
  auto source = consumeTensorOperand(loc, sourceValue, builder);

  // If needed insert a transfer to external resource lifetime.
  auto externalType = builder.getType<IREE::Stream::ResourceType>(
      IREE::Stream::Lifetime::External);
  if (source.resource.getType() != externalType) {
    source.resource = builder.create<IREE::Stream::AsyncTransferOp>(
        loc, externalType, source.resource, source.resourceSize,
        source.resourceSize,
        /*source_affinity=*/nullptr,
        /*result_affinity=*/nullptr);
  }

  // Associate the stream resource and external encoding and shape information.
  auto newOp = builder.create<IREE::Stream::TensorExportOp>(
      loc, targetType, source.resource, TypeAttr::get(targetType), dynamicDims,
      source.resourceSize,
      /*affinity=*/nullptr);
  return newOp.getResult();
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
    for (auto [oldOperand, newOperand] :
         llvm::zip_equal(op->getOperands(), operands)) {
      if (!newOperand.getType().isa<IREE::Stream::ResourceType>() &&
          !newOperand.getType().isa<TensorType>()) {
        newOperands.push_back(newOperand);
        continue;
      }
      auto tensorType = oldOperand.getType().dyn_cast<TensorType>();
      assert(tensorType && "must have a tensor type to map to a resource");

      auto dynamicDims = IREE::Util::buildDynamicDimsForValue(
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
          IREE::Util::buildDynamicDimsForValue(op->getLoc(), result, rewriter);
      SmallPtrSet<Operation *, 4> consumingOps;
      auto importedValue = buildTensorImportOp(
          op->getLoc(), result, rewriter.getType<IREE::Stream::ResourceType>(),
          consumingOps, rewriter);
      result.replaceAllUsesExcept(importedValue, consumingOps);
    }

    return success();
  }
};

class ConvertToStreamPass : public ConvertToStreamBase<ConvertToStreamPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();

    TypeConverter typeConverter;
    ConversionTarget conversionTarget(getContext());
    RewritePatternSet patterns(&getContext());

    // Always allow lowering target dialects and reasonable types.
    conversionTarget.addLegalDialect<IREE::Stream::StreamDialect>();
    typeConverter.addConversion(
        [](IREE::Stream::ResourceType type) { return type; });

    // Allow unknown types to pass through; these come from custom dialects that
    // may be mixed into the IR we are converting.
    typeConverter.addConversion([=](Type type) -> Type {
      // convert flow.channel into stream.channel
      if (type.isa<IREE::Flow::ChannelType>())
        return IREE::Stream::ChannelType::get(context);

      return !type.isa<TensorType>() ? type : Type{};
    });

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
           Location loc) -> std::optional<Value> {
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
              .getResult();
        });

    populateUtilConversionPatterns(context, conversionTarget, typeConverter,
                                   patterns);
    populateUtilToStreamConversionPatterns(context, conversionTarget,
                                           typeConverter, patterns);

    populateStandardToStreamConversionPatterns(context, conversionTarget,
                                               typeConverter, patterns);
    populateFlowToStreamConversionPatterns(context, conversionTarget,
                                           typeConverter, patterns);
    populateHALToStreamConversionPatterns(context, conversionTarget,
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
