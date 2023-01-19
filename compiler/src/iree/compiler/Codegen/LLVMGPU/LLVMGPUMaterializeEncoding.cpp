// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===---------------------------------------------------------------------===//
// Pass to materialize the encoding of tensor based on target information.
//===---------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/EncodingInfo.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-materialize-encoding"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

// Test flag that enables transpose THS matrix. It can be deleted.
constexpr bool ENABLE_TRANSPOSE = true;

namespace mlir {
namespace iree_compiler {

using namespace IREE::LinalgExt;
using IREE::HAL::ExecutableTargetAttr;

namespace {

static RankedTensorType getMaterializedType(RankedTensorType tensorType) {
  Optional<TensorEncoding> encoding = getEncoding(tensorType);
  if (!encoding) return tensorType;
  auto matmulOperandRole = getMatmulOperandRole(*encoding);
  ArrayRef<int64_t> transposed = tensorType.getShape();
  if (ENABLE_TRANSPOSE)
    if (matmulOperandRole.value() == MatmulOperandRole::RHS_TRANSPOSE) {
      transposed = llvm::to_vector(llvm::reverse(tensorType.getShape()));
    }
  return RankedTensorType::get(transposed, tensorType.getElementType());
}

/// TypeConverter to use for materializing the encoding.
struct EncodingTypeConverter : public TypeConverter {
  EncodingTypeConverter() {
    addConversion([](IntegerType intType) { return intType; });
    addConversion([](IndexType indexType) { return indexType; });
    addConversion([](FloatType floatType) { return floatType; });
    addConversion([](MemRefType memrefType) { return memrefType; });
    addConversion([](RankedTensorType t) -> RankedTensorType {
      return getMaterializedType(t);
    });
  }
};

/// Conversion target to use for for materializing the encoding.
struct EncodingConversionTarget : public ConversionTarget {
  EncodingConversionTarget(MLIRContext &context) : ConversionTarget(context) {
    // Mark any operation that has operands/results with encoding as
    // illegal.
    markUnknownOpDynamicallyLegal([](Operation *op) {
      auto typeHasEncoding = [](Type t) -> bool {
        auto tensorType = t.dyn_cast<RankedTensorType>();
        return tensorType && tensorType.getEncoding();
      };

      auto valueHasEncoding = [=](Value v) -> bool {
        return typeHasEncoding(v.getType());
      };

      bool hasOperandOrResultsWithEncoding =
          llvm::any_of(op->getOperands(), valueHasEncoding) ||
          llvm::any_of(op->getResultTypes(), typeHasEncoding);

      LLVM_DEBUG(DBGS() << "\n############################\n"
                        << *(op->getParentOp())
                        << "\n############################\n");

      if (hasOperandOrResultsWithEncoding) return false;

      if (auto storeOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(*op)) {
        if (auto subspanOp =
                storeOp.getTarget()
                    .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>()) {
          if (auto resultType =
                  subspanOp.getResult()
                      .getType()
                      .template dyn_cast<IREE::Flow::DispatchTensorType>()) {
            if (auto boundTensorType =
                    resultType.getBoundType().dyn_cast<RankedTensorType>()) {
              auto encoding = getEncoding(boundTensorType);
              if (encoding.has_value()) return false;
            }
          }
        }
      }

      return true;
    });
  }
};

/// Base class for patterns that materialize encoding.
template <typename OpTy>
class GPUMaterializeEncodingPattern : public OpConversionPattern<OpTy> {
 public:
  GPUMaterializeEncodingPattern(MLIRContext *context,
                                EncodingTypeConverter &typeConverter,
                                PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(typeConverter, context, benefit) {}
};

/// Pattern to materialize the encoding for `hal.interface.binding.subspan`
/// operations.
struct MaterializeInterfaceBindingEncoding
    : public GPUMaterializeEncodingPattern<
          IREE::HAL::InterfaceBindingSubspanOp> {
  using GPUMaterializeEncodingPattern<
      IREE::HAL::InterfaceBindingSubspanOp>::GPUMaterializeEncodingPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceBindingSubspanOp subspanOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = subspanOp.getResult()
                          .getType()
                          .dyn_cast<IREE::Flow::DispatchTensorType>();
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          subspanOp, "expected result type to be !flow.dispatch.tensor");
    }
    auto boundTensorType =
        resultType.getBoundType().dyn_cast<RankedTensorType>();
    if (!boundTensorType) {
      return rewriter.notifyMatchFailure(
          subspanOp, "bound type is not a RankedTensorType");
    }

    auto convertedBoundType = getTypeConverter()->convertType(boundTensorType);
    if (convertedBoundType == boundTensorType) {
      return rewriter.notifyMatchFailure(subspanOp, "bound type already valid");
    }

    auto newResultType = IREE::Flow::DispatchTensorType::get(
        resultType.getAccess(), convertedBoundType);

    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp, newResultType, subspanOp.getSet(), subspanOp.getBinding(),
        subspanOp.getDescriptorType(), subspanOp.getByteOffset(),
        subspanOp.getDynamicDims(), subspanOp.getAlignmentAttr());

    return success();
  }
};

static FailureOr<SmallVector<OpFoldResult>> myPacked(
    OpBuilder &builder, Location loc, EncodingTypeConverter &typeConverter,
    IREE::Flow::DispatchTensorType dispatchTensorType, ValueRange dynamicDims) {
  auto boundTensorType =
      dispatchTensorType.getBoundType().dyn_cast<RankedTensorType>();
  if (!boundTensorType) {
    return failure();
  }

  MaterializeEncodingInfo encodingInfo;

  SmallVector<OpFoldResult, 4> targetShape =
      getMixedValues(dispatchTensorType.getShape(), dynamicDims, builder);
  auto innerTileSizes =
      getInnerTileSizesOfr(builder, loc, boundTensorType, encodingInfo, {});
  if (failed(innerTileSizes)) return failure();
  SmallVector<OpFoldResult> convertedTargetShape = PackOp::getResultShape(
      builder, loc, targetShape, *innerTileSizes, encodingInfo.innerDimsPos,
      encodingInfo.outerDimsPerm);
  return convertedTargetShape;
}

/// Pattern to convert `flow.dispatch.tensor.store` operation when
/// materializing the encoding.
struct MaterializeFlowDispatchTensorLoadOp
    : public GPUMaterializeEncodingPattern<IREE::Flow::DispatchTensorLoadOp> {
  using GPUMaterializeEncodingPattern<
      IREE::Flow::DispatchTensorLoadOp>::GPUMaterializeEncodingPattern;

  LogicalResult matchAndRewrite(
      IREE::Flow::DispatchTensorLoadOp loadOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!loadOp.isLoadOfWholeSource()) {
      return rewriter.notifyMatchFailure(loadOp, "unhandled partial loads");
    }

    auto sourceType = loadOp.getSourceType();
    // auto boundTensorType = sourceType.getBoundType();
    auto *typeConverter =
        static_cast<EncodingTypeConverter *>(getTypeConverter());

    Location loc = loadOp.getLoc();
    FailureOr<SmallVector<OpFoldResult>> convertedMixedSizes = myPacked(
        rewriter, loc, *typeConverter, sourceType, adaptor.getSourceDims());

    if (failed(convertedMixedSizes)) {
      return rewriter.notifyMatchFailure(
          loadOp, "failed to get converted dynamic dims for result");
    }
    SmallVector<OpFoldResult> convertedOffsets(convertedMixedSizes->size(),
                                               rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> convertedStrides(convertedMixedSizes->size(),
                                               rewriter.getIndexAttr(1));
    SmallVector<int64_t> convertedStaticDims;
    SmallVector<Value> convertedDynamicDims;
    dispatchIndexOpFoldResults(convertedMixedSizes.value(),
                               convertedDynamicDims, convertedStaticDims);

    if (auto resultType =
            adaptor.getSource()
                .getType()
                .template dyn_cast<IREE::Flow::DispatchTensorType>()) {
      if (auto boundTensorType =
              resultType.getBoundType().dyn_cast<RankedTensorType>()) {
        rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
            loadOp, boundTensorType, adaptor.getSource(), convertedDynamicDims,
            convertedOffsets, convertedMixedSizes.value(), convertedStrides);
      }
    } else {
      rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
          loadOp, adaptor.getSource(), convertedDynamicDims, convertedOffsets,
          convertedMixedSizes.value(), convertedStrides);
    }
    return success();
  }
};

/// Pattern to convert `flow.dispatch.tensor.store` operation when
/// materializing the encoding.
struct MaterializeFlowDispatchTensorStoreOp
    : public GPUMaterializeEncodingPattern<IREE::Flow::DispatchTensorStoreOp> {
  using GPUMaterializeEncodingPattern<
      IREE::Flow::DispatchTensorStoreOp>::GPUMaterializeEncodingPattern;

  LogicalResult matchAndRewrite(
      IREE::Flow::DispatchTensorStoreOp storeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!storeOp.isStoreToWholeTarget()) {
      return rewriter.notifyMatchFailure(storeOp, "unhandled partial stores");
    }
    auto targetType = storeOp.getTargetType();
    auto *typeConverter =
        static_cast<EncodingTypeConverter *>(getTypeConverter());

    Location loc = storeOp.getLoc();
    FailureOr<SmallVector<OpFoldResult>> convertedMixedSizes = myPacked(
        rewriter, loc, *typeConverter, targetType, storeOp.getTargetDims());
    if (failed(convertedMixedSizes)) {
      return rewriter.notifyMatchFailure(
          storeOp, "failed to get converted dynamic dims for result");
    }
    SmallVector<OpFoldResult> convertedOffsets(convertedMixedSizes->size(),
                                               rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> convertedStrides(convertedMixedSizes->size(),
                                               rewriter.getIndexAttr(1));
    SmallVector<int64_t> convertedStaticDims;
    SmallVector<Value> convertedDynamicDims;
    dispatchIndexOpFoldResults(convertedMixedSizes.value(),
                               convertedDynamicDims, convertedStaticDims);

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        storeOp, adaptor.getValue(), adaptor.getTarget(), convertedDynamicDims,
        convertedOffsets, convertedMixedSizes.value(), convertedStrides);

    return success();
  }
};

//===---------------------------------------------------------------------===//
// Patterns to lower ops with encodings. These are written as
// dialect conversion patterns for now. These are just drivers around
// the core conversion utilities.
//===---------------------------------------------------------------------===//

/// Remove all the `set_encoding` op
struct SetEncodingOpToPackOpConversion
    : public GPUMaterializeEncodingPattern<SetEncodingOp> {
  using GPUMaterializeEncodingPattern<
      SetEncodingOp>::GPUMaterializeEncodingPattern;

  LogicalResult matchAndRewrite(
      SetEncodingOp encodingOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    RankedTensorType resultType = encodingOp.getResultType();
    Optional<TensorEncoding> encoding = getEncoding(resultType);

    if (!encoding)
      return encodingOp->emitError("The op should have encoding information");

    Value src = encodingOp.getSource();
    auto matmulOperandRole = getMatmulOperandRole(*encoding);
    if (ENABLE_TRANSPOSE &&
        matmulOperandRole.value() == MatmulOperandRole::RHS_TRANSPOSE) {
      auto loc = encodingOp->getLoc();

      SmallVector<int64_t> perm;
      SmallVector<int64_t> shape;
      for (int i = resultType.getRank() - 1; i != -1; --i) {
        perm.push_back(i);
        shape.push_back(resultType.getShape()[i]);
      }

      auto emptyOp = rewriter.create<tensor::EmptyOp>(
          loc, shape, resultType.getElementType());

      auto transposeOp =
          rewriter.create<linalg::TransposeOp>(loc, src, emptyOp, perm);

      rewriter.replaceAllUsesWith(encodingOp->getResult(0),
                                  transposeOp->getResult(0));
      rewriter.eraseOp(encodingOp);

    } else {
      rewriter.replaceAllUsesWith(encodingOp->getResult(0), src);
      rewriter.eraseOp(encodingOp);
    }
    return success();
  }
};

/// Remove `unset_encoding` op
struct UnsetEncodingOpToPackOpConversion
    : public GPUMaterializeEncodingPattern<UnsetEncodingOp> {
  using GPUMaterializeEncodingPattern<
      UnsetEncodingOp>::GPUMaterializeEncodingPattern;

  LogicalResult matchAndRewrite(
      UnsetEncodingOp encodingOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSource();
    rewriter.replaceAllUsesWith(encodingOp->getResult(0), src);
    rewriter.eraseOp(encodingOp);
    return success();
  }
};

/// Utility method to convert from `linalg.fill` on `tensor` type with
/// encoding to fill of the materialized type
static FailureOr<Operation *> lowerOpWithEncoding(
    RewriterBase &rewriter, linalg::FillOp fillOp,
    ValueRange convertedInputOperands, ValueRange convertedOutputOperands) {
  if (!fillOp.hasTensorSemantics()) return failure();
  Operation *materializedFillOp = rewriter.create<linalg::FillOp>(
      fillOp.getLoc(), convertedOutputOperands[0].getType(),
      convertedInputOperands, convertedOutputOperands);
  return materializedFillOp;
}

static FailureOr<Operation *> lowerOpWithEncoding(
    RewriterBase &rewriter, linalg::MatmulOp matmulOp,
    ValueRange convertedInputOperands, ValueRange convertedOutputOperands) {
  if (!matmulOp.hasTensorSemantics()) return failure();
  Operation *materializedFillOp;
  if (ENABLE_TRANSPOSE) {
    materializedFillOp = rewriter.create<linalg::MatmulTransposeBOp>(
        matmulOp.getLoc(), convertedOutputOperands[0].getType(),
        convertedInputOperands, convertedOutputOperands);
  } else {
    materializedFillOp = rewriter.create<linalg::MatmulOp>(
        matmulOp.getLoc(), convertedOutputOperands[0].getType(),
        convertedInputOperands, convertedOutputOperands);
  }
  return materializedFillOp;
}

/// Utility method to convert `tensor.empty` with encoding to a
/// `tensor.empty` of the materialized type.
static FailureOr<Operation *> lowerOpWithEncoding(RewriterBase &rewriter,
                                                  tensor::EmptyOp emptyOp,
                                                  ValueRange convertedOperands,
                                                  TypeConverter *converter) {
  auto result = emptyOp.getResult();
  auto resultType = result.getType().cast<RankedTensorType>();
  auto newType = converter->convertType(resultType).cast<RankedTensorType>();
  Operation *materializedOp = rewriter.create<tensor::EmptyOp>(
      emptyOp->getLoc(), newType.getShape(), newType.getElementType());
  return materializedOp;
}

/// Generic pattern to convert operaiton that is in Destination Passing
/// Style.
template <typename OpTy>
struct MaterializeDPSOperation : public GPUMaterializeEncodingPattern<OpTy> {
  using GPUMaterializeEncodingPattern<OpTy>::GPUMaterializeEncodingPattern;

  LogicalResult matchAndRewrite(
      OpTy dpsOp, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Operation *> convertedOp = lowerOpWithEncoding(
        rewriter, dpsOp, adaptor.getInputs(), adaptor.getOutputs());
    if (failed(convertedOp)) return failure();
    rewriter.replaceOp(dpsOp, convertedOp.value()->getResults());
    return success();
  }
};

/// Generic pattern to convert an operation.
template <typename OpTy>
struct MaterializeOperation : public GPUMaterializeEncodingPattern<OpTy> {
  using GPUMaterializeEncodingPattern<OpTy>::GPUMaterializeEncodingPattern;
  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FailureOr<Operation *> convertedOp = lowerOpWithEncoding(
        rewriter, op, adaptor.getOperands(), this->getTypeConverter());
    if (failed(convertedOp)) return failure();
    rewriter.replaceOp(op, convertedOp.value()->getResults());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Materialization Pass
//===----------------------------------------------------------------------===//

struct LLVMGPUMaterializeEncoding
    : public LLVMGPUMaterializeEncodingBase<LLVMGPUMaterializeEncoding> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, AffineDialect, IREE::Flow::FlowDialect,
                    scf::SCFDialect>();
  }
  void runOnOperation() override;
};

}  // namespace

void LLVMGPUMaterializeEncoding::runOnOperation() {
  MLIRContext *context = &getContext();
  auto operation = getOperation();

  {
    RewritePatternSet materializeEncodingPattern(context);

    EncodingTypeConverter typeConverter;
    // Add type conversion for `!flow.dispatch.tensor` type.
    typeConverter.addConversion(
        [&typeConverter](IREE::Flow::DispatchTensorType dispatchTensorType) {
          Type boundType = dispatchTensorType.getBoundType();
          Type convertedBoundType = typeConverter.convertType(boundType);
          if (convertedBoundType == boundType) {
            return dispatchTensorType;
          }
          return IREE::Flow::DispatchTensorType::get(
              dispatchTensorType.getAccess(), convertedBoundType);
        });

    EncodingConversionTarget target(*context);
    target.addDynamicallyLegalOp<IREE::HAL::InterfaceBindingSubspanOp>(
        [&typeConverter](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
          auto resultType =
              subspanOp.getResult()
                  .getType()
                  .template dyn_cast<IREE::Flow::DispatchTensorType>();
          // For types that are not `Flow::DispatchTensorType` mark as legal.
          if (!resultType) return true;
          return resultType == typeConverter.convertType(resultType);
        });

    materializeEncodingPattern.insert<
        MaterializeInterfaceBindingEncoding, SetEncodingOpToPackOpConversion,
        UnsetEncodingOpToPackOpConversion, MaterializeFlowDispatchTensorStoreOp,
        MaterializeFlowDispatchTensorLoadOp,
        MaterializeDPSOperation<linalg::MatmulOp>,
        MaterializeDPSOperation<linalg::FillOp>,
        MaterializeOperation<tensor::EmptyOp>>(context, typeConverter);
    if (failed(applyPartialConversion(operation, target,
                                      std::move(materializeEncodingPattern)))) {
      operation.emitOpError("materialization failed");
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUMaterializeEncoding() {
  return std::make_unique<LLVMGPUMaterializeEncoding>();
}

}  // namespace iree_compiler
}  // namespace mlir
