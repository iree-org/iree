// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- XLAToLinalgOnTensors.cpp - Pass to convert XLA to Linalg on tensors-===//
//
// Pass to convert from XLA to linalg on tensers. Uses the patterns from
// tensorflow/compiler/mlir/xla/transforms/legalize_to_linalg.cc along with
// some IREE specific patterns.
//
//===----------------------------------------------------------------------===//
#include <memory>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/InputConversion/MHLO/ConvertMHLOToFlow.h"
#include "iree/compiler/InputConversion/MHLO/PassDetail.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "iree/compiler/InputConversion/MHLO/Rewriters.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace MHLO {

//===----------------------------------------------------------------------===//
// mhlo.concatenate conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Converts mhlo.concatenate operation to extract_slice ops + insert_slice ops.
struct ConcatenateOpConversion
    : public OpConversionPattern<mhlo::ConcatenateOp> {
  using OpConversionPattern<mhlo::ConcatenateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConcatenateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = this->typeConverter->convertType(op.getResult().getType())
                          .dyn_cast<RankedTensorType>();
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected static shape for output");
    }

    Location loc = op.getLoc();
    int dim = op.dimension();
    int rank = resultType.getRank();
    SmallVector<Value, 3> offsets, sizes, strides;
    for (int i = 0; i < rank; ++i) {
      offsets.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      sizes.push_back(rewriter.createOrFold<tensor::DimOp>(
          loc, adaptor.getOperands()[0], i));
      strides.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    }
    Value resultDimSize = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    for (auto arg : adaptor.getOperands()) {
      auto size = rewriter.createOrFold<tensor::DimOp>(loc, arg, dim);
      resultDimSize =
          rewriter.createOrFold<arith::AddIOp>(loc, resultDimSize, size);
    }
    sizes[dim] = resultDimSize;
    auto initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultType.getShape(), resultType.getElementType());
    auto zeroAttr = rewriter.getZeroAttr(resultType.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    Value result =
        rewriter.create<linalg::FillOp>(loc, zero, initTensor).getResult(0);

    auto toOpFoldResult = [](Value v) -> OpFoldResult {
      auto op = v.getDefiningOp<arith::ConstantIndexOp>();
      if (!op) return v;
      return op.getValue();
    };

    Value accBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    for (auto arg : adaptor.getOperands()) {
      offsets[dim] = accBound;
      sizes[dim] = rewriter.createOrFold<tensor::DimOp>(loc, arg, dim);
      result = rewriter.create<tensor::InsertSliceOp>(
          loc, arg, result,
          llvm::to_vector(llvm::map_range(offsets, toOpFoldResult)),
          llvm::to_vector(llvm::map_range(sizes, toOpFoldResult)),
          llvm::to_vector(llvm::map_range(strides, toOpFoldResult)));
      accBound = rewriter.create<arith::AddIOp>(loc, accBound, sizes[dim]);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// mhlo.fft conversion patterns.
//===----------------------------------------------------------------------===//

/// Creats coefficients based on DFT definition, see
/// https://en.wikipedia.org/wiki/Discrete_Fourier_transform
Value getDFTMatmulCoeff(OpBuilder b, Location loc, RankedTensorType matrixType,
                        bool isRealPart) {
  // scale = 2 * pi / N
  double scale = 2 * M_PI / matrixType.getDimSize(0);

  SmallVector<Attribute> values;
  assert(matrixType.getRank() == 2 && "expected 2D matrix");
  for (auto i : llvm::seq<unsigned>(0, matrixType.getDimSize(0))) {
    for (auto j : llvm::seq<unsigned>(0, matrixType.getDimSize(1))) {
      double v = scale * i * j;
      if (isRealPart) {
        v = cos(v);
      } else {
        v = -sin(v);
      }
      values.push_back(b.getF32FloatAttr(v));
    }
  }
  return b.create<arith::ConstantOp>(
      loc, matrixType, DenseFPElementsAttr::get(matrixType, values));
}

Value createLinalgMatmulOnTensors(OpBuilder b, Location loc,
                                  RankedTensorType resultType, Value lhs,
                                  Value rhs) {
  Value zero = b.create<arith::ConstantOp>(
      loc, b.getZeroAttr(resultType.getElementType()));
  auto initTensor = b.create<linalg::InitTensorOp>(
      loc, /*dyn_size=*/ValueRange{}, resultType.getShape(),
      resultType.getElementType());
  Value zeroTensor =
      b.create<linalg::FillOp>(loc, zero, initTensor).getResult(0);

  switch (lhs.getType().cast<RankedTensorType>().getRank()) {
    case 1:
      return b
          .create<linalg::VecmatOp>(loc, TypeRange{resultType},
                                    ValueRange{lhs, rhs},
                                    ValueRange{zeroTensor})
          .getResult(0);
    case 2:
      return b
          .create<linalg::MatmulOp>(loc, TypeRange{resultType},
                                    ValueRange{lhs, rhs},
                                    ValueRange{zeroTensor})
          .getResult(0);
    default:
      llvm_unreachable("unhandled matmul type");
  }
}

/// Converts mhlo.fft operation to Linalg ops.
struct FftOpConversion : public OpConversionPattern<mhlo::FftOp> {
  using OpConversionPattern<mhlo::FftOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::FftOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op.fft_type() != "RFFT") {
      return rewriter.notifyMatchFailure(op,
                                         "non RFFT types are supported yet");
    }

    auto inputType = adaptor.operand().getType().dyn_cast<RankedTensorType>();
    if (!inputType || !inputType.hasStaticShape() || inputType.getRank() > 2) {
      return rewriter.notifyMatchFailure(op, "only static 1D or 2D dft ops");
    }

    int rank = inputType.getRank();
    int n = inputType.getDimSize(rank - 1);
    int fftLength =
        op.fft_length().getSplatValue<IntegerAttr>().getInt() / 2 + 1;

    Location loc = op.getLoc();
    auto matrixType =
        RankedTensorType::get({n, fftLength}, inputType.getElementType());
    auto resultType =
        RankedTensorType::get(op.getType().cast<RankedTensorType>().getShape(),
                              inputType.getElementType());

    auto realMatrix =
        getDFTMatmulCoeff(rewriter, loc, matrixType, /*isRealPart=*/true);
    auto real = createLinalgMatmulOnTensors(rewriter, loc, resultType,
                                            adaptor.operand(), realMatrix);

    auto imagMatrix =
        getDFTMatmulCoeff(rewriter, loc, matrixType, /*isRealPart=*/false);
    auto imag = createLinalgMatmulOnTensors(rewriter, loc, resultType,
                                            adaptor.operand(), imagMatrix);

    // Pack the results back to mhlo::ComplexOp.
    rewriter.replaceOpWithNewOp<mhlo::ComplexOp>(op, op.getType(), real, imag);
    return success();
  }
};

/// We can convert an mhlo.scatter to a sequence of slices and update slices,
/// with a linalg.generic operation to perform the computational update.
class ScatterToDynamicUpdateSlice
    : public OpConversionPattern<mhlo::ScatterOp> {
 public:
  using OpConversionPattern<mhlo::ScatterOp>::OpConversionPattern;

  static Value collapseFrontDimsIfNeeded(Value value, int64_t batchDims,
                                         ImplicitLocOpBuilder &b) {
    if (batchDims == 1) return value;

    auto type = value.getType().cast<RankedTensorType>();
    auto rank = type.getRank();
    int64_t batchSize = 1;
    for (int i = 0; i < batchDims; i++)
      batchSize = combineDims(batchSize, type.getDimSize(i));

    SmallVector<ReassociationIndices> map;
    map.emplace_back(llvm::to_vector<4>(llvm::seq<int64_t>(0, batchDims)));

    llvm::SmallVector<int64_t> newShape = {batchSize};
    for (int i = batchDims; i < rank; i++) {
      newShape.push_back(type.getDimSize(i));
      map.emplace_back(1, i);
    }

    auto resultType = RankedTensorType::get(newShape, type.getElementType());
    return b.create<tensor::CollapseShapeOp>(resultType, value, map);
  }

  static int64_t combineDims(int64_t a, int64_t b) {
    if (a == ShapedType::kDynamicSize || b == ShapedType::kDynamicSize)
      return ShapedType::kDynamicSize;
    return a * b;
  }

  LogicalResult matchAndRewrite(
      mhlo::ScatterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto operand = op.operand();
    auto indices = op.scatter_indices();
    auto updates = op.updates();

    auto operandTy = operand.getType().dyn_cast<RankedTensorType>();
    auto indicesTy = indices.getType().dyn_cast<RankedTensorType>();
    auto updatesTy = updates.getType().dyn_cast<RankedTensorType>();
    if (!operandTy || !indicesTy || !updatesTy) return failure();

    if (indicesTy.getRank() < 2) return failure();

    auto dimNumbers = op.scatter_dimension_numbers();
    int64_t batchDims = dimNumbers.getIndexVectorDim();
    int64_t numIndices = indicesTy.getDimSize(batchDims);

    // No support for a dynamic number of indices.
    if (batchDims != indicesTy.getRank() - 1 ||
        indicesTy.isDynamicDim(batchDims))
      return failure();

    // Bail on dynamic update dimensions right now to avoid slice validation.
    for (int i = batchDims, s = updatesTy.getRank(); i < s; i++) {
      if (updatesTy.isDynamicDim(i)) return failure();
    }

    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);

    // Make insertion dimensions explicit.
    llvm::SmallVector<int64_t> updatesShape;
    for (int i = 0; i < batchDims; i++)
      updatesShape.push_back(updatesTy.getDimSize(i));
    updatesShape.resize(operandTy.getRank() + batchDims, 1);
    SmallVector<ReassociationIndices> map;
    for (int i = 0; i < batchDims; i++) map.emplace_back(1, i);

    auto updateWindowDims = dimNumbers.getUpdateWindowDims();
    for (int operandDim = 0; operandDim < operandTy.getRank(); operandDim++) {
      int64_t updatesDim = operandDim + batchDims;
      auto begin = updateWindowDims.begin();
      auto end = updateWindowDims.end();
      if (std::find(begin, end, operandDim) == end) {
        map.back().emplace_back(updatesDim);
        continue;
      }
      updatesShape[updatesDim] = updatesTy.getDimSize(map.size());
      map.emplace_back(1, updatesDim);
    }

    if (updatesTy.getRank() < updatesShape.size()) {
      updatesTy =
          RankedTensorType::get(updatesShape, updatesTy.getElementType());
      updates = b.create<tensor::ExpandShapeOp>(updatesTy, updates, map);
    }

    // Collapse batch dimensions together so that we iterate over all batch
    // dimensions together.
    indices = collapseFrontDimsIfNeeded(indices, batchDims, b);
    updates = collapseFrontDimsIfNeeded(updates, batchDims, b);
    indicesTy = indices.getType().cast<RankedTensorType>();
    updatesTy = updates.getType().cast<RankedTensorType>();

    // Grab the first update dimension before it is merged with the batch. We
    // already verified this must be static.
    int64_t firstUpdateStaticDim = updatesTy.getDimSize(1);
    Value firstUpdateDynDim = b.create<tensor::DimOp>(updates, 1);

    // Collapsed the batch dimension into the first update dimension, this
    // avoids reshaping each slice.
    updates = collapseFrontDimsIfNeeded(updates, 2, b);
    updatesTy = updates.getType().cast<RankedTensorType>();

    // Iterate over the batch dimension of the indices.
    Value start = b.create<arith::ConstantIndexOp>(0);
    Value end = b.create<tensor::DimOp>(indices, 0);
    Value step = b.create<arith::ConstantIndexOp>(1);

    auto forOp = b.create<scf::ForOp>(start, end, step, ValueRange({operand}));
    b.setInsertionPointToStart(forOp.getBody());

    // Extract the individual scatter value from the updates.
    llvm::SmallVector<int64_t> updateSliceSizes(updatesTy.getShape().begin(),
                                                updatesTy.getShape().end());
    llvm::SmallVector<int64_t> updateSliceOffsets(operandTy.getRank(), 0);
    llvm::SmallVector<int64_t> updateSliceStrides(operandTy.getRank(), 1);

    updateSliceOffsets[0] = ShapedType::kDynamicStrideOrOffset;
    updateSliceSizes[0] = firstUpdateStaticDim;

    Value firstDimIndex =
        b.create<arith::MulIOp>(forOp.getInductionVar(), firstUpdateDynDim);
    Value updateSlice = b.create<tensor::ExtractSliceOp>(
        RankedTensorType::get(updateSliceSizes, updatesTy.getElementType()),
        updates, ValueRange{firstDimIndex}, ValueRange{}, ValueRange{},
        b.getI64ArrayAttr(updateSliceOffsets),
        b.getI64ArrayAttr(updateSliceSizes),
        b.getI64ArrayAttr(updateSliceStrides));

    // Determine the slice information for the operand value. This includes
    // adjusting for a dynamic batch.
    llvm::SmallVector<Value> updateIndices;
    for (int i = 0; i < numIndices; i++) {
      Value ix = b.create<arith::ConstantIndexOp>(i);
      Value extract =
          b.create<tensor::ExtractOp>(indicesTy.getElementType(), indices,
                                      ValueRange{forOp.getInductionVar(), ix})
              .getResult();
      Value cast = b.create<arith::IndexCastOp>(b.getIndexType(), extract);
      updateIndices.push_back(cast);
    }

    // Strides are dynamic for all indices and 0 otherwise.
    llvm::SmallVector<int64_t> insertOffset(updateIndices.size(),
                                            ShapedType::kDynamicStrideOrOffset);
    insertOffset.resize(operandTy.getRank(), 0);

    auto extractSliceOperandTy =
        RankedTensorType::get(updateSliceSizes, operandTy.getElementType());

    // Extract the slice from the source operand.
    auto extractSliceOperand = b.create<tensor::ExtractSliceOp>(
        extractSliceOperandTy, operand, updateIndices, ValueRange{},
        ValueRange{}, b.getI64ArrayAttr(insertOffset),
        b.getI64ArrayAttr(updateSliceSizes),
        b.getI64ArrayAttr(updateSliceStrides));

    // Generate a linalg::Generic operation to perform
    SmallVector<AffineMap> affineMaps(
        2, b.getMultiDimIdentityMap(operandTy.getRank()));
    SmallVector<StringRef> loopAttrs(operandTy.getRank(),
                                     getParallelIteratorTypeName());

    auto generic = b.create<linalg::GenericOp>(
        loc, TypeRange{extractSliceOperandTy}, ValueRange({updateSlice}),
        ValueRange({extractSliceOperand}), affineMaps, loopAttrs);
    Value genericValue = generic.getResult(0);

    // Inline the update computation and change to scalar operations from rank-0
    // tensors.
    rewriter.inlineRegionBefore(op.update_computation(), generic.region(),
                                generic.region().begin());
    TypeConverter::SignatureConversion signatureConverter(2);
    for (const auto &it : llvm::enumerate(generic->getOperands())) {
      signatureConverter.addInputs(
          it.index(), it.value().getType().cast<ShapedType>().getElementType());
    }
    rewriter.applySignatureConversion(&generic.region(), signatureConverter);

    // Insert the resulting value into the target.
    Value insertSlice = b.create<tensor::InsertSliceOp>(
        operandTy, genericValue, operand, updateIndices, ValueRange{},
        ValueRange{}, b.getI64ArrayAttr(insertOffset),
        b.getI64ArrayAttr(updateSliceSizes),
        b.getI64ArrayAttr(updateSliceStrides));

    // For each iteration of the loop we need to apply it to the insertion.
    b.create<scf::YieldOp>(loc, insertSlice);

    // Replace the previous operation.
    b.setInsertionPointAfterValue(forOp.getResult(0));
    rewriter.replaceOp(op, forOp.getResult(0));
    return success();
  }
};

// We need to convert func ops in order to convert types.
class BuiltinFuncOpPattern : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      FuncOp srcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FunctionType srcFuncType = srcOp.getType();
    TypeConverter::SignatureConversion signatureConversion(
        srcOp.getNumArguments());

    // Convert function arguments.
    for (unsigned i = 0, e = srcFuncType.getNumInputs(); i < e; ++i) {
      if (failed(getTypeConverter()->convertSignatureArg(
              i, srcFuncType.getInput(i), signatureConversion))) {
        return rewriter.notifyMatchFailure(srcOp, "argument failed to convert");
      }
    }

    // Convert function results.
    SmallVector<Type> convertedResultTypes;
    if (failed(getTypeConverter()->convertTypes(srcFuncType.getResults(),
                                                convertedResultTypes))) {
      return rewriter.notifyMatchFailure(srcOp, "results failed to convert");
    }

    // Create new function with converted argument and result types.
    auto newFuncType = mlir::FunctionType::get(
        srcOp.getContext(), signatureConversion.getConvertedTypes(),
        convertedResultTypes);

    // Update the function in place.
    rewriter.startRootUpdate(srcOp);
    srcOp.setType(newFuncType);

    // Tell the rewriter to convert the region signature.
    TypeConverter &typeConverter = *getTypeConverter();
    if (failed(rewriter.convertRegionTypes(&srcOp.getBody(), typeConverter,
                                           &signatureConversion))) {
      return failure();
    }

    rewriter.finalizeRootUpdate(srcOp);
    return success();
  }
};

class GenericTypeConvert : public ConversionPattern {
 public:
  GenericTypeConvert(StringRef rootName, TypeConverter &converter,
                     MLIRContext *context, PatternBenefit benefit = 0)
      : ConversionPattern(converter, rootName, benefit, context) {}
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<NamedAttribute, 4> newAttr;
    llvm::append_range(newAttr, op->getAttrs());
    llvm::SmallVector<Type, 4> newResults;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResults))) {
      return rewriter.notifyMatchFailure(op, "result type conversion failed");
    }
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, newAttr, op->getSuccessors());
    for (Region &r : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(r, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());
      if (failed(getTypeConverter()->convertSignatureArgs(
              newRegion->getArgumentTypes(), result))) {
        return rewriter.notifyMatchFailure(op,
                                           "argument type conversion failed");
      }
      rewriter.applySignatureConversion(newRegion, result);
    }
    Operation *newOp = rewriter.createOperation(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertMHLOToLinalgOnTensorsPass
    : public ConvertMHLOToLinalgOnTensorsBase<
          ConvertMHLOToLinalgOnTensorsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, linalg::LinalgDialect,
                    mhlo::MhloDialect, shape::ShapeDialect, math::MathDialect,
                    tensor::TensorDialect, scf::SCFDialect,
                    memref::MemRefDialect, complex::ComplexDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    MLIRContext *context = &getContext();

    auto typeConverter = mhlo::createHloToLinalgSignedIntegerConverter();
    // NOTE: not using corresponding setupMHLOToFlowPatterns because the entire
    // MHLO dialects are marked illegal by this pass.
    // TODO: Collapse/rework all of these patterns once the consolidation
    // lands. There is little reason to have these so spread out.
    populateMHLOToFlowPatterns(context, patterns);
    chlo::PopulateDecomposeChloPatterns(context, &patterns);
    populateMHLOBroadcastingToLinalgPatterns(context, *typeConverter, patterns);
    populateMHLOToLinalgOnTensorsConversionPatterns(context, *typeConverter,
                                                    patterns);
    populateMHLOComplexToRealPatterns(context, *typeConverter, patterns);

    // Structural patterns (functions, cfg, terminators).
    patterns.insert<BuiltinFuncOpPattern>(*typeConverter, context);
    patterns.insert<GenericTypeConvert>(ReturnOp::getOperationName(),
                                        *typeConverter, context);
    patterns.insert<GenericTypeConvert>(CallOp::getOperationName(),
                                        *typeConverter, context);
    patterns.insert<GenericTypeConvert>(cf::CondBranchOp::getOperationName(),
                                        *typeConverter, context);
    patterns.insert<GenericTypeConvert>(cf::BranchOp::getOperationName(),
                                        *typeConverter, context);

    ConversionTarget target(getContext());
    auto isIllegalType = [&](Type t) { return !typeConverter->isLegal(t); };
    auto isLegallyTypedOp = [&](Operation *op) -> bool {
      for (Type type : op->getResultTypes()) {
        if (isIllegalType(type)) return false;
      }
      for (Type type : op->getOperandTypes()) {
        if (isIllegalType(type)) return false;
      }
      return true;
    };

    target.addIllegalDialect<chlo::HloClientDialect>();
    target.addIllegalDialect<mhlo::MhloDialect>();

    // Functions must have legal types.
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) {
      for (Type type : funcOp.getType().getInputs()) {
        if (isIllegalType(type)) return false;
      }
      for (Type type : funcOp.getType().getResults()) {
        if (isIllegalType(type)) return false;
      }
      for (Block &block : funcOp.body()) {
        for (Type type : block.getArgumentTypes()) {
          if (isIllegalType(type)) return false;
        }
      }
      return true;
    });

    // Let the rest fall through.
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<IREE::LinalgExt::IREELinalgExtDialect>();
    target.markUnknownOpDynamicallyLegal(isLegallyTypedOp);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void populateMHLOToLinalgOnTensorsConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet &patterns) {
  mhlo::populateHLOToLinalgConversionPattern(context, typeConverter, &patterns);
  // TODO(#5809): Drop ConcatenateOp lowering in favor of the upstream version
  //              then remove the PatternBenefit here
  patterns.insert<ScatterToDynamicUpdateSlice>(typeConverter, context);
  patterns.insert<ConcatenateOpConversion, FftOpConversion>(
      typeConverter, context, PatternBenefit(1000));
}

std::unique_ptr<OperationPass<FuncOp>> createMHLOToLinalgOnTensorsPass() {
  return std::make_unique<ConvertMHLOToLinalgOnTensorsPass>();
}

}  // namespace MHLO
}  // namespace iree_compiler
}  // namespace mlir
