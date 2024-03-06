// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO dot product ops to Linalg dialect.
// These patterns are separated out to their own file to save on the compilation
// times, given that we instantiate a large number of class templates here.

#include "compiler/plugins/input/StableHLO/stablehlo-iree/Conversion/LegalizeToLinalgUtils.h"
#include "compiler/plugins/input/StableHLO/stablehlo-iree/Conversion/Rewriters.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {
enum class DotOperationType {
  kVectorDot = 0,
  kMatrixVector,
  kVectorMatrix,
  kMatrixMatrix,
  kUnsupported
};

DotOperationType getDotOperationType(mlir::stablehlo::DotOp dotOp) {
  ArrayRef<int64_t> lhsShape =
      cast<ShapedType>(dotOp.getLhs().getType()).getShape();
  ArrayRef<int64_t> rhsShape =
      cast<ShapedType>(dotOp.getRhs().getType()).getShape();
  auto shapeMatches = [](int64_t a, int64_t b) {
    return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
  };
  if (lhsShape.size() == 1 && rhsShape.size() == 1 &&
      shapeMatches(lhsShape[0], rhsShape[0])) {
    return DotOperationType::kVectorDot;
  }
  if (lhsShape.size() == 2 && rhsShape.size() == 1 &&
      shapeMatches(lhsShape[1], rhsShape[0])) {
    return DotOperationType::kMatrixVector;
  }
  if (lhsShape.size() == 1 && rhsShape.size() == 2 &&
      shapeMatches(lhsShape[0], rhsShape[0])) {
    return DotOperationType::kVectorMatrix;
  }
  if (lhsShape.size() == 2 && rhsShape.size() == 2 &&
      shapeMatches(lhsShape[1], rhsShape[0])) {
    return DotOperationType::kMatrixMatrix;
  }
  return DotOperationType::kUnsupported;
}

SmallVector<Value, 2> getDotOpEmptyTensorDynSizes(OpBuilder &b, Location loc,
                                                  Value lhs, Value rhs,
                                                  DotOperationType type) {
  SmallVector<Value, 2> dynShape;
  switch (type) {
  case DotOperationType::kMatrixMatrix: {
    if (llvm::cast<ShapedType>(lhs.getType()).isDynamicDim(0))
      dynShape.push_back(b.create<tensor::DimOp>(loc, lhs, 0));
    if (llvm::cast<ShapedType>(rhs.getType()).isDynamicDim(1))
      dynShape.push_back(b.create<tensor::DimOp>(loc, rhs, 1));
    break;
  }
  case DotOperationType::kMatrixVector: {
    if (llvm::cast<ShapedType>(lhs.getType()).isDynamicDim(0))
      dynShape.push_back(b.create<tensor::DimOp>(loc, lhs, 0));
    break;
  }
  case DotOperationType::kVectorMatrix: {
    if (llvm::cast<ShapedType>(rhs.getType()).isDynamicDim(1))
      dynShape.push_back(b.create<tensor::DimOp>(loc, rhs, 1));
    break;
  }
  case DotOperationType::kVectorDot:
  case DotOperationType::kUnsupported:
    break;
  }
  return dynShape;
}

template <DotOperationType op_type, typename LinalgOp>
struct DotOpConversion final : OpConversionPattern<mlir::stablehlo::DotOp> {
  using OpConversionPattern<mlir::stablehlo::DotOp>::OpConversionPattern;
  using OpAdaptor = mlir::stablehlo::DotOp::Adaptor;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (failed(verifyHloOpBufferOrTensorSemantics(op))) {
      return failure();
    }
    if (getDotOperationType(op) != op_type)
      return failure();

    Location loc = op.getLoc();
    // Convert unsigned to signed. This works because signed and unsigned
    // integer matmul is the same operation in two's complement.
    auto outputType =
        cast<ShapedType>(getTypeConverter()->convertType(op.getType()));
    SmallVector<Value, 2> dynShape = getDotOpEmptyTensorDynSizes(
        rewriter, loc, adaptor.getLhs(), adaptor.getRhs(), op_type);
    Value emptyTensor =
        !sparse_tensor::getSparseTensorEncoding(outputType)
            ? getEmptyTensor(rewriter, loc, outputType, dynShape)
            : getEmptySparseTensor(rewriter, loc, outputType, dynShape);
    Value zeroTensor = fillTensorWithZeros(rewriter, loc, emptyTensor);
    rewriter.replaceOpWithNewOp<LinalgOp>(
        op, TypeRange{outputType},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{zeroTensor},
        linalg::getPrunedAttributeList(op));
    return success();
  }
};

struct DotGeneralBatchMatMulOpConversion final
    : OpConversionPattern<mlir::stablehlo::DotGeneralOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DotGeneralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (failed(verifyHloOpBufferOrTensorSemantics(op))) {
      return failure();
    }
    if (llvm::cast<RankedTensorType>(op.getType()).getRank() != 3) {
      return rewriter.notifyMatchFailure(op, "expected a batch matmul");
    }

    mlir::stablehlo::DotDimensionNumbersAttr dimNumbers =
        op.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
    ArrayRef<int64_t> lhsContractingDims =
        dimNumbers.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContractingDims =
        dimNumbers.getRhsContractingDimensions();
    if (lhsBatchingDims.size() != 1 || lhsBatchingDims[0] != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected lhs batching dimensions exactly {0}");
    }
    if (rhsBatchingDims.size() != 1 || rhsBatchingDims[0] != 0) {
      return rewriter.notifyMatchFailure(
          op, "expected rhs batching dimensions exactly {0}");
    }
    if (lhsContractingDims.size() != 1 || lhsContractingDims[0] != 2) {
      return rewriter.notifyMatchFailure(
          op, "expected lhs contracting dimensions exactly {2}");
    }
    if (rhsContractingDims.size() != 1 || rhsContractingDims[0] != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected rhs contracting dimensions exactly {1}");
    }

    Location loc = op.getLoc();
    // Convert unsigned to signed. This works because signed and unsigned
    // integer matmul is the same operation in two's complement.
    auto outputType =
        cast<ShapedType>(typeConverter->convertType(op.getType()));
    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, outputType, op, adaptor.getOperands());
    Value zeroTensor = fillTensorWithZeros(rewriter, loc, emptyTensor);
    Operation *linalgOp = rewriter.create<linalg::BatchMatmulOp>(
        loc, /*resultTensorTypes=*/TypeRange{outputType},
        /*inputs=*/ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        /*outputBuffers=*/ValueRange{zeroTensor},
        linalg::getPrunedAttributeList(op));

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct DotGeneralOpConversion final
    : OpConversionPattern<mlir::stablehlo::DotGeneralOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::stablehlo::DotGeneralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (failed(verifyHloOpBufferOrTensorSemantics(op))) {
      return failure();
    }

    // Get various dimension iterator information
    mlir::stablehlo::DotDimensionNumbersAttr dimNumbers =
        op.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
    ArrayRef<int64_t> lhsContractingDims =
        dimNumbers.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContractingDims =
        dimNumbers.getRhsContractingDimensions();

    // Get shape information and initialize output
    assert(lhsContractingDims.size() == rhsContractingDims.size() &&
           "number of contracting dims must be equal");
    size_t numContracting = lhsContractingDims.size();
    // Convert unsigned to signed. This works because signed and unsigned
    // integer matmul is the same operation in two's complement.
    auto outputType =
        cast<ShapedType>(typeConverter->convertType(op.getType()));
    size_t targetRank = outputType.getRank();
    size_t totalLoopCount = numContracting + targetRank;

    int64_t lhsRank =
        llvm::cast<ShapedType>(adaptor.getLhs().getType()).getRank();
    size_t lhsExtraDims =
        lhsRank - lhsBatchingDims.size() - lhsContractingDims.size();
    int64_t rhsRank =
        llvm::cast<ShapedType>(adaptor.getRhs().getType()).getRank();

    Location loc = op.getLoc();
    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, outputType, op, adaptor.getOperands());
    Value zeroTensor = fillTensorWithZeros(rewriter, loc, emptyTensor);
    SmallVector<AffineMap, 3> indexingMaps;

    auto getMap = [&](int64_t rank, ArrayRef<int64_t> batchingDims,
                      ArrayRef<int64_t> contractingDims, size_t extraDims) {
      llvm::SmallVector<AffineExpr> indices(rank);
      for (const auto &i : llvm::enumerate(batchingDims)) {
        indices[i.value()] = rewriter.getAffineDimExpr(i.index());
      }
      for (const auto &i : llvm::enumerate(contractingDims)) {
        indices[i.value()] = rewriter.getAffineDimExpr(i.index() + targetRank);
      }
      for (int i = 0; i < rank; ++i) {
        if (!indices[i]) {
          indices[i] = rewriter.getAffineDimExpr(extraDims++);
        }
      }
      indexingMaps.push_back(AffineMap::get(/*dimCount=*/totalLoopCount,
                                            /*symbolCount=*/0, indices,
                                            op->getContext()));
    };
    getMap(lhsRank, lhsBatchingDims, lhsContractingDims,
           lhsBatchingDims.size());
    getMap(rhsRank, rhsBatchingDims, rhsContractingDims,
           rhsBatchingDims.size() + lhsExtraDims);

    {
      SmallVector<AffineExpr> dimExprs;
      dimExprs.reserve(targetRank);
      for (unsigned i = 0; i < targetRank; ++i)
        dimExprs.push_back(rewriter.getAffineDimExpr(i));
      indexingMaps.push_back(AffineMap::get(/*dimCount=*/totalLoopCount,
                                            /*symbolCount=*/0, dimExprs,
                                            op.getContext()));
    }

    Operation *linalgOp = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/TypeRange{outputType},
        /*inputs=*/ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        /*outputBuffers=*/ValueRange{zeroTensor}, indexingMaps,
        getParallelAndReductionIterators(
            /*nLoops=*/totalLoopCount,
            /*nReduction=*/numContracting),
        [](OpBuilder &b, Location loc, ValueRange) {
          ImplicitLocOpBuilder builder(loc, b);
          linalg::MatmulOp::regionBuilder(builder, *b.getInsertionBlock(), {});
        },
        linalg::getPrunedAttributeList(op));

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

} // namespace

namespace detail {
void populateStableHloDotProdToLinalgConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    RewritePatternSet *patterns) {
  // Ensure specialized patterns are higher priority than their generic
  // versions.
  patterns
      ->add<DotOpConversion<DotOperationType::kMatrixMatrix, linalg::MatmulOp>,
            DotOpConversion<DotOperationType::kMatrixVector, linalg::MatvecOp>,
            DotOpConversion<DotOperationType::kVectorMatrix, linalg::VecmatOp>,
            DotOpConversion<DotOperationType::kVectorDot, linalg::DotOp>,
            DotGeneralBatchMatMulOpConversion>(typeConverter, context,
                                               PatternBenefit(2));
  patterns->add<DotGeneralOpConversion>(typeConverter, context,
                                        PatternBenefit(1));
}
} // namespace detail
} // namespace mlir::iree_compiler::stablehlo
