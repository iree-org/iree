// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Returns the add-reduction of the input 2D tensor `matrix` along one of the
// two dimensions. The `parallelDim` argument specifies which of the two
// dimensions (0 or 1) is the parallel (i.e. not reduction) dimension.
// The input `matrix`'s element type is assumed to be signless integer.
// The result's element type is `accElTy`. The input elements are sign-extended
// to `accElTy` before being added.
Value additiveReductionLeaving1ParallelDim(PatternRewriter &rewriter,
                                           Location loc, Value matrix,
                                           int parallelDim, Type accElTy) {
  RankedTensorType matrixType = matrix.getType().cast<RankedTensorType>();
  assert(matrixType.getRank() == 2);
  assert(parallelDim == 0 || parallelDim == 1);
  // Create the accumulator.
  int64_t dstStaticSize = matrixType.getShape()[parallelDim];
  SmallVector<Value> dstDynSizes;
  if (dstStaticSize == ShapedType::kDynamicSize) {
    dstDynSizes.push_back(
        rewriter.create<tensor::DimOp>(loc, matrix, parallelDim));
  }
  Value initAcc =
      rewriter
          .create<linalg::InitTensorOp>(
              loc, dstDynSizes, ArrayRef<int64_t>{dstStaticSize}, accElTy)
          .getResult();
  // Zero-fill the accumulator.
  Value zeroInt =
      rewriter.create<arith::ConstantIntOp>(loc, 0, accElTy).getResult();
  Value zeroAcc =
      rewriter.create<linalg::FillOp>(loc, zeroInt, initAcc).getResult(0);
  // Create the indexing maps for the generic.
  MLIRContext *context = rewriter.getContext();
  AffineExpr expr[2];
  bindDims(context, expr[0], expr[1]);
  AffineExpr parallelExpr = expr[parallelDim];
  AffineMap mapIdentity = AffineMap::get(2, 0, expr, context);
  AffineMap mapToParallelDim = AffineMap::get(2, 0, parallelExpr, context);
  SmallVector<AffineMap> indexingMaps{mapIdentity, mapToParallelDim};
  // Create the iterators for the generic.
  auto iterator = [=](int dim) -> StringRef {
    return dim == parallelDim ? "parallel" : "reduction";
  };
  SmallVector<StringRef> iterators{iterator(0), iterator(1)};
  // Create the generic.
  return rewriter
      .create<linalg::GenericOp>(
          loc, zeroAcc.getType(), ValueRange{matrix}, ValueRange{zeroAcc},
          indexingMaps, iterators,
          [=](OpBuilder &b, Location loc, ValueRange args) {
            Value matrixEl = args[0];
            // Sign-extend the input matrix elem to accElTy before adding.
            Value promotedMatrixEl =
                b.create<arith::ExtSIOp>(loc, accElTy, matrixEl);
            Value accEl = args[1];
            Value sum = b.create<arith::AddIOp>(loc, promotedMatrixEl, accEl);
            b.create<linalg::YieldOp>(loc, sum);
          })
      .getResult(0);
}

bool isConstantZero(Value val) {
  auto constIntOp = val.getDefiningOp<arith::ConstantIntOp>();
  return constIntOp && constIntOp.value() == 0;
}

// Pattern lowering quantized_matmul to matmul.
// Always succeeds.
//
// This is implementing the math explained in Section 2.3 of
// https://arxiv.org/abs/1712.05877.
struct QuantizedMatmulToMatmul
    : public OpRewritePattern<linalg::QuantizedMatmulOp> {
  using OpRewritePattern<linalg::QuantizedMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::QuantizedMatmulOp quantizedMatmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = quantizedMatmulOp.getLoc();
    ValueRange inputs = quantizedMatmulOp.inputs();
    assert(inputs.size() == 4);
    Value lhs = inputs[0];
    Value rhs = inputs[1];
    Value lhsZp = inputs[2];
    Value rhsZp = inputs[3];
    ValueRange outputs = quantizedMatmulOp.outputs();
    // Compute the matmul part.
    Value acc = outputs[0];
    Value matmul = rewriter
                       .create<linalg::MatmulOp>(loc, ValueRange{lhs, rhs},
                                                 ValueRange{acc})
                       .getResult(0);
    bool lhsZpIsConstantZero = isConstantZero(lhsZp);
    bool rhsZpIsConstantZero = isConstantZero(rhsZp);
    if (lhsZpIsConstantZero && rhsZpIsConstantZero) {
      // Easy case: both zero points are constant zeros, so the quantized_matmul
      // was just a matmul all along.
      rewriter.replaceOp(quantizedMatmulOp, matmul);
      return success();
    }
    // Create the result. No need to zero-fill it as we will overwrite it.
    ShapedType accType = acc.getType().cast<ShapedType>();
    auto accDynShape = linalg::getDynOperands(loc, acc, rewriter);
    Value initResult = rewriter.create<linalg::InitTensorOp>(
        loc, accDynShape, accType.getShape(), accType.getElementType());
    // Create the indexing maps for the generic.
    MLIRContext *context = rewriter.getContext();
    AffineExpr m, n;
    bindDims(context, m, n);
    AffineMap mapToNone = AffineMap::get(2, 0, context);
    AffineMap mapToRowDim = AffineMap::get(2, 0, m, context);
    AffineMap mapToColumnDim = AffineMap::get(2, 0, n, context);
    AffineMap mapIdentity = AffineMap::get(2, 0, {m, n}, context);
    SmallVector<AffineMap> indexingMaps;
    SmallVector<Value> ins;
    auto addInput = [&](Value val, AffineMap map) -> int {
      ins.push_back(val);
      indexingMaps.push_back(map);
      return ins.size() - 1;
    };
    int indexOfMatmulInput = addInput(matmul, mapIdentity);
    int indexOfLhsSumsInput = 0;
    int indexOfLhsZpInput = 0;
    int indexOfRhsSumsInput = 0;
    int indexOfRhsZpInput = 0;
    int indexOfLhsZpTimesRhsZpTimesKSizeInput = 0;
    Type accElTy = accType.getElementType();
    if (!rhsZpIsConstantZero) {
      Value lhsSums =
          additiveReductionLeaving1ParallelDim(rewriter, loc, lhs, 0, accElTy);
      indexOfLhsSumsInput = addInput(lhsSums, mapToRowDim);
      indexOfRhsZpInput = addInput(rhsZp, mapToNone);
    }
    if (!lhsZpIsConstantZero) {
      Value rhsSums =
          additiveReductionLeaving1ParallelDim(rewriter, loc, rhs, 1, accElTy);
      indexOfRhsSumsInput = addInput(rhsSums, mapToColumnDim);
      indexOfLhsZpInput = addInput(lhsZp, mapToNone);
    }
    if (!lhsZpIsConstantZero && !rhsZpIsConstantZero) {
      Value lhsZpTimesRhsZp = rewriter.create<arith::MulIOp>(loc, lhsZp, rhsZp);
      Value kSize = rewriter.create<arith::IndexCastOp>(
          loc, accElTy, rewriter.create<tensor::DimOp>(loc, lhs, 1));
      Value lhsZpTimesRhsZpTimesKSize =
          rewriter.create<arith::MulIOp>(loc, lhsZpTimesRhsZp, kSize);
      indexOfLhsZpTimesRhsZpTimesKSizeInput =
          addInput(lhsZpTimesRhsZpTimesKSize, mapToNone);
    }
    // Add the indexing map for the initResult 'output' even though it's unused.
    indexingMaps.push_back(mapIdentity);
    // Create the generic putting all the terms together.
    SmallVector<StringRef> iterators{"parallel", "parallel"};
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        quantizedMatmulOp, acc.getType(), ins, ValueRange{initResult},
        indexingMaps, iterators,
        [=](OpBuilder &b, Location loc, ValueRange args) {
          Value matmulEl = args[indexOfMatmulInput];
          Value lhsSumsEl = args[indexOfLhsSumsInput];
          Value rhsSumsEl = args[indexOfRhsSumsInput];
          Value lhsZp = args[indexOfLhsZpInput];
          Value rhsZp = args[indexOfRhsZpInput];
          Value lhsZpTimesRhsZpTimesKSize =
              args[indexOfLhsZpTimesRhsZpTimesKSizeInput];
          Value result = matmulEl;
          // If the rhs zero-point is not a constant zero, we need to add it
          // times the sums along rows of lhs.
          if (!rhsZpIsConstantZero) {
            Value lhsSumsElTimesRhsZp =
                b.create<arith::MulIOp>(loc, lhsSumsEl, rhsZp);
            result = b.create<arith::SubIOp>(loc, result, lhsSumsElTimesRhsZp);
          }
          // If the lhs zero-point is not a constant zero, we need to add it
          // times the sums along columns of rhs.
          if (!lhsZpIsConstantZero) {
            Value rhsSumsElTimesLhsZp =
                b.create<arith::MulIOp>(loc, rhsSumsEl, lhsZp);
            result = b.create<arith::SubIOp>(loc, result, rhsSumsElTimesLhsZp);
          }
          // Add the final correction term, if neither zero-point is cst zero.
          if (!lhsZpIsConstantZero && !rhsZpIsConstantZero) {
            result =
                b.create<arith::AddIOp>(loc, result, lhsZpTimesRhsZpTimesKSize);
          }
          b.create<linalg::YieldOp>(loc, result);
        });
    return success();
  }
};

/// Pass that lowers quantized_matmul to matmul.
struct LinalgQuantizedMatmulToMatmulPass
    : public LinalgQuantizedMatmulToMatmulPassBase<
          LinalgQuantizedMatmulToMatmulPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);
    patterns.add<QuantizedMatmulToMatmul>(context);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
createLinalgQuantizedMatmulToMatmulPass() {
  return std::make_unique<LinalgQuantizedMatmulToMatmulPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
