// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "iree/compiler/InputConversion/Common/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

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
    ImplicitLocOpBuilder builder(loc, rewriter);
    ValueRange inputs = quantizedMatmulOp.getInputs();
    assert(inputs.size() == 4);
    Value lhs = inputs[0];
    Value rhs = inputs[1];
    Value lhsZp = inputs[2];
    Value rhsZp = inputs[3];
    ValueRange outputs = quantizedMatmulOp.getOutputs();
    // Compute the matmul part.
    Value acc = outputs[0];
    Value matmul =
        builder.create<linalg::MatmulOp>(ValueRange{lhs, rhs}, ValueRange{acc})
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
    ShapedType accType = llvm::cast<ShapedType>(acc.getType());
    Value initResult = builder.create<tensor::EmptyOp>(
        tensor::getMixedSizes(builder, loc, acc), accType.getElementType());
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
      Value lhsSums = sumReduceDimensionSubset(builder, lhs, accElTy,
                                               /*is_reduction=*/{false, true});
      indexOfLhsSumsInput = addInput(lhsSums, mapToRowDim);
      indexOfRhsZpInput = addInput(rhsZp, mapToNone);
    }
    if (!lhsZpIsConstantZero) {
      Value rhsSums = sumReduceDimensionSubset(builder, rhs, accElTy,
                                               /*is_reduction=*/{true, false});
      indexOfRhsSumsInput = addInput(rhsSums, mapToColumnDim);
      indexOfLhsZpInput = addInput(lhsZp, mapToNone);
    }
    if (!lhsZpIsConstantZero && !rhsZpIsConstantZero) {
      Value lhsZpTimesRhsZp = builder.create<arith::MulIOp>(lhsZp, rhsZp);
      Value kSize = rewriter.create<arith::IndexCastOp>(
          loc, accElTy, builder.create<tensor::DimOp>(lhs, 1));
      Value lhsZpTimesRhsZpTimesKSize =
          builder.create<arith::MulIOp>(lhsZpTimesRhsZp, kSize);
      indexOfLhsZpTimesRhsZpTimesKSizeInput =
          addInput(lhsZpTimesRhsZpTimesKSize, mapToNone);
    }
    // Add the indexing map for the initResult 'output' even though it's unused.
    indexingMaps.push_back(mapIdentity);
    // Create the generic putting all the terms together.
    SmallVector<utils::IteratorType> iterators{utils::IteratorType::parallel,
                                               utils::IteratorType::parallel};
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
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLinalgQuantizedMatmulToMatmulPass() {
  return std::make_unique<LinalgQuantizedMatmulToMatmulPass>();
}

} // namespace mlir::iree_compiler
