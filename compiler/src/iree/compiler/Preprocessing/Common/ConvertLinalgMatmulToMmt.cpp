// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>

#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/Optional.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

namespace {

// Converts linalg.matmul to an linalg.transpose + linalg.matmul.
// Such that matrix B layout changes to col major.
class LinalgMatmulOpToLinalgMmtPattern final
    : public OpRewritePattern<linalg::MatmulOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();
    Value lhs = matmulOp.getDpsInputOperand(0)->get();
    Value rhs = matmulOp.getDpsInputOperand(1)->get();
    Value acc = matmulOp.getDpsInitOperand(0)->get();
    if (dyn_cast<linalg::TransposeOp>(rhs.getDefiningOp())) {
      return failure();
    }
    auto rhsType = rhs.getType().cast<RankedTensorType>();
    auto rhsShape = rhsType.getShape();
    auto rhsElemType = rhsType.getElementType();
    SmallVector<int64_t, 2> transposedRhsShape = {rhsShape[1], rhsShape[0]};

    // GenericOp
    int64_t nloops = rhsShape.size();
    AffineExpr mDim, nDim;
    bindDims(getContext(), mDim, nDim);
    auto inputMap = AffineMap::get(2, 0, {mDim, nDim}, getContext());
    auto packedMap = AffineMap::get(2, 0, {nDim, mDim}, getContext());
    SmallVector<AffineMap> indexingMaps = {inputMap, packedMap};

    Value transposedRhs =
        rewriter.create<tensor::EmptyOp>(loc, transposedRhsShape, rhsElemType);
    SmallVector<utils::IteratorType> loopAttributeTypes(
        nloops, utils::IteratorType::parallel);

    Value packedRhs =
        rewriter
            .create<linalg::GenericOp>(
                loc, transposedRhs.getType(),
                /*inputs=*/rhs, /*outputs=*/transposedRhs, indexingMaps,
                loopAttributeTypes,
                [&](OpBuilder &nestedBuilder, Location nestedLoc,
                    ValueRange args) {
                  nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
                })
            .getResult(0);

    // TransposeOp
    Value initOp = rewriter.create<tensor::EmptyOp>(loc, rhsShape, rhsElemType);
    SmallVector<int64_t, 2> transposedPerm = {1, 0};
    Value transposePackedRhs =
        rewriter
            .create<linalg::TransposeOp>(loc, packedRhs, initOp, transposedPerm)
            .getResults()[0];

    // MatmulOp
    Value packedMatmul =
        rewriter
            .create<linalg::MatmulOp>(loc, matmulOp.getResult(0).getType(),
                                      ArrayRef<Value>{lhs, transposePackedRhs},
                                      ArrayRef<Value>{acc})
            .getResult(0);
    rewriter.replaceOp(matmulOp, packedMatmul);
    return success();
  }
};

struct ConvertLinalgMatmulToMmtPass
    : public ConvertLinalgMatmulToMmtBase<ConvertLinalgMatmulToMmtPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    // Main pattern.
    {
      RewritePatternSet patterns(&getContext());
      patterns.insert<LinalgMatmulOpToLinalgMmtPattern>(context);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createConvertLinalgMatmulToMmtPass() {
  return std::make_unique<ConvertLinalgMatmulToMmtPass>();
}

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
