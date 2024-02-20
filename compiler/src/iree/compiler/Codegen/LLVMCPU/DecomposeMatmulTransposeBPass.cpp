// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {
struct DecomposeMatmulTransposeBPass
    : public DecomposeMatmulTransposeBBase<DecomposeMatmulTransposeBPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<linalg::MatmulTransposeBOp> candidates;
    funcOp->walk([&](linalg::MatmulTransposeBOp op) {
      Value rhs = op.getInputs()[1];
      auto type = dyn_cast<RankedTensorType>(rhs.getType());
      if (type.hasStaticShape() && type.getShape()[0] % 16 == 0 &&
          type.getShape()[1] == 16)
        candidates.push_back(op);
    });

    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);
    for (auto op : candidates) {
      Location loc = op.getLoc();
      rewriter.setInsertionPointAfter(op);
      Value lhs = op.getInputs()[0];
      Value rhs = op.getInputs()[1];
      Value out = op.getOutputs()[0];

      auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
      SmallVector<OpFoldResult> transpShape =
          tensor::getMixedSizes(rewriter, loc, rhs);
      std::swap(transpShape[0], transpShape[1]);
      Value init = rewriter.create<tensor::EmptyOp>(loc, transpShape,
                                                    rhsType.getElementType());
      SmallVector<int64_t> perm = {1, 0};
      Value transp = rewriter.create<linalg::TransposeOp>(loc, rhs, init, perm)
                         .getResults()[0];

      auto options = scf::SCFTilingOptions().setTileSizeComputationFunction(
          [](OpBuilder &builder, Operation *op) {
            SmallVector<OpFoldResult> tileSizes(2, builder.getIndexAttr(16));

            return tileSizes;
          });
      FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
          rewriter, cast<TilingInterface>(transp.getDefiningOp()), options);
      transp = tilingResult->replacements[0];

      Value matmul =
          rewriter
              .create<linalg::MatmulOp>(loc, TypeRange{out.getType()},
                                        ValueRange{lhs, transp}, out)
              .getResults()[0];
      rewriter.replaceOp(op, matmul);
    }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createDecomposeMatmulTransposeBPass() {
  return std::make_unique<DecomposeMatmulTransposeBPass>();
}

} // namespace mlir::iree_compiler

