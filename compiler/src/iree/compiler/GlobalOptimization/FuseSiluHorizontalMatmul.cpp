// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-global-opt-fuse-dequantization-matmul"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

namespace {

static LogicalResult isGenericSigmoidOp(linalg::GenericOp genericOp) {
  // Check that all loops are parallel
  unsigned numLoops = genericOp.getNumLoops();
  unsigned numParallelLoops = genericOp.getNumParallelLoops();
  if (numLoops != numParallelLoops) {
    return failure();
  }

  // Work back from linalg.yield and check body of genericOp.
  // The genericOp should yield the result of an arith.divf,
  // preceded by an arith.addf, arith.exp, and arith.negf
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  Value producerOutput;
  Operation *producer;

  // Producer of linalg.yield op is arith.divf
  {
    producerOutput = yieldOp->getOperand(0);
    producer = producerOutput.getDefiningOp<arith::DivFOp>();
    if (!producer || producer->getNumOperands() == 0) {
      return failure();
    }
  }

  // Producer of arith.divf op is arith.addf
  {
    producerOutput = producer->getOperand(1);
    producer = producerOutput.getDefiningOp<arith::AddFOp>();
    if (!producer || producer->getNumOperands() == 0) {
      return failure();
    }
  }

  // Producer of arith.addf op is math.exp
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp<math::ExpOp>();
    if (!producer || producer->getNumOperands() == 0) {
      return failure();
    }
  }

  // Producer of math.expf op is arith.negf
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp<arith::NegFOp>();
    if (!producer) {
      return failure();
    }
  }

  return success();
}

static LogicalResult isGenericMultiplyOp(linalg::GenericOp genericOp) {
  // Check that all loops are parallel
  unsigned numLoops = genericOp.getNumLoops();
  unsigned numParallelLoops = genericOp.getNumParallelLoops();
  if (numLoops != numParallelLoops) {
    return failure();
  }

  // Work back from linalg.yield and check body of genericOp.
  // The genericOp should yield the result of an arith.mulf.
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  Value producerOutput;
  Operation *producer;

  // Producer of linalg.yield op is input
  {
    producerOutput = yieldOp->getOperand(0);
    producer = producerOutput.getDefiningOp<arith::MulFOp>();
    if (!producer || producer->getNumOperands() == 0) {
      return failure();
    }
  }

  return success();
}

// This pattern does a basic fusion of two matmuls and three linalg.generics.
// The pattern matches only on a DAG representing:
// output = Silu(matmul(A, B) * matmul(A, C).
class FuseSiluHorizontalMatmulPattern final
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Fail if not the right type of linalg.generic
    if (isGenericMultiplyOp(genericOp).failed())
      return failure();

    auto genericMultiplyOp =
        genericOp->getOperand(0).getDefiningOp<linalg::GenericOp>();
    auto matmulOp1 =
        genericOp->getOperand(1).getDefiningOp<linalg::MatmulTransposeBOp>();
    if (!genericMultiplyOp || !matmulOp1 ||
        isGenericMultiplyOp(genericMultiplyOp).failed())
      return failure();

    auto genericSigmoidOp =
        genericMultiplyOp->getOperand(0).getDefiningOp<linalg::GenericOp>();
    auto matmulOp2 = genericMultiplyOp->getOperand(1)
                         .getDefiningOp<linalg::MatmulTransposeBOp>();
    if (!genericSigmoidOp || !matmulOp2 ||
        isGenericSigmoidOp(genericSigmoidOp).failed())
      return failure();
    auto genericSigmoidOpInput =
        genericSigmoidOp->getOperand(0)
            .getDefiningOp<linalg::MatmulTransposeBOp>();
    if (!genericSigmoidOpInput || genericSigmoidOpInput != matmulOp2)
      return failure();

    auto matmulOp1Input = matmulOp1->getOperand(0);
    auto matmulOp2Input = matmulOp2->getOperand(0);
    if (!matmulOp1Input || !matmulOp2Input || matmulOp1Input != matmulOp2Input)
      return failure();

    SmallVector<Operation *> opsToFuse = {
        matmulOp2, genericSigmoidOp, genericMultiplyOp, matmulOp1, genericOp};

    // Fail if matmul is already in a dispatch.
    for (Operation *op : opsToFuse) {
      if (!IREE::Flow::isNonNullAndOutsideDispatch(op)) {
        return failure();
      }
    }

    auto result = wrapConsecutiveOpsInDispatchRegion(rewriter, opsToFuse);
    if (failed(result)) {
      return failure();
    }

    return success();
  }
};

struct FuseSiluHorizontalMatmulPass
    : public FuseSiluHorizontalMatmulBase<FuseSiluHorizontalMatmulPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREE::Flow::FlowDialect,
                    math::MathDialect>();
  }
  FuseSiluHorizontalMatmulPass() {}
  FuseSiluHorizontalMatmulPass(const FuseSiluHorizontalMatmulPass &pass)
      : FuseSiluHorizontalMatmulPass() {}

  void runOnOperation() override;
};

} // namespace

void FuseSiluHorizontalMatmulPass::runOnOperation() {
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  patterns.insert<FuseSiluHorizontalMatmulPattern>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFuseSiluHorizontalMatmulPass() {
  return std::make_unique<FuseSiluHorizontalMatmulPass>();
}

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
