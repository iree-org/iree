// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"

#include <sstream>

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

namespace {

template <typename T, unsigned int N>
mlir::raw_ostream& operator<< (mlir::raw_ostream& s, const SmallVector<T, N>& v) {
  s << "[ ";
  for (const auto& e : v) {
    s << e << " ";
  }
  s << "]";
  return s;
}

// Converts linalg.conv_2d_input_nhwc_filter_nhwc op to linalg.matmul
class LiftGenericToTransposeBatchMatmul
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    llvm::dbgs() << "LiftGenericToTransposeBatchMatmul on " << genericOp << "\n\n\n";

    FailureOr<linalg::ContractionDimensions> contractionDims = linalg::inferContractionDims(genericOp);
    if (failed(contractionDims)) {
      llvm::dbgs() << "failed to infer contraction dims\n\n";
      return failure();
    }

    llvm::dbgs() << "batch: " << contractionDims->batch << "\n";
    llvm::dbgs() << "m: " << contractionDims->m << "\n";
    llvm::dbgs() << "n: " << contractionDims->n << "\n";
    llvm::dbgs() << "k: " << contractionDims->k << "\n";

    return failure();
  }
};

struct LiftGenericToTransposeBatchMatmulPass
    : public LiftGenericToTransposeBatchMatmulBase<
          LiftGenericToTransposeBatchMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<LiftGenericToTransposeBatchMatmul>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> createLiftGenericToTransposeBatchMatmulPass() {
  return std::make_unique<LiftGenericToTransposeBatchMatmulPass>();
}

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
