// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/TransformUtils.h"
#include "iree/compiler/Conversion/PassDetail.h"
#include "iree/compiler/Conversion/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-linalg-vectorization"

static llvm::cl::list<int64_t> clVectorUnrollShape(
    "iree-codegen-llvm-vector-unroll-shape",
    llvm::cl::desc("Comma-separated list of vector sizes for vector unrolling"),
    llvm::cl::CommaSeparated);

namespace mlir {
namespace iree_compiler {

static Optional<SmallVector<int64_t, 4>> getShape(mlir::Operation *op) {
  auto unrollOp = dyn_cast<VectorUnrollOpInterface>(op);
  if (!unrollOp) return None;
  SmallVector<int64_t, 4> shape(clVectorUnrollShape.begin(),
                                clVectorUnrollShape.end());
  shape.resize(unrollOp.getShapeForUnroll()->size(),
               std::numeric_limits<int64_t>::max());
  return shape;
}

namespace {
struct LinalgVectorizationPass
    : public PassWrapper<LinalgVectorizationPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, AffineDialect>();
  }
  void runOnFunction() override;
};
}  // namespace

void LinalgVectorizationPass::runOnFunction() {
  FuncOp funcOp = getOperation();
  MLIRContext *context = &getContext();
  // Apply vectorization patterns.
  {
    OwningRewritePatternList vectorizationPatterns(&getContext());
    linalg::insertVectorizationPatterns<linalg::GenericOp,
                                        linalg::ContractionOpInterface>(
        vectorizationPatterns, linalg::LinalgVectorizationOptions(),
        linalg::LinalgTransformationFilter(ArrayRef<Identifier>(
            Identifier::get(getWorkgroupMarker(), context))));
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(vectorizationPatterns));

    LLVM_DEBUG({
      llvm::dbgs() << "--- After Vectorization ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  // TODO: This should be a folding of Add into Contract in core but while they
  // live in different dialects, it is not possible without unnatural
  // dependencies.
  funcOp.walk([&](Operation *op) {
    if (auto contract = canonicalizeContractionAdd(op))
      op->replaceAllUsesWith(contract);
  });

  // Apply unrolling patterns.
  {
    OwningRewritePatternList vectorUnrollPatterns(&getContext());
    vectorUnrollPatterns.insert<vector::UnrollVectorPattern>(
        context, vector::UnrollVectorOptions().setNativeShapeFn(getShape));
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(vectorUnrollPatterns));

    OwningRewritePatternList canonicalizationPatterns1(&getContext());
    vector::populateVectorToVectorCanonicalizationPatterns(
        canonicalizationPatterns1);
    vector::populateVectorToVectorTransformationPatterns(
        canonicalizationPatterns1);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(canonicalizationPatterns1));

    OwningRewritePatternList canonicalizationPatterns2(&getContext());
    vector::populateVectorSlicesLoweringPatterns(canonicalizationPatterns2);
    (void)applyPatternsAndFoldGreedily(funcOp,
                                       std::move(canonicalizationPatterns2));

    LLVM_DEBUG({
      llvm::dbgs() << "--- After Vector Unroll ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
  // Apply hoisting patterns.
  {
    linalg::hoistRedundantVectorTransfersOnTensor(funcOp);

    LLVM_DEBUG({
      llvm::dbgs() << "--- After Hoisting ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
}

std::unique_ptr<OperationPass<FuncOp>> createLinalgVectorizePass() {
  return std::make_unique<LinalgVectorizationPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
