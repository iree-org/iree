// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- LinalgFusion.cpp - Fuse Linalg operations within a dispatch region--===//
//
// Fuses all Linalg operations with a dispatch region into a single linalg
// operation.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Pattern to implement the fusion. Only fuses op with its producer if the
/// latter has a single use (this op).
// TODO(ravishankarm): Generalize this to handle more valid fusion cases.
struct IREEFuseGenericTensorOps : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  PatternMatchResult matchAndRewrite(linalg::GenericOp op,
                                     PatternRewriter &rewriter) const override;
};

/// Fuses linalg operations on tensors in dispatch function. For now does only
/// producer consumer fusion.
struct IREELinalgFusionPass : public FunctionPass<IREELinalgFusionPass> {
  void runOnFunction() override;
};
}  // namespace

PatternMatchResult IREEFuseGenericTensorOps::matchAndRewrite(
    linalg::GenericOp op, PatternRewriter &rewriter) const {
  if (!op.hasTensorSemantics()) return matchFailure();
  for (unsigned i = 0, e = op.getOperation()->getNumOperands(); i != e; ++i) {
    auto producerOp = dyn_cast_or_null<linalg::LinalgOp>(
        op.getOperation()->getOperand(i).getDefiningOp());
    if (!producerOp || producerOp.getOperation()->getNumResults() != 1)
      continue;
    bool isDeadIfUsed = producerOp.getOperation()->getResult(0).hasOneUse();
    if (Optional<linalg::LinalgOp> fusedOp = linalg::fuseTensorOps(
            rewriter, producerOp, cast<linalg::LinalgOp>(op.getOperation()),
            i)) {
      rewriter.replaceOp(op, fusedOp.getValue().getOperation()->getResults());
      if (isDeadIfUsed) rewriter.eraseOp(producerOp);
      return matchSuccess();
    }
  }
  return matchFailure();
}

void IREELinalgFusionPass::runOnFunction() {
  OwningRewritePatternList patterns;
  Operation *op = getOperation();
  patterns.insert<IREEFuseGenericTensorOps>(op->getContext());
  applyPatternsGreedily(op->getRegions(), patterns);
}

std::unique_ptr<OpPassBase<FuncOp>> createLinalgFusionPass() {
  return std::make_unique<IREELinalgFusionPass>();
}

static PassRegistration<IREELinalgFusionPass> pass(
    "iree-linalg-fusion", "Fuse Linalg operations within a dispatch region");
}  // namespace iree_compiler
}  // namespace mlir
