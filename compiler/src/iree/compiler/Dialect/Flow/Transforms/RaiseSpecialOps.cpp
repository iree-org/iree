// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using transform_ext::StructuredOpMatcher;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// Method to match a transpose operation.
static bool match2DTranspose(linalg::LinalgOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return false;
  }
  // Check only for 2D ops.
  if (genericOp.getNumLoops() != 2 ||
      genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
    return false;
  }
  // Check for transpose map.
  AffineExpr d0, d1;
  MLIRContext *context = genericOp.getContext();
  bindDims(context, d0, d1);
  SmallVector<AffineMap> expectedMaps = {
      AffineMap::get(2, 0, {d0, d1}, context),
      AffineMap::get(2, 0, {d1, d0}, context)};
  if (genericOp.getIndexingMapsArray() != expectedMaps) {
    return false;
  }

  Block *body = genericOp.getBlock();
  if (!llvm::hasSingleElement(*body)) {
    return false;
  }
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  auto blockArg = yieldOp.getOperand(0).dyn_cast<BlockArgument>();
  if (!blockArg || blockArg.getOwner() != body ||
      blockArg.getArgNumber() != 0) {
    return false;
  }
  return true;
}

// Method to match a linalg.matmul(a, linalg.transpose(b)). Returns `b` on
// success.
std::optional<Value> matchATransposeBMatmul(linalg::LinalgOp matmulOp) {
  if (!isa<linalg::MatmulOp>(matmulOp.getOperation())) {
    return std::nullopt;
  }
  auto rhs = matmulOp.getDpsInputOperand(1);
  auto genericOp = rhs->get().getDefiningOp<linalg::GenericOp>();
  if (genericOp && match2DTranspose(genericOp)) {
    return genericOp.getDpsInputOperand(0)->get();
  }
  return std::nullopt;
}

struct RaiseSpecialOpsPass : public RaiseSpecialOpsBase<RaiseSpecialOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override {
    SmallVector<std::pair<linalg::LinalgOp, Value>> softmaxRoots;
    SmallVector<std::pair<linalg::MatmulOp, Value>> transposeMatmulRoots;
    getOperation()->walk([&](linalg::LinalgOp op) {
      {
        transform_ext::MatcherContext matcherContext;
        transform_ext::StructuredOpMatcher *maxReduction;
        transform_ext::StructuredOpMatcher *softmaxroot;
        makeSoftmaxMatcher(matcherContext, maxReduction, softmaxroot);
        if (matchPattern(op, *softmaxroot)) {
          Value src = maxReduction->getCaptured()->getOperand(0);
          softmaxRoots.push_back(std::make_pair(op, src));
        }
        if (std::optional<Value> newRhs = matchATransposeBMatmul(op)) {
          transposeMatmulRoots.push_back(std::make_pair(
              cast<linalg::MatmulOp>(op.getOperation()), newRhs.value()));
        }
      }
    });
    IRRewriter rewriter(&getContext());
    for (std::pair<linalg::LinalgOp, Value> softmax : softmaxRoots) {
      linalg::LinalgOp op = softmax.first;
      Value src = softmax.second;
      rewriter.setInsertionPoint(softmax.first);
      rewriter.replaceOpWithNewOp<IREE::LinalgExt::SoftmaxOp>(
          op, src, op.getDpsInitOperand(0)->get(), op.getNumLoops() - 1);
    }
    for (std::pair<linalg::MatmulOp, Value> aTransposeBMatmul :
         transposeMatmulRoots) {
      auto matmulOp = aTransposeBMatmul.first;
      Value lhs = matmulOp.getDpsInputOperand(0)->get();
      auto newRhs = aTransposeBMatmul.second;
      Value init = matmulOp.getDpsInitOperand(0)->get();
      rewriter.setInsertionPoint(matmulOp);
      SmallVector<NamedAttribute> attrs = getPrunedAttributeList(matmulOp);
      rewriter.replaceOpWithNewOp<linalg::MatmulTransposeBOp>(
          matmulOp, ValueRange{lhs, newRhs}, ValueRange{init}, attrs);
    }
  }
};

} // namespace

std::unique_ptr<Pass> createRaiseSpecialOps() {
  return std::make_unique<RaiseSpecialOpsPass>();
}

} // namespace Flow
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
