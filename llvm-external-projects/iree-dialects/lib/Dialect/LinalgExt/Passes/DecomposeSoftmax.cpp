// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/PassDetail.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

namespace {

std::tuple<SmallVector<utils::IteratorType>, SmallVector<AffineMap>>
computeIteratorTypesAndIndexingMaps(int64_t inputRank, int64_t dim,
                                    OpBuilder &builder,
                                    bool allParallel = false) {
  SmallVector<utils::IteratorType> iteratorTypes(inputRank,
                                                 utils::IteratorType::parallel);
  if (!allParallel)
    iteratorTypes[dim] = utils::IteratorType::reduction;
  auto identityMap =
      AffineMap::getMultiDimIdentityMap(inputRank, builder.getContext());
  SmallVector<AffineExpr, 2> affineExprs;
  for (int i = 0; i < inputRank; i++) {
    if (i != dim)
      affineExprs.push_back(mlir::getAffineDimExpr(i, builder.getContext()));
  }
  auto reductionMap =
      AffineMap::get(inputRank, 0, affineExprs, builder.getContext());
  SmallVector<AffineMap> indexingMaps{identityMap, reductionMap};
  return std::make_tuple(iteratorTypes, indexingMaps);
}

template <typename T>
static Value reduce(Value input, Value output, int64_t dim, Location loc,
                    OpBuilder &builder) {
  auto inputType = input.getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();
  auto [iteratorTypes, indexingMaps] =
      computeIteratorTypesAndIndexingMaps(inputRank, dim, builder);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, output.getType(), input, output, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<T>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

static linalg::GenericOp subtractAndExp(Value input, Value max, Value output,
                                        int64_t dim, Location loc,
                                        OpBuilder &builder) {
  auto inputType = input.getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();
  auto [iteratorTypes, indexingMaps] =
      computeIteratorTypesAndIndexingMaps(inputRank, dim, builder, true);
  indexingMaps.push_back(indexingMaps[0]);
  return builder.create<linalg::GenericOp>(
      loc, input.getType(), ValueRange{input, max}, output, indexingMaps,
      iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value diff = b.create<arith::SubFOp>(loc, args[0], args[1]);
        Value result = b.create<math::ExpOp>(loc, diff);
        b.create<linalg::YieldOp>(loc, result);
      });
}

static Value computeSoftmax(Value numerator, Value denominator, Value output,
                            int64_t dim, Location loc, OpBuilder &builder) {
  auto inputType = numerator.getType().cast<ShapedType>();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputShape.size();
  auto [iteratorTypes, indexingMaps] =
      computeIteratorTypesAndIndexingMaps(inputRank, dim, builder, true);
  indexingMaps.push_back(indexingMaps[0]);
  auto genericOp = builder.create<linalg::GenericOp>(
      loc, numerator.getType(), ValueRange{numerator, denominator}, output,
      indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result = b.create<arith::DivFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, result);
      });
  return genericOp.getResult(0);
}

/// Given an N-dimensional tensor x, this op converts
/// softmax(x) to the following sequence of operations:
///
/// 1. Compute the max of x along dimension d. This results
///    in a N-1 dimensional tensor m.
///    m = max(x, dim = d)
///
/// 2. Subtract m from x and exponentiate. This results in
///    a N dimensional tensor z.
///    z = exp(x - m)
///
/// 3. Compute the sum of z along dimension d. This results in
///    a N-1 dimensional tensor l.
///    l = sum(z, dim = d)
///
/// 4. Divide z and l. This gives the N-dimensional softmax.
///    softmax = z / l
///
LogicalResult convertSoftmaxToGenerics(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());
  SmallVector<Operation *> toDelete;
  SmallVector<Operation *> softmaxOpsToDecompose;
  funcOp.walk([&](linalg::SoftmaxOp softmaxOp) {
    softmaxOpsToDecompose.push_back(softmaxOp);
  });

  OpBuilder::InsertionGuard guard(rewriter);
  for (Operation *softmaxOp : softmaxOpsToDecompose) {
    // Cast linalg::softmax to AggregatedOpInterface since this where
    // `decomposeOperation` is implemented.
    auto decomposableSoftmaxOp =
        cast<mlir::linalg::AggregatedOpInterface>(softmaxOp);

    // Decompose linalg::softmax.
    FailureOr<SmallVector<Value>> result =
        decomposableSoftmaxOp.decomposeOperation(rewriter);
    if (failed(result)) {
      failed(rewriter.notifyMatchFailure(
          softmaxOp, "linalg::SoftmaxOp could not be decomposed"));
      return failure();
    }

    // Replace the result of linalg::softmax with the `result` generated via
    // the decomposition above.
    rewriter.replaceOp(decomposableSoftmaxOp, *result);

    // Fusion later depends on couple of Ops/Values - we try to obtain the same
    // by backtracking through the generated value's def-chain.
    Operation *resultOp = (*result)[0].getDefiningOp();
    Value numerator = resultOp->getOperand(0);
    Operation *numeratorOp = numerator.getDefiningOp();

    // Rematerialize operands that are marked for this.
    SmallVector<OpOperand *> uses = llvm::to_vector(llvm::map_range(
        numerator.getUses(), [](OpOperand &use) { return &use; }));
    for (OpOperand *use : uses) {
      Operation *consumer = use->getOwner();
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(consumer);
      FailureOr<linalg::ElementwiseOpFusionResult> fusionResult =
          linalg::fuseElementwiseOps(rewriter, use);
      if (succeeded(fusionResult)) {
        SmallVector<Value> replacements = llvm::to_vector(
            llvm::map_range(consumer->getResults(), [&](Value oldValue) {
              return fusionResult->replacements.lookup(oldValue);
            }));
        rewriter.replaceOp(consumer, replacements);
      }
    }
    toDelete.push_back(numeratorOp);
  }
  for (Operation *op : toDelete) {
    rewriter.eraseOp(op);
  }

  return success();
}

struct DecomposeSoftmaxPass : DecomposeSoftmaxBase<DecomposeSoftmaxPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, IREE::LinalgExt::IREELinalgExtDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    if (failed(convertSoftmaxToGenerics(getOperation())))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createDecomposeSoftmaxPass() {
  return std::make_unique<DecomposeSoftmaxPass>();
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
