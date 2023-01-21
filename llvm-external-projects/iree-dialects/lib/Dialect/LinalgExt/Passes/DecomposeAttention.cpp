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

static Value computeBatchMatmul(Value a, Value b, Value c, Value init,
                                Location loc, OpBuilder &builder) {
  Value initialized =
      builder.create<linalg::FillOp>(loc, ValueRange{init}, c).result();
  auto batchMatmulOp = builder.create<linalg::BatchMatmulOp>(
      loc, initialized.getType(), ValueRange({a, b}), initialized);
  return batchMatmulOp.getResult(0);
}

/// Given 3 BxNxd tensors query(Q), key(K) and value(V), this op converts
/// attention(Q, K, V) to the following sequence of operations:
///
/// 1. QKT = BatchMatmul(Q, Transpose(K))
/// 2. S = Softmax(QKT)
/// 3. A = BatchMatmul(S, V)
///
LogicalResult convertAttentionToLinalg(func::FuncOp funcOp) {
  IRRewriter rewriter(funcOp.getContext());
  funcOp.walk([&](IREE::LinalgExt::AttentionOp attnOp) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(attnOp);
    Location loc = attnOp.getLoc();
    Value query = attnOp.query();
    ShapedType queryType = attnOp.getQueryType();
    Type elementType = queryType.getElementType();

    // Compute transpose(K)
    Value key = attnOp.key();
    SmallVector<int64_t> perm{0, 2, 1};
    SmallVector<OpFoldResult> transposedDims;
    SmallVector<OpFoldResult> keyDims =
        tensor::createDimValues(rewriter, loc, key);
    for (int i = 0; i < perm.size(); i++)
      transposedDims.push_back(keyDims[perm[i]]);
    Value output =
        rewriter.create<tensor::EmptyOp>(loc, transposedDims, elementType);
    auto transposeOp =
        rewriter.create<linalg::TransposeOp>(loc, key, output, perm);
    Value transposedKey = transposeOp.getResult()[0];

    // Compute batchmatmul(Q, transpose(K))
    SmallVector<OpFoldResult> queryDims =
        tensor::createDimValues(rewriter, loc, query);
    SmallVector<OpFoldResult> outputDims(queryDims);
    outputDims[outputDims.size() - 1] = keyDims[1];
    output = rewriter.create<tensor::EmptyOp>(loc, outputDims, elementType);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    Value result =
        computeBatchMatmul(query, transposedKey, output, zero, loc, rewriter);

    // Compute softmax(S)
    int64_t reductionDim = 2;
    auto softmaxOp = rewriter.create<IREE::LinalgExt::SoftmaxOp>(
        loc, result.getType(), result, output, reductionDim);
    Value softmax = softmaxOp.getResult()[0];

    // Compute batchmatmul(S, V)
    Value value = attnOp.value();
    output = rewriter.create<tensor::EmptyOp>(loc, queryDims, elementType);
    result = computeBatchMatmul(softmax, value, output, zero, loc, rewriter);

    attnOp.getResult()[0].replaceAllUsesWith(result);
    return WalkResult::advance();
  });
  return success();
}

struct DecomposeAttentionPass : DecomposeAttentionBase<DecomposeAttentionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, IREE::LinalgExt::IREELinalgExtDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);
    if (failed(convertAttentionToLinalg(getOperation())))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createDecomposeAttentionPass() {
  return std::make_unique<DecomposeAttentionPass>();
}

} // namespace LinalgExt
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
