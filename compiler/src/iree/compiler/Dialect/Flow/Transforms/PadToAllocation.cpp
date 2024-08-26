// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- PadToAllocation.cpp ----- Pass to increase cache bandwidth ---------===//
//
// Inserts tensor padding to pad the underlying allocations and increase the
// L1 cache bandwidth.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_PADTOALLOCATIONPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {
struct PadMmt final : OpRewritePattern<linalg::MatmulTransposeBOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulTransposeBOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsType = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
    auto rhsType = dyn_cast<RankedTensorType>(op.getOperand(1).getType());
    if (!lhsType || !rhsType)
      return failure();

    if (lhsType.isDynamicDim(0) || lhsType.isDynamicDim(1))
      return failure();
    if (rhsType.isDynamicDim(0) || rhsType.isDynamicDim(1))
      return failure();

    // int64_t mDim = lhsType.getDimSize(0);
    // int64_t nDim = rhsType.getDimSize(0);
    int64_t kDim = rhsType.getDimSize(1);
    int64_t elementTypeBytes = lhsType.getElementTypeBitWidth() / 8;

    int64_t kSize = kDim * elementTypeBytes;
    if (kSize % (128 * 4) != 0)
      return failure();

    llvm::outs() << "MMT: " << op << "\n";
    llvm::outs() << "\tkDim size: " << kSize << " B\n";
    return failure();
  }
};

struct PaddToAllocationPass final
    : impl::PadToAllocationPassBase<PaddToAllocationPass> {
  using impl::PadToAllocationPassBase<
      PaddToAllocationPass>::PadToAllocationPassBase;
  void runOnOperation() override {
    llvm::outs() << "JAKUB\n";
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<PadMmt>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
