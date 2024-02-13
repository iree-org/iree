// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::iree_compiler {

namespace {

struct AMDGPUChainedMatmulPass
    : public AMDGPUChainedMatmulBase<AMDGPUChainedMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  bool
  isChainedMatmul(SmallVector<vector::ContractionOp> &chainedMatmuls) const {
    SetVector<Operation *> forwardSlice;
    getForwardSlice(chainedMatmuls[0].getOperation(), &forwardSlice);
    for (auto *sliceOp : llvm::reverse(forwardSlice)) {
      auto candidateContract = dyn_cast<vector::ContractionOp>(sliceOp);
      if (!candidateContract)
        continue;
      if (candidateContract == chainedMatmuls[1])
        return true;
    }
    return false;
  }

  void swapOperandsAndTranspose(RewriterBase &rewriter,
                                vector::ContractionOp contractOp) const {
    Value lhs = contractOp.getLhs();
    Value rhs = contractOp.getRhs();
    Value acc = contractOp.getAcc();
    rewriter.setInsertionPoint(contractOp);
    Value transposed = rewriter.create<vector::TransposeOp>(
        contractOp.getLoc(), acc, SmallVector<int64_t>{1, 0});
    AffineExpr m, n, k;
    bindDims(rewriter.getContext(), m, n, k);
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    SmallVector<AffineMap> newIndexingMaps = infer({{n, k}, {m, k}, {n, m}});
    vector::ContractionOp swappedOp = rewriter.create<vector::ContractionOp>(
        contractOp.getLoc(), rhs, lhs, transposed,
        rewriter.getAffineMapArrayAttr(newIndexingMaps),
        contractOp.getIteratorTypesAttr());
    Value newResult = swappedOp.getResult();
    transposed = rewriter.create<vector::TransposeOp>(
        contractOp.getLoc(), newResult, SmallVector<int64_t>{1, 0});
    rewriter.replaceAllUsesWith(contractOp.getResult(), transposed);
  }

  bool isCompatibleIndexingMap(vector::ContractionOp contractOp,
                               MLIRContext *ctx) {
    AffineExpr m, n, k;
    bindDims(ctx, m, n, k);
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    SmallVector<AffineMap> newIndexingMaps = infer({{m, k}, {n, k}, {m, n}});
    return newIndexingMaps == contractOp.getIndexingMapsArray();
  }

  // Given a contraction: D = A x B.T + C, this pattern
  // modifies the contraction to: D.T = B x A.T + C.T
  // where .T computes the transpose. Applying this pattern
  // can remove layout conflicts in certain scenarios
  // (such as flash attention on CDNA GPUs).
  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<vector::ContractionOp> chainedMatmuls;
    funcOp.walk([&](vector::ContractionOp contractOp) {
      if (!isCompatibleIndexingMap(contractOp, funcOp.getContext()))
        return WalkResult::advance();
      chainedMatmuls.push_back(contractOp);
      return WalkResult::advance();
    });
    if (chainedMatmuls.size() != 2)
      return;
    if (!isChainedMatmul(chainedMatmuls))
      return;
    IRRewriter rewriter(funcOp.getContext());
    for (vector::ContractionOp op : chainedMatmuls) {
      swapOperandsAndTranspose(rewriter, op);
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createAMDGPUChainedMatmulPass() {
  return std::make_unique<AMDGPUChainedMatmulPass>();
}

} // namespace mlir::iree_compiler
