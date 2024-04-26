// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "iree-codegen-amdgpu-chained-matmul-pass"

namespace mlir::iree_compiler {

namespace {

/// Let's assume that we only have vector.contract with the standard indexing
/// maps:
///    (m, n, k), A: (m, k), B: (k, n), C: (m, n).
/// We will represent this contract operation by a "@".
///
/// Given a matmul:
///
/// C = A @ B
///
/// This pass decides when to convert this matmul to:
///
/// A.T = transpose(A)
/// B.T = transpose(B)
/// C.T = B.T @ A.T
/// C = transpose(C.T)
///
/// This is useful when the "@" instruction that the hardware lowers to
/// has a specific layout (see VectorLayoutInterface for more information)
/// but the further uses of C expects a transposed layout to the produced
/// layout.
///
/// For example, for "@" lowering to AMDGPU MFMA instructions, the operands
/// have layout L and L.T and the result has the layout L.T .
/// So if you have a chain of matmuls:
///
/// C (L.T) = A (L) @ B (L.T)
/// E (L.T) = C (L.T)  @ D (L.T)
///            ^^^^^^^
///            Expected layout by instruction is L
///
/// To fix this, we can apply this transformation on the first matrix:
///
/// C.T (L.T) = B.T (L) @ A (L.T)
/// C   (L)   = transpose C.T (L.T)
/// E   (L.T) = C (L)  @ D (L.T)
///            ^^^^^
///            Layout matches the instruction!
///
/// Note that the mathematical formula
///   C = A @ B --> C.T = B.T @ A.T
/// is only defined on standard "@" function, it may be a different
/// transformation for other indexing maps.
struct AMDGPUPrepareForChainedMatmulPass
    : public AMDGPUPrepareForChainedMatmulBase<
          AMDGPUPrepareForChainedMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  /// Given a vector contract of the form
  /// %output = vector.contract %lhs, %rhs, %acc
  /// this function swaps the operands (%rhs, %lhs),
  /// transposes the accumulator and output and updates
  /// the indexing maps for the new contract op.
  ///
  /// Given a contract:
  ///
  ///   result = vector.contract lhs, rhs, acc
  ///
  /// transform it to
  ///
  ///   lhs.T = transpose(lhs)
  ///   rhs.T = transpose(rhs)
  ///   acc.T = transpose(acc)
  ///   result.T = vector.contract rhs.T, lhs.T, acc.T
  ///   result = transpose(result.T)
  ///
  /// This transformation holds for the "@" case we described above. For
  /// other indexing maps, we need to take into account transposed which are
  /// fused into the contract. `isOperandSwapInvariant` tells us when we can
  /// simply swap the operands without transposing them.
  void swapOperandsAndTranspose(RewriterBase &rewriter,
                                vector::ContractionOp contractOp) const {
    Value lhs = contractOp.getLhs();
    Value rhs = contractOp.getRhs();
    Value acc = contractOp.getAcc();
    rewriter.setInsertionPoint(contractOp);

    SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();

    SmallVector<int64_t> perm = {1, 0};
    auto findPermutedMap = [&perm](AffineMap M, AffineMap S) -> AffineMap {
      LLVM_DEBUG(llvm::dbgs() << "M:\n"; M.print(llvm::dbgs());
                 llvm::dbgs() << "\n"; llvm::dbgs() << "S:\n";
                 S.print(llvm::dbgs()); llvm::dbgs() << "\n";);
      // X(S) = M
      // X:
      SmallVector<int64_t> X;
      for (AffineExpr s : S.getResults()) {
        for (auto [i, m] : llvm::enumerate(M.getResults())) {
          if (s == m) {
            X.push_back(i);
            break;
          }
        }
      }

      LLVM_DEBUG(llvm::dbgs() << "X:\n"; interleaveComma(X, llvm::dbgs());
                 llvm::dbgs() << "\n";);

      // X(T):
      applyPermutationToVector(X, perm);

      LLVM_DEBUG(llvm::dbgs() << "X(T):\n"; interleaveComma(X, llvm::dbgs());
                 llvm::dbgs() << "\n";);

      auto xt = AffineMap::getPermutationMap(X, M.getContext());
      return xt;
    };

    MLIRContext *ctx = rewriter.getContext();
    AffineMap lhsMap = indexingMaps[0];
    AffineMap rhsMap = indexingMaps[1];
    AffineMap accMap = indexingMaps[2];

    LLVM_DEBUG(llvm::dbgs() << "Old Contraction Maps:\n";
               lhsMap.print(llvm::dbgs()); llvm::errs() << "\n";
               rhsMap.print(llvm::dbgs()); llvm::errs() << "\n";
               accMap.print(llvm::dbgs()); llvm::errs() << "\n";);

    AffineExpr m, n, k;
    bindDims(ctx, m, n, k);

    auto lhsStandardMap = AffineMap::get(3, 0, {m, k}, ctx);
    auto rhsStandardMap = AffineMap::get(3, 0, {k, n}, ctx);
    auto accStandardMap = AffineMap::get(3, 0, {m, n}, ctx);
    // rhs
    indexingMaps[1] =
        findPermutedMap(lhsMap, lhsStandardMap).compose(rhsStandardMap);
    // lhs
    indexingMaps[0] =
        findPermutedMap(rhsMap, rhsStandardMap).compose(lhsStandardMap);
    // acc
    indexingMaps[2] =
        findPermutedMap(accMap, accStandardMap).compose(accStandardMap);

    LLVM_DEBUG(llvm::dbgs() << "New Contraction Maps:\n";
               indexingMaps[0].print(llvm::dbgs()); llvm::errs() << "\n";
               indexingMaps[1].print(llvm::dbgs()); llvm::errs() << "\n";
               indexingMaps[2].print(llvm::dbgs()); llvm::errs() << "\n";);

    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        contractOp, rhs, lhs, acc, rewriter.getAffineMapArrayAttr(indexingMaps),
        contractOp.getIteratorTypesAttr());
  }

  /// Returns a vector.contract operation that this value was transitively
  /// produced from.
  ///
  /// A chained matmul is one where the lhs of the candidate matrix
  /// is a result of another matmul (a matmul lies in the backward slice of lhs
  /// of the first matmul).
  ///
  /// TODO: This definition of a chained matmul is crude. We should actually be
  /// checking if the layout of the result of the first matmul is transposed
  /// to that expected by the second matmul.
  FailureOr<vector::ContractionOp>
  getTransitiveMatmulParent(vector::ContractionOp contractOp) const {
    SetVector<Operation *> backwardSlice;
    getBackwardSlice(contractOp.getLhs(), &backwardSlice);
    vector::ContractionOp result;
    for (Operation *sliceOp : backwardSlice) {
      auto chainParent = dyn_cast<vector::ContractionOp>(sliceOp);
      if (!chainParent) {
        continue;
      }

      // If we have multiple matmul parents, we fail.
      if (result) {
        return failure();
      }

      result = chainParent;
    }

    if (result) {
      return result;
    }

    return failure();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    SmallVector<vector::ContractionOp> matmulCandidates;
    funcOp.walk([&](vector::ContractionOp contractOp) {
      matmulCandidates.push_back(contractOp);
    });

    IRRewriter rewriter(funcOp.getContext());
    for (vector::ContractionOp candidate : matmulCandidates) {
      FailureOr<vector::ContractionOp> maybeChainedParent =
          getTransitiveMatmulParent(candidate);
      if (failed(maybeChainedParent)) {
        continue;
      }
      auto chainParent = maybeChainedParent.value();
      swapOperandsAndTranspose(rewriter, chainParent);

      // TODO: We should be only transposing the second matrix if the
      // result of the first matmul is used by the second matmul transitively.
      swapOperandsAndTranspose(rewriter, candidate);
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createAMDGPUPrepareForChainedMatmulPass() {
  return std::make_unique<AMDGPUPrepareForChainedMatmulPass>();
}

} // namespace mlir::iree_compiler
