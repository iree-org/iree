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
/// C.T = A.T @ B.T
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
/// C.T (L.T) = A.T (L) @ B (L.T)
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
    acc = rewriter.create<vector::TransposeOp>(contractOp.getLoc(), acc,
                                               SmallVector<int64_t>{1, 0});

    if (!isOperandSwapInvariant(contractOp)) {
      lhs = rewriter.create<vector::TransposeOp>(contractOp.getLoc(), lhs,
                                                 SmallVector<int64_t>{1, 0});
      rhs = rewriter.create<vector::TransposeOp>(contractOp.getLoc(), rhs,
                                                 SmallVector<int64_t>{1, 0});
    }

    vector::ContractionOp swappedOp = rewriter.create<vector::ContractionOp>(
        contractOp.getLoc(), rhs, lhs, acc, contractOp.getIndexingMaps(),
        contractOp.getIteratorTypesAttr());
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(
        contractOp, swappedOp.getResult(), SmallVector<int64_t>{1, 0});
  }

  /// For a matmul_transpose_b, this transformation boils down to an operand
  /// swap and result transpose:
  ///
  /// def matmul_transpose_b(A, B):
  ///   B.T = transpose(B)
  ///   C = A @ B.T
  ///   return C
  ///
  /// def matmul_transpose_b_swapped(A, B):
  ///   A.T = transpose(A)
  ///   C.T = B @ A.T
  ///   C   = transpose(C.T)
  ///   return C
  ///
  /// matmul_transpose_b(B, A) = matmul_transpose_b_swapped(B, A).T
  ///
  /// TODO: This check applies more generally when one of the operands in the
  /// function is transposed compared to what "@" expects.
  bool isOperandSwapInvariant(vector::ContractionOp contractOp) const {
    AffineExpr m, n, k;
    bindDims(contractOp.getContext(), m, n, k);
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [&](MapList m) {
      return AffineMap::inferFromExprList(m, ctx);
    };
    SmallVector<AffineMap> newIndexingMaps = infer({{m, k}, {n, k}, {m, n}});
    return newIndexingMaps == contractOp.getIndexingMapsArray();
  }

  /// Returns a vector.contract operation that this value was transitively
  /// produced from.
  ///
  /// A chained matmul is one where the lhs of the candidate matrix
  /// is a result of another matmul (a matmul lies in the backward slice of lhs
  /// of the first matmul).
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

      // For now, we only support transpose invariant matmuls. This is because
      // transposing the inputs may have a non-trivial cost which we need
      // to think about.
      if (!isOperandSwapInvariant(chainParent)) {
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
    for (auto candidate : matmulCandidates) {
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
