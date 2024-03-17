// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler {

/// A class for querying information about a contract op.
class VectorContractOpInfo {
public:
  enum class OpKind { MK_KN_MN, MK_NK_MN, UNKNOWN };

  explicit VectorContractOpInfo(vector::ContractionOp op) {
    contractionDims = *linalg::inferContractionDims(op.getIndexingMapsArray());
    opKind = inferOpKind(op.getContext(), op.getIndexingMapsArray());
  }

  OpKind getOpKind() const { return opKind; }

  // Returns the (LHS M, RHS N) dimension index pair.
  std::optional<std::pair<int, int>> getOperandMNIndex() const;
  std::pair<int, int> getOperandFullMNIndex() const;

  // Returns the (LHS K, RHS K) dimension index pair.
  std::optional<std::pair<int, int>> getOperandKIndex() const;
  std::pair<int, int> getOperandFullKIndex() const;

  // Returns the result (M, N) dimension index pair.
  std::optional<std::pair<int, int>> getResultMNIndex() const;
  std::pair<int, int> getResultFullMNIndex() const;

  SmallVector<unsigned, 2> getMDims() const { return contractionDims.m; }

  SmallVector<unsigned, 2> getNDims() const { return contractionDims.n; }

  int64_t getARank() {
    return contractionDims.m.size() + contractionDims.k.size();
  }
  int64_t getBRank() {
    return contractionDims.k.size() + contractionDims.n.size();
  }
  int64_t getCRank() {
    return contractionDims.m.size() + contractionDims.n.size();
  }

  SmallVector<int64_t> lhsMDims;
  int64_t lhsKDim;
  SmallVector<int64_t> rhsNDims;
  int64_t rhsKDim;
  SmallVector<int64_t> outMDims;
  SmallVector<int64_t> outNDims;

private:
  // Gets the kind of a contract op with the given indexing |maps|.
  OpKind inferOpKind(MLIRContext *ctx, SmallVector<AffineMap> maps);

  OpKind opKind = OpKind::UNKNOWN;

  linalg::ContractionDimensions contractionDims;
};

} // namespace mlir::iree_compiler
