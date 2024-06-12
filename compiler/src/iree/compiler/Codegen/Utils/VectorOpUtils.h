// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::iree_compiler {

/// A class for querying information about a contract op.
class VectorContractOpInfo {
public:
  explicit VectorContractOpInfo(vector::ContractionOp op);

  // Returns the (LHS M, RHS N) dimension index pair.
  std::pair<int, int> getOperandMNIndex() const;

  // Returns the (LHS K, RHS K) dimension index pair.
  std::pair<int, int> getOperandKIndex() const;

  // Returns the result (M, N) dimension index pair.
  std::pair<int, int> getResultMNIndex() const;

  SmallVector<unsigned, 2> getMDims() const { return contractionDims.m; }
  SmallVector<unsigned, 2> getNDims() const { return contractionDims.n; }
  SmallVector<unsigned, 2> getKDims() const { return contractionDims.k; }

  int64_t getARank() const {
    return contractionDims.batch.size() + contractionDims.m.size() +
           contractionDims.k.size();
  }
  int64_t getBRank() const {
    return contractionDims.batch.size() + contractionDims.k.size() +
           contractionDims.n.size();
  }
  int64_t getCRank() const {
    return contractionDims.batch.size() + contractionDims.m.size() +
           contractionDims.n.size();
  }

  int64_t getBatchCount() const { return contractionDims.batch.size(); }

  SmallVector<int64_t> lhsMDims;
  int64_t lhsKDim;
  SmallVector<int64_t> rhsNDims;
  int64_t rhsKDim;
  SmallVector<int64_t> outMDims;
  SmallVector<int64_t> outNDims;

private:
  linalg::ContractionDimensions contractionDims;
};

} // namespace mlir::iree_compiler
