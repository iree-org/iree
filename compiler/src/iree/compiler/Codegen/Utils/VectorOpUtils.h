// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::iree_compiler {

/// A class for querying information about a contract op.
class VectorContractOpInfo {
public:
  enum class OpKind { MK_KN_MN, MK_NK_MN, UNKNOWN };

  explicit VectorContractOpInfo(vector::ContractionOp op) {
    opKind = inferOpKind(op.getContext(), op.getIndexingMapsArray());
  }

  OpKind getOpKind() const { return opKind; }

  // Returns the (LHS M, RHS N) dimension index pair.
  std::optional<std::pair<int, int>> getOperandMNIndex() const;

  // Returns the (LHS K, RHS K) dimension index pair.
  std::optional<std::pair<int, int>> getOperandKIndex() const;

  // Returns the result (M, N) dimension index pair.
  std::optional<std::pair<int, int>> getResultMNIndex() const;

private:
  // Gets the kind of a contract op with the given indexing |maps|.
  OpKind inferOpKind(MLIRContext *ctx, SmallVector<AffineMap> maps) const;

  OpKind opKind = OpKind::UNKNOWN;
};

} // namespace mlir::iree_compiler
