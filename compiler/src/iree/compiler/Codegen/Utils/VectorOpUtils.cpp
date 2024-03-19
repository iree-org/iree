// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

namespace mlir::iree_compiler {

std::optional<std::pair<int, int>>
VectorContractOpInfo::getOperandMNIndex() const {
  switch (opKind) {
  case OpKind::MK_KN_MN:
    return std::make_pair(0, 1);
  case OpKind::MK_NK_MN:
    return std::make_pair(0, 0);
  case OpKind::UNKNOWN:
    break;
  }
  return std::nullopt;
}

// Returns the (LHS K, RHS K) dimension index pair.
std::optional<std::pair<int, int>>
VectorContractOpInfo::getOperandKIndex() const {
  switch (opKind) {
  case OpKind::MK_KN_MN:
    return std::make_pair(1, 0);
  case OpKind::MK_NK_MN:
    return std::make_pair(1, 1);
  case OpKind::UNKNOWN:
    break;
  }
  return std::nullopt;
}

// Returns the result (M, N) dimension index pair.
std::optional<std::pair<int, int>>
VectorContractOpInfo::getResultMNIndex() const {
  switch (opKind) {
  case OpKind::MK_KN_MN:
  case OpKind::MK_NK_MN:
    return std::make_pair(0, 1);
  default:
    break;
  }
  return std::nullopt;
}

VectorContractOpInfo::OpKind
VectorContractOpInfo::inferOpKind(MLIRContext *ctx,
                                  SmallVector<AffineMap> maps) const {
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [&](MapList m) { return AffineMap::inferFromExprList(m, ctx); };
  AffineExpr m, n, k;
  bindDims(ctx, m, n, k);
  if (maps == infer({{m, k}, {k, n}, {m, n}}))
    return OpKind::MK_KN_MN;
  if (maps == infer({{m, k}, {n, k}, {m, n}}))
    return OpKind::MK_NK_MN;
  return OpKind::UNKNOWN;
}

} // namespace mlir::iree_compiler
