// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

namespace mlir::iree_compiler {

std::pair<int, int> VectorContractOpInfo::getOperandMNIndex() const {
  return std::make_pair(lhsMDims.back(), rhsNDims.back());
}

// Returns the (LHS K, RHS K) dimension index pair.
std::pair<int, int> VectorContractOpInfo::getOperandKIndex() const {
  return std::make_pair(lhsKDim, rhsKDim);
}

// Returns the result (M, N) dimension index pair.
std::pair<int, int> VectorContractOpInfo::getResultMNIndex() const {
  return std::make_pair(outMDims.back(), outNDims.back());
}

VectorContractOpInfo::OpKind
VectorContractOpInfo::inferOpKind(MLIRContext *ctx,
                                  SmallVector<AffineMap> maps) {
  if (contractionDims.k.size() != 1) {
    return OpKind::UNKNOWN;
  }
  if (!contractionDims.batch.empty()) {
    if (contractionDims.batch.size() > 1 || contractionDims.batch[0] != 0) {
      return OpKind::UNKNOWN;
    }
    if (*maps[0].getResultPosition(getAffineDimExpr(0, ctx)) != 0 ||
        *maps[1].getResultPosition(getAffineDimExpr(0, ctx)) != 0 ||
        *maps[2].getResultPosition(getAffineDimExpr(0, ctx)) != 0) {
      return OpKind::UNKNOWN;
    }
  }

  int64_t innerM = contractionDims.m.back();
  int64_t innerN = contractionDims.n.back();
  int64_t k = contractionDims.k.back();

  int64_t lhsM = *maps[0].getResultPosition(getAffineDimExpr(innerM, ctx));
  lhsKDim = *maps[0].getResultPosition(getAffineDimExpr(k, ctx));
  int64_t rhsN = *maps[1].getResultPosition(getAffineDimExpr(innerN, ctx));
  rhsKDim = *maps[1].getResultPosition(getAffineDimExpr(k, ctx));
  int64_t outM = *maps[2].getResultPosition(getAffineDimExpr(innerM, ctx));
  int64_t outN = *maps[2].getResultPosition(getAffineDimExpr(innerN, ctx));

  for (auto m : contractionDims.m) {
    lhsMDims.push_back(*maps[0].getResultPosition(getAffineDimExpr(m, ctx)));
    outMDims.push_back(*maps[2].getResultPosition(getAffineDimExpr(m, ctx)));
  }
  for (auto n : contractionDims.n) {
    rhsNDims.push_back(*maps[1].getResultPosition(getAffineDimExpr(n, ctx)));
    outNDims.push_back(*maps[2].getResultPosition(getAffineDimExpr(n, ctx)));
  }

  if (outM < outN) {
    if (lhsM < lhsKDim) {
      if (rhsN < rhsKDim) {
        return OpKind::MK_NK_MN;
      }
      return OpKind::MK_KN_MN;
    }
  }
  return OpKind::UNKNOWN;
}

} // namespace mlir::iree_compiler
