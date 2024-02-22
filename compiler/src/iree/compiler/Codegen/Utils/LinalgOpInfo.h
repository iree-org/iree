// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_LINALGOPINFO_H_
#define IREE_COMPILER_CODEGEN_COMMON_LINALGOPINFO_H_

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/AffineMap.h"

namespace mlir::iree_compiler {

class LinalgOpInfo {
public:
  /// Returns true if a map represents a chosen transpose granularity.
  using TransposeMapFilter = std::function<bool(AffineMap map)>;

  LinalgOpInfo(linalg::LinalgOp linalgOp);
  LinalgOpInfo(linalg::LinalgOp linalgOp,
               TransposeMapFilter transposeMapFilter);

  bool isTranspose() const { return !transposeOperands.empty(); }
  bool isReduction() const { return reductionTrait; }
  bool isDynamic() const { return dynamicTrait; }

  ArrayRef<OpOperand *> getTransposeOperands() const {
    return transposeOperands;
  }

private:
  void computeInfo(linalg::LinalgOp);

  TransposeMapFilter transposeMapFilter;
  bool reductionTrait;
  bool dynamicTrait;
  SmallVector<OpOperand *> transposeOperands;
};

// Returns true if the given |linalgOp| is a matmul or batch matmul.
// This also looks into the shape to filter out cases like matvec.
bool isMatmulOrBatchMatmul(linalg::LinalgOp linalgOp);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_LINALGOPINFO_H_
