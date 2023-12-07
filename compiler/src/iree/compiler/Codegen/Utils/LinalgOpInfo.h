// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_LINALGOPINFO_H_
#define IREE_COMPILER_CODEGEN_COMMON_LINALGOPINFO_H_
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir::linalg {
class LinalgOp;
} // namespace mlir::linalg

namespace mlir::iree_compiler {

/// Returns true if a map represents the appropriate transpose. Pass this into
/// the LinalgOpInfo for additional transpose granularity.
using TransposeMapFilter = std::function<bool(AffineMap map)>;

class LinalgOpInfo {
public:
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
  bool transposeTrait;
  bool reductionTrait;
  bool dynamicTrait;
  SmallVector<OpOperand *> transposeOperands;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_LINALGOPINFO_H_
