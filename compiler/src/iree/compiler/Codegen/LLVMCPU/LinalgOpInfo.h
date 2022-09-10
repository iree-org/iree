// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_LINALGOPINFO_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_LINALGOPINFO_H_

namespace mlir {

namespace linalg {
class LinalgOp;
}

namespace iree_compiler {

class LinalgOpInfo {
 public:
  LinalgOpInfo(linalg::LinalgOp linalgOp);

  bool isTranspose() const { return hasTransposeTrait; }
  bool isReduction() const { return hasReductionTrait; }

 private:
  void computeInfo(linalg::LinalgOp);

  bool hasTransposeTrait;
  bool hasReductionTrait;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_LLVMCPU_LINALGOPINFO_H_
