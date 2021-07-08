// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_

#include "iree/compiler/Dialect/LinalgExt/IR/TiledOpInterface.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"

namespace mlir {
namespace iree_compiler {
namespace linalg_ext {

struct TiledOp {
  Operation *op;
  SmallVector<Operation *> loops;
  SmallVector<Value> results;
};

FailureOr<TiledOp> tileLinalgExtOp(OpBuilder &b, TiledOpInterface op,
                                   const linalg::LinalgTilingOptions &options);

}  // namespace linalg_ext
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H_
