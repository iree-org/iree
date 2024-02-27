// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LOWERINGSTRATEGY_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LOWERINGSTRATEGY_H_

#include "mlir/IR/BuiltinOps.h"

namespace mlir::iree_compiler::IREE::HAL {

class LoweringStrategy {
public:
  virtual ~LoweringStrategy() = default;

  virtual LogicalResult
  matchAndSetTranslationInfo(mlir::FunctionOpInterface funcOp) {
    assert(false && "unimplemented matchAndSetTranslationInfo");
    return failure();
  }
};

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_LOWERINGSTRATEGY_H_
