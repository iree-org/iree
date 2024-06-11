// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_DERIVEDCONFIGUTILS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_DERIVEDCONFIGUTILS_H_

#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler::IREE::GPU {

SmallVector<int64_t> deriveThreadTileSizes(Operation *op);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IR_DERIVEDCONFIGUTILS_H_
