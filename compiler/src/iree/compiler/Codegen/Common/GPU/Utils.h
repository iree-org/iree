// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {

struct LogicalResult;

namespace iree_compiler {

LogicalResult swizzleWorkgroupsInFunc(func::FuncOp funcOp,
                                      unsigned swizzleLogTile);

} // namespace iree_compiler
} // namespace mlir
