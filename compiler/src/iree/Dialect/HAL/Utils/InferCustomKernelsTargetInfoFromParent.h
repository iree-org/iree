// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_UTILS_INFERCUSTOMKERNELSTARGETINFOFROMPARENT_H_
#define IREE_COMPILER_DIALECT_HAL_UTILS_INFERCUSTOMKERNELSTARGETINFOFROMPARENT_H_

#include <stdint.h>

#include <cassert>

#include "iree/compiler/Utils/CustomKernelsTargetInfo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {

LogicalResult InferCustomKernelsTargetInfoFromParent(
    func::FuncOp entryPointFn, CustomKernelsTargetInfo &targetInfo);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_UTILS_INFERCUSTOMKERNELSTARGETINFOFROMPARENT_H_
