// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_PASSUTILS_H_
#define IREE_COMPILER_CODEGEN_COMMON_PASSUTILS_H_

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

/// Pass manager nesting for `FunctionOpInterface` ops.
using FunctionLikeNest = MultiOpNest<func::FuncOp>;

/// Helper method to get a pass manager nested at `FunctionOpInterface`.
std::optional<OpPassManager>
getFunctionOpInterfacePassManager(FunctionOpInterface funcOp);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_PASSUTILS_H_
