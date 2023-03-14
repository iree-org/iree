// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_TILINGHELPER_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_TILINGHELPER_H_

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace func {
class FuncOp;
}
namespace iree_compiler {

/// Apply tiling to reduction dimensions based on op attributes.
LogicalResult tileToSerialLoops(func::FuncOp funcOp, bool onlyReduction = true);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_LLVMGPU_TILINGHELPER_H_
