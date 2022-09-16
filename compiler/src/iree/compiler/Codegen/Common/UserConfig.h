// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"

namespace mlir {
namespace iree_compiler {

/// Sets compilation configuration annotated in the incoming IR.
LogicalResult setUserConfig(func::FuncOp entryPointFn, Operation *computeOp,
                            IREE::Codegen::CompilationInfoAttr compilationInfo);

}  // namespace iree_compiler
}  // namespace mlir
