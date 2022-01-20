// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_
#define IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace LinalgExt {

std::unique_ptr<OperationPass<FuncOp>> createTiledOpInterfaceTilingPass();

std::unique_ptr<OperationPass<FuncOp>> createLinalgExtToLoopsPass();

std::unique_ptr<OperationPass<>> createPadContractionToBlockSizePass();

void registerPasses();

}  // namespace LinalgExt
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_DIALECTS_DIALECT_LINALGEXT_TRANSFORMS_PASSES_H_
