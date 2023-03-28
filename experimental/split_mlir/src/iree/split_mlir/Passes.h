// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SPLIT_MLIR_TRANSFORM_PASSES_H_
#define IREE_SPLIT_MLIR_TRANSFORM_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {

class ModuleOp;
namespace func {
class FuncOp;
}  // namespace func

namespace iree {
namespace split_mlir {

#define GEN_PASS_DECL
#include "iree/split_mlir/Passes.h.inc"  // IWYU pragma: export

std::unique_ptr<OperationPass<ModuleOp>> createOutlineFunctionsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createMarkBisectPass();

#define GEN_PASS_REGISTRATION
#include "iree/split_mlir/Passes.h.inc"  // IWYU pragma: export

}  // namespace split_mlir
}  // namespace iree
}  // namespace mlir

#endif  // IREE_SPLIT_MLIR_TRANSFORM_PASSES_H_
