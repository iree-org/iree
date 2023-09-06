// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORCH_IREE_INPUTCONVERSION_PASSES_H_
#define TORCH_IREE_INPUTCONVERSION_PASSES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace TorchInput {

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertTMTensorToLinalgExtPass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerTMTensorConversionPasses();

} // namespace TorchInput
} // namespace iree_compiler
} // namespace mlir

#endif // TORCH_IREE_INPUTCONVERSION_PASSES_H_
