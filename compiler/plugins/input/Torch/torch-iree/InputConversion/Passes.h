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

// Creates a pipeline that lowers from the torch backend contract to IREE.
// This is based on the torch-backend-to-linalg-on-tensors-backend-pipeline
// pipeline in torch-mlir but includes IREE specific lowerings.
void createTorchToIREEPipeline(OpPassManager &pm);

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerTMTensorConversionPasses();

} // namespace TorchInput
} // namespace iree_compiler
} // namespace mlir

#endif // TORCH_IREE_INPUTCONVERSION_PASSES_H_
