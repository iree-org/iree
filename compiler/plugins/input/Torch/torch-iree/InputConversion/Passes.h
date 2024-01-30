// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORCH_IREE_INPUTCONVERSION_PASSES_H_
#define TORCH_IREE_INPUTCONVERSION_PASSES_H_

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::TorchInput {

struct TorchToIREELoweringPipelineOptions
    : public PassPipelineOptions<TorchToIREELoweringPipelineOptions> {
  Option<bool> strictSymbolicShapes{
      *this, "strict-symbolic-shapes",
      llvm::cl::desc("Use strict symbolic shapes."), llvm::cl::init(true)};
};

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createBitCastQuantTensorPass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConvertTMTensorToLinalgExtPass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createSetStrictSymbolicShapesPass();

// Creates a pipeline that lowers from the torch backend contract to IREE.
// This is based on the torch-backend-to-linalg-on-tensors-backend-pipeline
// pipeline in torch-mlir but includes IREE specific lowerings.
void createTorchToIREEPipeline(
    OpPassManager &pm, const TorchToIREELoweringPipelineOptions &options);

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerTMTensorConversionPasses();

} // namespace mlir::iree_compiler::TorchInput

#endif // TORCH_IREE_INPUTCONVERSION_PASSES_H_
