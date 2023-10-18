// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_PASSES_H_
#define IREE_COMPILER_GLOBALOPTIMIZATION_PASSES_H_

#include <functional>

#include "iree/compiler/Pipelines/Options.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

// We have a layer of indirection around the GlobalOptimizationOptions because
// we also need a reference to the const-eval builder, which is injected
// in by callers.
struct TransformOptions : public PassPipelineOptions<TransformOptions> {
  GlobalOptimizationOptions options;

  // Hook to populate a constant evaluation pass pipeline. If nullptr, then
  // no passes are added for constant evaluation. This must be injected in
  // because constant-evaluators can depend on the whole compiler, of which
  // this is a part, and we maintain strict optionality for this component.
  std::function<void(OpPassManager &passManager)> buildConstEvalPassPipeline;
};

// Subset of the overall pass pipeline for optimizing globals and numerics.
// We may ultimately break this out separately so creating a syntactic
// distinction to keep that as an option.
void buildGlobalOptimizationPassPipeline(
    OpPassManager &mainPassManager, const TransformOptions &transformOptions);

//===----------------------------------------------------------------------===//
// Input canonicalization and legalization
//===----------------------------------------------------------------------===//

// Creates a pass to convert linalg convolution ops with 1x1 kernels into
// linalg.matmul
std::unique_ptr<Pass> createConvert1X1FilterConv2DToMatmulPass();

// Create a pass to detach elementwise ops from named Linalg ops.
std::unique_ptr<Pass> createDetachElementwiseFromNamedOpsPass();

// Apply patterns to erase unused linalg operands and remove dead code
// associated.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createEraseUnusedLinalgOperands();

// Expands vectors in vector/matrix operations into linalg.batch_matmul/matmul
// forms.
std::unique_ptr<Pass> createExpandVectorsPass();

// Materializes logical encodings to physical encodings if there is a single
// device target.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createMaterializeExternDispatchesPass(
    ArrayRef<std::string> pdlModuleFileName = {});

// Materializes logical encodings to physical encodings if there is a single
// device target.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createMaterializeHomogeneousEncodingsPass();

// Removes tensors that have 0-extents.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createRemoveZeroExtentTensorsPass();

// Sets encoding for tensors to allow tiled execution of operations.
std::unique_ptr<Pass> createSetEncodingPass();

void registerGlobalOptimizationPipeline();

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_PASSES_H_
