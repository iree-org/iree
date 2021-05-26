// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CONVERSION_PASSES_H_
#define IREE_COMPILER_CONVERSION_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

// Registers all conversion passes in this directory.
void registerConversionPasses();

//------------------------------------------------------------------------------
// Conversions into Linalg
//------------------------------------------------------------------------------

/// Creates a pass to fuse Linalg operations on tensors.
std::unique_ptr<Pass> createFusionOfTensorOpsPass();

/// Creates XLA-HLO to Linalg on tensors transformation pass.
std::unique_ptr<OperationPass<FuncOp>> createHLOToLinalgOnTensorsPass();

/// Resolves shape related ops (std.dim, shapex.tie_shape, etc.) by tracing
/// them back to the original HAL interface bindings.
std::unique_ptr<OperationPass<FuncOp>> createResolveShapeOpsPass();

//------------------------------------------------------------------------------
// Misc/common conversions
//------------------------------------------------------------------------------

/// Create a pass to convert a model using f32 type to the equivalent one
/// using f16.
std::unique_ptr<OperationPass<ModuleOp>> createDemoteF32ToF16Pass();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_PASSES_H_
