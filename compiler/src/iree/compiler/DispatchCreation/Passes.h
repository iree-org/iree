// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DISPATCHCREATION_PASSES_H_
#define IREE_COMPILER_DISPATCHCREATION_PASSES_H_

#include <functional>

#include "iree/compiler/Pipelines/Options.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::DispatchCreation {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

/// This is a placeholder for future. We should pass all the options through the
/// struct.
struct TransformOptions : public PassPipelineOptions<TransformOptions> {};

void buildDispatchCreationPassPipeline(
    OpPassManager &passManager, const TransformOptions &transformOptions);

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "iree/compiler/DispatchCreation/Passes.h.inc" // IWYU pragma: keep

void registerDispatchCreationPasses();

//===----------------------------------------------------------------------===//
// Register Pipelines
//===----------------------------------------------------------------------===//
void registerDispatchCreationPipelines();

} // namespace mlir::iree_compiler::DispatchCreation

#endif
