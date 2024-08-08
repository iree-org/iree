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

namespace mlir::iree_compiler::GlobalOptimization {

/// We have a layer of indirection around the GlobalOptimizationOptions because
/// we also need a reference to the const-eval builder, which is injected
/// in by callers.
struct TransformOptions : public PassPipelineOptions<TransformOptions> {
  GlobalOptimizationOptions options;

  // Hook to populate a constant evaluation pass pipeline. If nullptr, then
  // no passes are added for constant evaluation. This must be injected in
  // because constant-evaluators can depend on the whole compiler, of which
  // this is a part, and we maintain strict optionality for this component.
  std::function<void(OpPassManager &passManager)> buildConstEvalPassPipeline;
};

/// Subset of the overall pass pipeline for optimizing globals and numerics.
/// We may ultimately break this out separately so creating a syntactic
/// distinction to keep that as an option.
void buildGlobalOptimizationPassPipeline(
    OpPassManager &mainPassManager, const TransformOptions &transformOptions);

//------------------------------------------------------------------------------
// Wrappers that not use tablegen options.
//------------------------------------------------------------------------------

std::unique_ptr<Pass> createDecomposeConcatPass(bool enableConcatTransposition);

// Used by the demoteContractionInputsToBF16 pass to determine which op inputs
// to demote.
enum class DemotionOption { All, Conv, Matmul, None };
std::unique_ptr<Pass>
createDemoteContractionInputsToBF16Pass(DemotionOption option);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createPropagateLinalgTransposePass(bool enableAggressivePropagation);

//----------------------------------------------------------------------------//
// Register GlobalOptimization Passes
//----------------------------------------------------------------------------//

#define GEN_PASS_DECL
#include "iree/compiler/GlobalOptimization/Passes.h.inc" // IWYU pragma: keep

void registerGlobalOptimizationPipeline();

} // namespace mlir::iree_compiler::GlobalOptimization

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_PASSES_H_
