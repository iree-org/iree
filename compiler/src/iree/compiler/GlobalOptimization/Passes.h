// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_PASSES_H_
#define IREE_COMPILER_GLOBALOPTIMIZATION_PASSES_H_

#include <functional>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::GlobalOptimization {

struct TransformOptions : public PassPipelineOptions<TransformOptions> {
  ListOption<std::string> parameterImportPaths{
      *this,
      "parameter-import-paths",
      llvm::cl::desc("File paths to archives to import parameters from with an "
                     "optional `scope=` prefix."),
  };
  ListOption<std::string> parameterImportKeys{
      *this,
      "parameter-import-keys",
      llvm::cl::desc("List of parameter keys to import. Any matching keys from "
                     "any scope will be imported."),
  };
  Option<int64_t> parameterImportMaximumSize{
      *this,
      "parameter-import-maximum-size",
      llvm::cl::desc("Maximum size of parameters to import or 0 to disable "
                     "automatic import."),
      llvm::cl::init(0),
  };
  Option<std::string> parameterExportPath{
      *this,
      "parameter-export-path",
      llvm::cl::desc("File path to an archive to export parameters to with an "
                     "optional `scope=` prefix."),
      llvm::cl::init(""),
  };
  Option<int64_t> parameterExportMinimumSize{
      *this,
      "parameter-export-minimum-size",
      llvm::cl::desc("Minimum size of constants to export as parameters."),
      llvm::cl::init(0),
  };
  Option<std::string> parameterSplatExportFile{
      *this,
      "parameter-splat-export-file",
      llvm::cl::desc("File path to create a splat parameter archive out of all "
                     "parameters in the module."),
      llvm::cl::init(""),
  };
  Option<bool> aggressiveTransposePropagation{
      *this,
      "aggressive-transpose-propagation",
      llvm::cl::desc(
          "Enables aggressive propagation of transposes to the inputs of named "
          "ops, rewriting named ops as fused generics."),
      llvm::cl::init(false),
  };
  Option<bool> propagateTransposesThroughConv{
      *this,
      "propagate-transposes-through-conv",
      llvm::cl::desc(
          "Enables propagation of transpose ops through convolutions"),
      llvm::cl::init(false),
  };
  Option<bool> sinkTransposeThroughPad{
      *this,
      "sink-transpose-through-pad",
      llvm::cl::desc("Enables sinking transpose through pad operations"),
      llvm::cl::init(false),
  };
  Option<bool> outerDimConcat{
      *this,
      "outer-dim-concat",
      llvm::cl::desc("Enables transposing all concatenations to the outer most "
                     "dimension."),
      llvm::cl::init(false),
  };
  Option<bool> dataTiling{
      *this,
      "data-tiling",
      llvm::cl::desc("Enables data tiling in global optimization phase. There "
                     "are two data-tiling flags during the transition state. "
                     "The other has to be off if this one is enabled. Any "
                     "feature built on top of this path will be deprecated."),
      llvm::cl::init(false),
  };
  Option<bool> constEval{
      *this,
      "const-eval",
      llvm::cl::desc("Enables recursive evaluation of immutable globals using "
                     "the compiler and runtime."),
      llvm::cl::init(true),
  };
  Option<bool> numericPrecisionReduction{
      *this,
      "numeric-precision-reduction",
      llvm::cl::desc("Optimizations to reduce numeric precision where it is "
                     "safe to do so."),
      llvm::cl::init(false),
  };
  Option<bool> stripAssertions{
      *this,
      "strip-assertions",
      llvm::cl::desc("Strips debug assertions after any useful information has "
                     "been extracted."),
      llvm::cl::init(false),
  };
  Option<bool> generalizeMatmul{
      *this,
      "generalize-matmul",
      llvm::cl::desc("Converts linalg named matmul ops to linalg generic ops."),
      llvm::cl::init(false),
  };
  Option<bool> constExprHoisting{
      *this,
      "const-expr-hoisting",
      llvm::cl::desc("Enables hoisting of constant expressions."),
      llvm::cl::init(true),
  };
  Option<int64_t> constExprMaxSizeIncreaseThreshold{
      *this,
      "const-expr-max-size-increase-threshold",
      llvm::cl::desc(
          "Maximum size increase threshold for constant expression hoisting."),
      llvm::cl::init(1024 * 1024),
  };

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
