// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DISPATCHCREATION_PASSES_H_
#define IREE_COMPILER_DISPATCHCREATION_PASSES_H_

#include <functional>

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::DispatchCreation {

enum class EncodingOptions { Padding, Generic };

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

struct TransformOptions : public PassPipelineOptions<TransformOptions> {
  Option<bool> enableAggressiveFusion{
      *this,
      "aggressive-fusion",
      llvm::cl::desc("Enable aggressive fusion for dispatch creation pipeline"),
      llvm::cl::init(false),
  };
  Option<bool> enableFuseMultiUse{
      *this,
      "fuse-multi-use",
      llvm::cl::desc("Fuse operations with multiple uses."),
      llvm::cl::init(true),
  };
  Option<bool> dataTiling{
      *this,
      "data-tiling",
      llvm::cl::desc("Enable data-tiling for dispatch creation pipeline"),
      llvm::cl::init(false),
  };
  Option<bool> enableSplitReduction{
      *this,
      "split-reduction",
      llvm::cl::desc("Enable split reduction for dispatch creation pipeline"),
      llvm::cl::init(false),
  };
  Option<bool> enableAggressiveReshapeMovement{
      *this,
      "aggressive-reshape-movement",
      llvm::cl::desc(
          "Enable aggressive reshape movement (bubbling expand/collapse "
          "shapes across reduction ops)"),
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
};

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
