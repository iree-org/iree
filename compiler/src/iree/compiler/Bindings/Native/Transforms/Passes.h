// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_BINDINGS_NATIVE_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_BINDINGS_NATIVE_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::ABI {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// Specifies the execution model used for invocations.
enum class InvocationModel {
  // Fully synchronous behavior with no fences.
  Sync,
  // Exposes one wait fence for all inputs and one signal fence for all outputs.
  CoarseFences,
};

struct InvocationOptions : public PassPipelineOptions<InvocationOptions> {
  Option<InvocationModel> invocationModel{
      *this,
      "invocation-model",
      llvm::cl::desc("Specifies the execution model used for invocations."),
      llvm::cl::init(IREE::ABI::InvocationModel::Sync),
      llvm::cl::values(
          clEnumValN(IREE::ABI::InvocationModel::Sync, "sync",
                     "Fully synchronous behavior with no fences."),
          clEnumValN(IREE::ABI::InvocationModel::CoarseFences, "coarse-fences",
                     "Exposes one wait fence for all inputs and one signal "
                     "fence for all outputs.")),
  };
};

// Adds a set of passes to the given pass manager that setup a module for use
// with bindings following the native IREE ABI.
void buildTransformPassPipeline(OpPassManager &passManager,
                                const InvocationOptions &invocationOptions);

void registerTransformPassPipeline();

//===----------------------------------------------------------------------===//
// IREE native ABI bindings support
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "iree/compiler/Bindings/Native/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "iree/compiler/Bindings/Native/Transforms/Passes.h.inc"

} // namespace mlir::iree_compiler::IREE::ABI

#endif // IREE_COMPILER_BINDINGS_NATIVE_TRANSFORMS_PASSES_H_
