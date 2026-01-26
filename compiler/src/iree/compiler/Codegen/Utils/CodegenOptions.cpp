// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/CodegenOptions.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::CPUCodegenOptions);

namespace mlir::iree_compiler {

void CPUCodegenOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category("IREE CPU Codegen Options");

  auto init_at_opt = binder.optimizationLevel(
      "iree-llvmcpu-opt-level", optLevel,
      llvm::cl::desc("Optimization level for CPU codegen."),
      llvm::cl::cat(category));

  binder.opt<bool>("iree-llvmcpu-disable-distribution", disableDistribution,
                   llvm::cl::desc("Disable thread distribution in codegen."),
                   llvm::cl::cat(category));

  binder.opt<bool>(
      "iree-llvmcpu-fail-on-out-of-bounds-stack-allocation",
      failOnOutOfBoundsStackAllocation,
      llvm::cl::desc("Fail if the upper bound of dynamic stack allocation "
                     "cannot be solved."),
      llvm::cl::cat(category));

  binder.opt<bool>("iree-llvmcpu-reassociate-fp-reductions",
                   reassociateFpReductions,
                   {init_at_opt(llvm::OptimizationLevel::O0, false),
                    init_at_opt(llvm::OptimizationLevel::O1, true)},
                   llvm::cl::desc("Enables reassociation for FP reductions."),
                   llvm::cl::cat(category));
}

} // namespace mlir::iree_compiler
