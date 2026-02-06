// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/CodegenOptions.h"

#include <mutex>

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::CPUCodegenOptions);
IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::GPUCodegenOptions);

namespace mlir::iree_compiler {

std::string CodegenOptions::tuningSpecPath = "";

void CodegenOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category("IREE Codegen Options");

  // Use std::call_once to ensure the option is registered exactly once,
  // even when multiple derived classes (CPU/GPU) call this method.
  static std::once_flag bindFlag;
  std::call_once(bindFlag, [&]() {
    binder.opt<std::string>(
        "iree-codegen-tuning-spec-path", tuningSpecPath,
        llvm::cl::cat(category),
        llvm::cl::desc("Path to a module containing a tuning spec (transform "
                       "dialect library). Accepts MLIR text (.mlir) and "
                       "bytecode (.mlirbc) formats."));
  });
}

void CPUCodegenOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category("IREE CPU Codegen Options");
  CodegenOptions::bindOptions(binder);

  auto initAtOpt = binder.optimizationLevel(
      "iree-llvmcpu-mlir-opt-level", optLevel,
      llvm::cl::desc("Optimization level for MLIR codegen passes."),
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
                   {initAtOpt(llvm::OptimizationLevel::O0, false),
                    initAtOpt(llvm::OptimizationLevel::O2, true)},
                   llvm::cl::desc("Enables reassociation for FP reductions."),
                   llvm::cl::cat(category));
}

void GPUCodegenOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category("IREE GPU Codegen Options");
  CodegenOptions::bindOptions(binder);

  binder.opt<bool>(
      "iree-llvmgpu-enable-prefetch", enablePrefetch,
      llvm::cl::desc("Enable prefetch in the vector distribute pipeline."),
      llvm::cl::cat(category));
}

} // namespace mlir::iree_compiler
