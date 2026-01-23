// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_CODEGENOPTIONS_H_
#define IREE_COMPILER_CODEGEN_UTILS_CODEGENOPTIONS_H_

#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler {

struct CodegenOptions {
  llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0;

  // Emit warnings for slow codegen paths (e.g., type emulation, missing
  // intrinsics, suboptimal configurations).
  bool emitPerformanceWarnings = false;

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<CodegenOptions>;
};

struct CPUCodegenOptions {
  llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0;

  // Disable thread distribution in codegen.
  bool disableDistribution = false;

  // Fail if the upper bound of dynamic stack allocation cannot be solved.
  bool failOnOutOfBoundsStackAllocation = true;

  // Enables reassociation for FP reductions.
  bool reassociateFpReductions = true;

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<CPUCodegenOptions>;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_UTILS_CODEGENOPTIONS_H_
