// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_CODEGENOPTIONS_H_
#define IREE_COMPILER_CODEGEN_UTILS_CODEGENOPTIONS_H_

#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler {

// A base class that defines common codegen options that are shared across
// different backends (e.g., CPU and GPU). Derived classes can add
// backend-specific options as needed.
//
// Note: We need static members because they are shared across all derived
// instances to avoid duplicate LLVM cl::opt registration when multiple backends
// inherit from this class.
struct CodegenOptions {
  // Path to a module containing a tuning spec.
  static std::string tuningSpecPath;

  void bindOptions(OptionsBinder &binder);
};

struct CPUCodegenOptions : CodegenOptions {
  llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0;

  // Disable thread distribution in codegen.
  bool disableDistribution = false;

  // Fail if the upper bound of dynamic stack allocation cannot be solved.
  bool failOnOutOfBoundsStackAllocation = true;

  // Enables reassociation for FP reductions.
  bool reassociateFpReductions = false;

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<CPUCodegenOptions>;
};

struct GPUCodegenOptions : CodegenOptions {
  // Enable prefetch in the vector distribute pipeline.
  bool enablePrefetch = false;

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<GPUCodegenOptions>;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_UTILS_CODEGENOPTIONS_H_
