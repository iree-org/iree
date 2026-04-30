// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UTILS_CODEGENOPTIONS_H_
#define IREE_COMPILER_CODEGEN_UTILS_CODEGENOPTIONS_H_

#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler {

// Bridge type for MLIR pass/pipeline options, which cannot store
// llvm::OptimizationLevel directly because it is a final class.
enum class CodegenPipelineOptLevel {
  O0 = 0,
  O1 = 1,
  O2 = 2,
  O3 = 3,
};

// Maps the pass/pipeline bridge enum to llvm::OptimizationLevel.
llvm::OptimizationLevel
mapCodegenPipelineOptLevel(CodegenPipelineOptLevel optLevel);

// A base class that defines common codegen options that are shared across
// different backends (e.g., CPU and GPU). Derived classes can add
// backend-specific options as needed.
//
// Note: We keep a few members static so that CPU and GPU subclasses' global
// CLI bindings share the same storage (and so dedup-by-name works across the
// two FromFlags singletons). Static storage is only safe to bind on the
// global binder singleton, not on per-instance local binders; see
// bindOptLevelCascadeOptions() on each subclass for the thread-safe cascade
// path.
struct CodegenOptions {
  // Path to a module containing a tuning spec.
  static std::string tuningSpecPath;

  // Whether to add attributes for the tuner on root ops.
  static bool setTunerAttributes;

  // Whether to emit pipeline constraints for root ops.
  static bool emitPipelineConstraints;

  // Registers the shared CLI flags that back CodegenOptions' static members.
  // Must only be called on the global binder (via the FromFlags singleton);
  // binding on a local binder would write through shared static storage and
  // is not safe under MLIR's parallel pass pipelines.
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

  // Use arith.minf/maxf instead of arith.minimumf/maximumf.
  bool useFastMinMaxOps = false;

  // Allow skipping intermediate roundings (e.g., in f16 matmul on f32
  // hardware).
  bool skipIntermediateRoundings = true;

  // Enables inter-pass fusion for the DecomposeSoftmax pass.
  bool useSoftmaxInterFusion = true;

  // Instruments memory reads and writes in dispatches for address tracking.
  bool instrumentMemoryAccesses = false;

  // Enables experimental vectorization to transfer_gather.
  bool enableTransferGather = false;

  // Registers all CPU codegen CLI flags, including the opt-level cascade and
  // the shared CodegenOptions statics. Intended for the FromFlags singleton
  // on the global binder only.
  void bindOptions(OptionsBinder &binder);

  // Registers only the opt-level-sensitive instance fields on the provided
  // binder. Safe to call per-instance on a local binder: every bound flag
  // points at a field on `*this` (thread-local), so concurrent invocations
  // from pass option defaults do not race.
  void bindOptLevelCascadeOptions(OptionsBinder &binder);

  using FromFlags = OptionsFromFlags<CPUCodegenOptions>;

  // Applies opt-level-dependent defaults to this instance by running the
  // binder-driven cascade on a local binder bound solely to this instance's
  // opt-level-sensitive fields.
  void setWithOptLevel(llvm::OptimizationLevel level);

  // Returns a CPUCodegenOptions with all opt-level-dependent defaults derived
  // from `level`.
  static CPUCodegenOptions getWithOptLevel(llvm::OptimizationLevel level);
};

struct GPUCodegenOptions : CodegenOptions {
  // Registers all GPU codegen CLI flags, including the shared CodegenOptions
  // statics. Intended for the FromFlags singleton on the global binder only.
  void bindOptions(OptionsBinder &binder);

  // Registers only the opt-level-sensitive instance fields on the provided
  // binder. Today GPU has no opt-level-sensitive fields, so this is a no-op.
  // Kept parallel to the CPU side so both backends share the same cascade
  // shape; extend here when GPU adds opt-level-sensitive defaults.
  void bindOptLevelCascadeOptions(OptionsBinder &binder);

  using FromFlags = OptionsFromFlags<GPUCodegenOptions>;

  // Applies opt-level-dependent defaults to this instance by running the
  // binder-driven cascade on a local binder bound solely to this instance's
  // opt-level-sensitive fields.
  void setWithOptLevel(llvm::OptimizationLevel level);

  // Returns a GPUCodegenOptions with all opt-level-dependent defaults derived
  // from `level`.
  static GPUCodegenOptions getWithOptLevel(llvm::OptimizationLevel level);
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_UTILS_CODEGENOPTIONS_H_
