// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/CodegenOptions.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::CPUCodegenOptions);
IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::GPUCodegenOptions);

namespace mlir::iree_compiler {

namespace {
// Single instance per option category, shared across all bind* methods so
// that splitting bindOptions across functions does not register duplicate
// categories with LLVM's command-line machinery.
llvm::cl::OptionCategory &commonCategory() {
  static llvm::cl::OptionCategory category("IREE Codegen Options");
  return category;
}
llvm::cl::OptionCategory &cpuCategory() {
  static llvm::cl::OptionCategory category("IREE CPU Codegen Options");
  return category;
}
} // namespace

std::string CodegenOptions::tuningSpecPath = "";
bool CodegenOptions::setTunerAttributes = false;
bool CodegenOptions::emitPipelineConstraints = false;

llvm::OptimizationLevel
mapCodegenPipelineOptLevel(CodegenPipelineOptLevel optLevel) {
  switch (optLevel) {
  case CodegenPipelineOptLevel::O0:
    return llvm::OptimizationLevel::O0;
  case CodegenPipelineOptLevel::O1:
    return llvm::OptimizationLevel::O1;
  case CodegenPipelineOptLevel::O2:
    return llvm::OptimizationLevel::O2;
  case CodegenPipelineOptLevel::O3:
    return llvm::OptimizationLevel::O3;
  }
  assert(false && "unhandled codegen pipeline optimization level");
  return llvm::OptimizationLevel::O0;
}

void CodegenOptions::bindOptions(OptionsBinder &binder) {
  llvm::cl::OptionCategory &category = commonCategory();

  binder.opt<std::string>(
      "iree-codegen-tuning-spec-path", tuningSpecPath, llvm::cl::cat(category),
      llvm::cl::desc("Path to a module containing a tuning spec (transform "
                     "dialect library). Accepts MLIR text (.mlir) and "
                     "bytecode (.mlirbc) formats."));

  binder.opt<bool>("iree-codegen-add-tuner-attributes", setTunerAttributes,
                   llvm::cl::cat(category),
                   llvm::cl::desc("Adds attribute for tuner."));

  // Deprecated alias for the old spelling.
  binder.opt<bool>(
      "iree-config-add-tuner-attributes", setTunerAttributes,
      Deprecated("use --iree-codegen-add-tuner-attributes instead"),
      llvm::cl::Hidden, llvm::cl::desc("Adds attribute for tuner."),
      llvm::cl::cat(category));

  binder.opt<bool>(
      "iree-codegen-experimental-verify-pipeline-constraints",
      emitPipelineConstraints, llvm::cl::cat(category),
      llvm::cl::desc("Emit and verify SMT pipeline constraints for root ops. "
                     "Implies --iree-codegen-add-tuner-attributes."));
}

void CPUCodegenOptions::bindOptLevelCascadeOptions(OptionsBinder &binder) {
  llvm::cl::OptionCategory &category = cpuCategory();

  auto initAtOpt = binder.optimizationLevel(
      "iree-llvmcpu-mlir-opt-level", optLevel,
      llvm::cl::desc("Optimization level for MLIR codegen passes."),
      llvm::cl::cat(category));

  binder.opt<bool>("iree-llvmcpu-reassociate-fp-reductions",
                   reassociateFpReductions,
                   {initAtOpt(llvm::OptimizationLevel::O0, false),
                    initAtOpt(llvm::OptimizationLevel::O2, true)},
                   llvm::cl::desc("Enables reassociation for FP reductions."),
                   llvm::cl::cat(category));
}

void CPUCodegenOptions::bindOptions(OptionsBinder &binder) {
  llvm::cl::OptionCategory &category = cpuCategory();
  CodegenOptions::bindOptions(binder);
  bindOptLevelCascadeOptions(binder);

  binder.opt<bool>("iree-llvmcpu-disable-distribution", disableDistribution,
                   llvm::cl::desc("Disable thread distribution in codegen."),
                   llvm::cl::cat(category));

  binder.opt<bool>(
      "iree-llvmcpu-fail-on-out-of-bounds-stack-allocation",
      failOnOutOfBoundsStackAllocation,
      llvm::cl::desc("Fail if the upper bound of dynamic stack allocation "
                     "cannot be solved."),
      llvm::cl::cat(category));

  binder.opt<bool>(
      "iree-llvmcpu-use-fast-min-max-ops", useFastMinMaxOps,
      llvm::cl::desc(
          "Use `arith.minf/maxf` instead of `arith.minimumf/maximumf` ops."),
      llvm::cl::cat(category));

  binder.opt<bool>(
      "iree-llvmcpu-skip-intermediate-roundings", skipIntermediateRoundings,
      llvm::cl::desc(
          "Allow skipping intermediate roundings. For example, in f16 matmul "
          "kernels on targets with only f32 arithmetic, we have to perform "
          "each multiply-accumulate in f32, and if this flag is false, then "
          "we have to round those f32 accumulators to the nearest f16 every "
          "time, which is slow."),
      llvm::cl::cat(category));

  binder.opt<bool>(
      "iree-llvmcpu-use-decompose-softmax-fuse", useSoftmaxInterFusion,
      llvm::cl::desc(
          "Enables inter-pass fusion for the DecomposeSoftmax pass."),
      llvm::cl::cat(category));

  binder.opt<bool>(
      "iree-llvmcpu-instrument-memory-accesses", instrumentMemoryAccesses,
      llvm::cl::desc(
          "Instruments memory reads and writes in dispatches for address "
          "tracking. Use with --iree-hal-instrument-dispatches=<buffer-size> "
          "and analyze results with iree-dump-instruments."),
      llvm::cl::cat(category));

  binder.opt<bool>("iree-llvmcpu-experimental-vectorize-to-transfer-gather",
                   enableTransferGather,
                   llvm::cl::desc("Experimental: enables vectorization to "
                                  "iree_vector_ext.transfer_gather."),
                   llvm::cl::cat(category));
}

void CPUCodegenOptions::setWithOptLevel(llvm::OptimizationLevel level) {
  // Run the opt-level cascade on a local binder restricted to this instance's
  // opt-level-sensitive fields. All bound storage lives on `*this` (a local
  // object on the caller's stack), so concurrent invocations from parallel
  // pass option defaults write only to thread-local memory.
  auto binder = OptionsBinder::local();
  binder.topLevelOpt("opt-level", level);
  bindOptLevelCascadeOptions(binder);
  binder.applyOptimizationDefaults();
}

CPUCodegenOptions
CPUCodegenOptions::getWithOptLevel(llvm::OptimizationLevel level) {
  CPUCodegenOptions opts;
  opts.setWithOptLevel(level);
  return opts;
}

void GPUCodegenOptions::bindOptLevelCascadeOptions(OptionsBinder & /*binder*/) {
  // No GPU-specific opt-level-sensitive fields yet.
}

void GPUCodegenOptions::bindOptions(OptionsBinder &binder) {
  CodegenOptions::bindOptions(binder);
  bindOptLevelCascadeOptions(binder);
}

void GPUCodegenOptions::setWithOptLevel(llvm::OptimizationLevel level) {
  auto binder = OptionsBinder::local();
  binder.topLevelOpt("opt-level", level);
  bindOptLevelCascadeOptions(binder);
  binder.applyOptimizationDefaults();
}

GPUCodegenOptions
GPUCodegenOptions::getWithOptLevel(llvm::OptimizationLevel level) {
  GPUCodegenOptions opts;
  opts.setWithOptLevel(level);
  return opts;
}

} // namespace mlir::iree_compiler
