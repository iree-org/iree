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
// Note: We need static members because they are shared across all derived
// instances to bind LLVM cl::opt registration at the single storage when
// multiple backends inherit from this class.
struct CodegenOptions {
  // Path to a module containing a tuning spec.
  static std::string tuningSpecPath;

  // Whether to add attributes for the tuner on root ops.
  static bool setTunerAttributes;

  // Whether to emit pipeline constraints for root ops.
  static bool emitPipelineConstraints;

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

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<CPUCodegenOptions>;

  // Applies opt-level-dependent defaults to the current option set.
  void setWithOptLevel(llvm::OptimizationLevel level);

  // Returns a CPUCodegenOptions with all opt-level-dependent defaults derived
  // from `level`. Uses a local OptionsBinder so the global flags are not
  // touched.
  static CPUCodegenOptions getWithOptLevel(llvm::OptimizationLevel level);
};

struct GPUCodegenOptions : CodegenOptions {
  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<GPUCodegenOptions>;

  // Applies opt-level-dependent defaults to the current option set.
  void setWithOptLevel(llvm::OptimizationLevel level);

  // Returns a GPUCodegenOptions with all opt-level-dependent defaults derived
  // from `level`. Uses a local OptionsBinder so the global flags are not
  // touched.
  static GPUCodegenOptions getWithOptLevel(llvm::OptimizationLevel level);
};

// Provide `operator<<` in the associated namespace so MLIR's pass-option
// printing (which goes through ADL via has_stream_operator) can serialize
// these values. We print a placeholder because the options are populated
// programmatically from a session-scoped instance, not from pass pipeline
// strings; there is no meaningful textual representation to round-trip.
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const CPUCodegenOptions &) {
  return os << "opaque";
}
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const GPUCodegenOptions &) {
  return os << "opaque";
}

} // namespace mlir::iree_compiler

// Specialize `llvm::cl::parser` for the codegen option structs to inherit
// `basic_parser` instead of the default `generic_parser_base`. This sidesteps
// `GenericOptionParser::findArgStrForValue`, which `llvm_unreachable`s on
// struct-typed values that have no registered enum entries.
//
// The parse methods are no-ops: these options never flow in from strings.
namespace llvm::cl {

extern template class basic_parser<mlir::iree_compiler::CPUCodegenOptions>;
template <>
class parser<mlir::iree_compiler::CPUCodegenOptions>
    : public basic_parser<mlir::iree_compiler::CPUCodegenOptions> {
public:
  parser(Option &o) : basic_parser(o) {}
  bool parse(Option &, StringRef, StringRef,
             mlir::iree_compiler::CPUCodegenOptions &);
  StringRef getValueName() const override { return "cpu codegen options"; }
  void printOptionDiff(const Option &, mlir::iree_compiler::CPUCodegenOptions,
                       const OptVal &, size_t) const;
  void anchor() override;
};

extern template class basic_parser<mlir::iree_compiler::GPUCodegenOptions>;
template <>
class parser<mlir::iree_compiler::GPUCodegenOptions>
    : public basic_parser<mlir::iree_compiler::GPUCodegenOptions> {
public:
  parser(Option &o) : basic_parser(o) {}
  bool parse(Option &, StringRef, StringRef,
             mlir::iree_compiler::GPUCodegenOptions &);
  StringRef getValueName() const override { return "gpu codegen options"; }
  void printOptionDiff(const Option &, mlir::iree_compiler::GPUCodegenOptions,
                       const OptVal &, size_t) const;
  void anchor() override;
};

} // namespace llvm::cl

#endif // IREE_COMPILER_CODEGEN_UTILS_CODEGENOPTIONS_H_
