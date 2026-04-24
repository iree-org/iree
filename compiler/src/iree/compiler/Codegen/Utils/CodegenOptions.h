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

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<CPUCodegenOptions>;
};

struct GPUCodegenOptions : CodegenOptions {
  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<GPUCodegenOptions>;
};

} // namespace mlir::iree_compiler

// Declares the machinery required to use `TYPE` as an MLIR pass option:
//
//   (1) `operator<<` in the type's associated namespace, so MLIR's pass-option
//       printing (which goes through ADL via has_stream_operator) can serialize
//       the value. We print nothing because the options are populated
//       programmatically from a session-scoped instance, not from pass
//       pipeline strings.
//   (2) `llvm::cl::parser<TYPE>` specialized to inherit `basic_parser` instead
//       of the default `generic_parser_base`. This sidesteps
//       `GenericOptionParser::findArgStrForValue`, which `llvm_unreachable`s
//       on struct-typed values that have no registered enum entries.
//
// The parse method is a no-op: these options never flow in from strings.
#define IREE_DECLARE_CODEGEN_OPTIONS_PASS_OPTION(TYPE, VALUE_NAME_STR)         \
  namespace mlir::iree_compiler {                                              \
  inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TYPE &) {  \
    return os;                                                                 \
  }                                                                            \
  } /* namespace mlir::iree_compiler */                                        \
                                                                               \
  namespace llvm::cl {                                                         \
  extern template class basic_parser<TYPE>;                                    \
  template <>                                                                  \
  class parser<TYPE> : public basic_parser<TYPE> {                             \
  public:                                                                      \
    parser(Option &O) : basic_parser(O) {}                                     \
    bool parse(Option &O, StringRef ArgName, StringRef Arg, TYPE &Val);        \
    StringRef getValueName() const override { return VALUE_NAME_STR; }         \
    void printOptionDiff(const Option &O, TYPE V, const OptVal &Default,       \
                         size_t GlobalWidth) const;                            \
    void anchor() override;                                                    \
  };                                                                           \
  } /* namespace llvm::cl */

IREE_DECLARE_CODEGEN_OPTIONS_PASS_OPTION(mlir::iree_compiler::CPUCodegenOptions,
                                         "cpu codegen options")
IREE_DECLARE_CODEGEN_OPTIONS_PASS_OPTION(mlir::iree_compiler::GPUCodegenOptions,
                                         "gpu codegen options")
#undef IREE_DECLARE_CODEGEN_OPTIONS_PASS_OPTION

#endif // IREE_COMPILER_CODEGEN_UTILS_CODEGENOPTIONS_H_
