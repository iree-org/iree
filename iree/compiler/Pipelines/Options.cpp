// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Pipelines/Options.h"

namespace mlir {
namespace iree_compiler {

void BindingOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory bindingOptionsCategory(
      "IREE translation binding support options.");

  binder.opt<bool>(
      "iree-native-bindings-support", native,
      llvm::cl::desc(
          "Include runtime support for native IREE ABI-compatible bindings."),
      llvm::cl::cat(bindingOptionsCategory));
  binder.opt<bool>("iree-tflite-bindings-support", tflite,
                   llvm::cl::desc("Include runtime support for the IREE TFLite "
                                  "compatibility bindings."),
                   llvm::cl::cat(bindingOptionsCategory));
}

void InputDialectOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory inputDialectOptions(
      "IREE options for controlling the input transformations to apply.");

  binder.opt<InputDialectOptions::Type>(
      "iree-input-type", type,
      llvm::cl::desc("Specifies the input program representation."),
      llvm::cl::values(
          clEnumValN(InputDialectOptions::Type::none, "none",
                     "No input dialect transformation."),
          clEnumValN(InputDialectOptions::Type::tosa, "tosa",
                     "Legalize from TOSA ops."),
          clEnumValN(InputDialectOptions::Type::mhlo, "mhlo",
                     "Legalize from MHLO ops."),
#ifdef IREE_HAVE_TORCH_MLIR_DIALECTS
          clEnumValN(InputDialectOptions::Type::tm_tensor, "tm_tensor",
                     "Legalize from TMTensor ops."),
#endif
          clEnumValN(
              InputDialectOptions::Type::xla, "xla",
              "Legalize from MHLO ops (with XLA cleanup preprocessing).")),
      llvm::cl::cat(inputDialectOptions));
}

void HighLevelOptimizationOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category(
      "IREE options for controlling high level optimizations.");

  binder.opt<bool>(
      "iree-opt-const-eval", constEval,
      llvm::cl::desc("Enables eager evaluation of constants using the full "
                     "compiler and runtime."),
      llvm::cl::cat(category));
  binder.opt<bool>(
      "iree-opt-const-expr-hoisting", constExprHoisting,
      llvm::cl::desc(
          "Hoists the results of latent constant expressions into immutable "
          "global initializers for evaluation at program load."),
      llvm::cl::cat(category));
  binder.opt<bool>(
      "iree-opt-numeric-precision-reduction", numericPrecisionReduction,
      llvm::cl::desc(
          "Reduces numeric precision to lower bit depths where possible."),
      llvm::cl::cat(category));
  binder.opt<bool>("iree-opt-strip-assertions", stripAssertions,
                   llvm::cl::desc("Strips debug assertions after any useful "
                                  "information has been extracted."),
                   llvm::cl::cat(category));
}

void SchedulingOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category(
      "IREE options for controlling host/device scheduling.");

  binder.opt<DumpOutputFormat>(
      "iree-scheduling-dump-statistics-format", dumpStatisticsFormat,
      llvm::cl::desc("Dumps statistics in the specified output format."),
      llvm::cl::cat(category),
      llvm::cl::values(
          clEnumValN(DumpOutputFormat::Pretty, "pretty",
                     "Human-readable pretty printed output."),
          clEnumValN(DumpOutputFormat::Verbose, "verbose",
                     "Pretty printed output with additional IR."),
          clEnumValN(DumpOutputFormat::CSV, "csv", "Comma separated values."),
          clEnumValN(DumpOutputFormat::JSON, "json",
                     "JSON output with structures for data exchange")));
  binder.opt<std::string>("iree-scheduling-dump-statistics-file",
                          dumpStatisticsFile,
                          llvm::cl::desc("File path to write statistics to; or "
                                         "`` for stderr or `-` for stdout."),
                          llvm::cl::cat(category));
}

}  // namespace iree_compiler
}  // namespace mlir
