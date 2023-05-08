// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Pipelines/Options.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::BindingOptions);
IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::InputDialectOptions);
IREE_DEFINE_COMPILER_OPTION_FLAGS(
    mlir::iree_compiler::HighLevelOptimizationOptions);
IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::SchedulingOptions);
IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::PreprocessingOptions);

namespace mlir {
namespace iree_compiler {

void BindingOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category(
      "IREE translation binding support options.");

  binder.opt<bool>(
      "iree-native-bindings-support", native,
      llvm::cl::desc(
          "Include runtime support for native IREE ABI-compatible bindings."),
      llvm::cl::cat(category));
  binder.opt<bool>("iree-tflite-bindings-support", tflite,
                   llvm::cl::desc("Include runtime support for the IREE TFLite "
                                  "compatibility bindings."),
                   llvm::cl::cat(category));
}

void InputDialectOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category(
      "IREE options for controlling the input transformations to apply.");

  binder.opt<InputDialectOptions::Type>(
      "iree-input-type", type,
      llvm::cl::desc("Specifies the input program representation."),
      llvm::cl::values(
          clEnumValN(InputDialectOptions::Type::none, "none",
                     "No input dialect transformation."),
          clEnumValN(InputDialectOptions::Type::auto_detect, "auto",
                     "Analyze the input program to choose conversion.")
  // clang-format off
#ifdef IREE_HAVE_MHLO_INPUT
        , clEnumValN(InputDialectOptions::Type::mhlo, "mhlo",
                     "Legalize from MHLO ops.")
        , clEnumValN(InputDialectOptions::Type::xla, "xla",
            "Legalize from MHLO ops (with XLA cleanup preprocessing).")
        , clEnumValN(InputDialectOptions::Type::stablehlo_experimental,
            "stablehlo_experimental",
            "Legalize from StableHLO ops. WARNING: This is work in progress.")
#endif  // IREE_HAVE_MHLO_INPUT
#ifdef IREE_HAVE_TORCH_INPUT
        , clEnumValN(InputDialectOptions::Type::tm_tensor, "tm_tensor",
                     "Legalize from TMTensor ops.")
#endif  // IREE_HAVE_TORCH_INPUT
#ifdef IREE_HAVE_TOSA_INPUT
        , clEnumValN(InputDialectOptions::Type::tosa, "tosa",
                     "Legalize from TOSA ops.")
#endif  // IREE_HAVE_TOSA_INPUT
          ),
      // clang-format on
      llvm::cl::cat(category));
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

  binder.opt<ExecutionModel>(
      "iree-execution-model", executionModel,
      llvm::cl::desc("Specifies the execution model used for scheduling tensor "
                     "compute operations."),
      llvm::cl::values(
          clEnumValN(
              ExecutionModel::HostOnly, "host-only",
              "Host-local code only that does not need execution scheduling."),
          clEnumValN(ExecutionModel::AsyncInternal, "async-internal",
                     "Full HAL using asynchronous host/device execution "
                     "internally but exporting functions as if synchronous."),
          clEnumValN(ExecutionModel::AsyncExternal, "async-external",
                     "Full HAL using asynchronous host/device execution both "
                     "internally and externally."),
          clEnumValN(ExecutionModel::InlineStatic, "inline-static",
                     "Inline host-local in-process execution with executable "
                     "code statically linked into the host program."),
          clEnumValN(ExecutionModel::InlineDynamic, "inline-dynamic",
                     "Inline host-local in-process execution using dynamic "
                     "executables.")),
      llvm::cl::cat(category));

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

void PreprocessingOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category(
      "IREE options for apply custom preprocessing before normal IREE "
      "compilation flow");

  binder.opt<std::string>(
      "iree-preprocessing-pass-pipeline", preprocessingPassPipeline,
      llvm::cl::desc("Textual description of the pass pipeline to run before "
                     "running normal IREE compilation pipelines"),
      llvm::cl::cat(category));
}

}  // namespace iree_compiler
}  // namespace mlir
