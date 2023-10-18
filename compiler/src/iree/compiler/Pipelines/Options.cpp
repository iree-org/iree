// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Pipelines/Options.h"

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::BindingOptions);
IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::InputDialectOptions);
IREE_DEFINE_COMPILER_OPTION_FLAGS(
    mlir::iree_compiler::GlobalOptimizationOptions);
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
  binder.opt<std::string>(
      "iree-input-type", inputTypeMnemonic,
      llvm::cl::desc(
          // clang-format off
          "Specifies the input program representation:\n"
          "  =none          - No input dialect transformation.\n"
          "  =auto          - Analyze the input program to choose conversion.\n"
#ifdef IREE_HAVE_STABLEHLO_INPUT
          "  =stablehlo     - Legalize from StableHLO ops.\n"
          "  =stablehlo_xla - Legalize from StableHLO ops (with XLA cleanup preprocessing).\n"
#endif // IREE_HAVE_STABLEHLO_INPUT
#ifdef IREE_HAVE_TOSA_INPUT
          "  =tosa          - Legalize from TOSA ops.\n"
#endif  // IREE_HAVE_TOSA_INPUT
// NOTE: The plugin system does not have a good way to populate CL help
// messages, so we err on the side of being helpful and populating Torch
// options here, even though it is a layering violation.
#ifdef IREE_COMPILER_PLUGIN_HAVE_STATIC_TORCH_IREE
          "  =tm_tensor     - Legalize a subset of Torch input ops.\n"
          "  =torch         - Legalize from the 'torch' dialect.\n"
#endif  // IREE_COMPILER_PLUGIN_HAVE_STATIC_TORCH_IREE
          "  =*             - An extensible input type defined in a plugin."
          // clang-format on
          ),
      llvm::cl::cat(category));

#ifdef IREE_HAVE_STABLEHLO_INPUT
  binder.opt<bool>(
      "iree-input-demote-i64-to-i32", demoteI64ToI32,
      llvm::cl::desc("Converts all i64 ops and values into i32 counterparts."),
      llvm::cl::cat(category));

  binder.opt<bool>(
      "iree-input-demote-f64-to-f32", demoteF64ToF32,
      llvm::cl::desc("Converts all f64 ops and values into f32 counterparts."),
      llvm::cl::cat(category));

  binder.opt<bool>(
      "iree-input-promote-bf16-to-f32", promoteBF16ToF32,
      llvm::cl::desc("Converts all bf16 ops and values into f32 counterparts."),
      llvm::cl::cat(category));
#endif // IREE_HAVE_STABLEHLO_INPUT
}

InputDialectOptions::Type InputDialectOptions::parseInputTypeMnemonic() {
  if (inputTypeMnemonic == "none") {
    return Type::none;
  } else if (inputTypeMnemonic == "auto") {
    return Type::auto_detect;
#ifdef IREE_HAVE_STABLEHLO_INPUT
  } else if (inputTypeMnemonic == "stablehlo") {
    return Type::stablehlo;
  } else if (inputTypeMnemonic == "stablehlo_xla") {
    return Type::stablehlo_xla;
#endif
#ifdef IREE_HAVE_TOSA_INPUT
  } else if (inputTypeMnemonic == "tosa") {
    return Type::tosa;
#endif
  } else {
    return Type::plugin;
  }
}

void GlobalOptimizationOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category(
      "IREE options for controlling global optimizations.");
  // Type promotion/demotion options.
  binder.opt<bool>(
      "iree-opt-demote-f64-to-f32", demoteF64ToF32,
      llvm::cl::desc("Converts all f64 ops and values into f32 counterparts "
                     "unconditionally before main global optimizations."),
      llvm::cl::cat(category));
  binder.opt<bool>(
      "iree-opt-demote-f32-to-f16", demoteF32ToF16,
      llvm::cl::desc("Converts all f32 ops and values into f16 counterparts "
                     "unconditionally before main global optimizations."),
      llvm::cl::cat(category));
  binder.opt<bool>(
      "iree-opt-promote-f16-to-f32", promoteF16ToF32,
      llvm::cl::desc("Converts all f16 ops and values into f32 counterparts "
                     "unconditionally before main global optimizations."),
      llvm::cl::cat(category));
  binder.opt<bool>(
      "iree-opt-promote-bf16-to-f32", promoteBF16ToF32,
      llvm::cl::desc("Converts all bf16 ops and values into f32 counterparts "
                     "unconditionally before main global optimizations."),
      llvm::cl::cat(category));
  binder.opt<bool>(
      "iree-opt-demote-i64-to-i32", demoteI64ToI32,
      llvm::cl::desc("Converts all i64 ops and values into i32 counterparts "
                     "unconditionally before main global optimizations."),
      llvm::cl::cat(category));

  binder.opt<bool>("iree-opt-data-tiling", dataTiling,
                   llvm::cl::desc("Enables data tiling path."),
                   llvm::cl::cat(category));

  binder.opt<bool>(
      "iree-opt-const-eval", constEval,
      llvm::cl::desc("Enables eager evaluation of constants using the full "
                     "compiler and runtime (on by default)."),
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
  binder.list<std::string>(
      "iree-opt-extern-dispatch-pattern-module",
      customDispatchPatternModuleFileNames,
      llvm::cl::desc("File path to custom dispatch rewrite pattern module."),
      llvm::cl::ZeroOrMore, llvm::cl::cat(category));
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

  binder.opt<bool>(
      "iree-scheduling-optimize-bindings", optimizeBindings,
      llvm::cl::desc(
          "Enables binding fusion and dispatch site specialization."),
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

} // namespace iree_compiler
} // namespace mlir
