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

namespace mlir::iree_compiler {

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
// NOTE: The plugin system does not have a good way to populate CL help
// messages, so we err on the side of being helpful and populating plugin
// options here, even though it is a layering violation.
#ifdef IREE_COMPILER_PLUGIN_HAVE_STATIC_INPUT_STABLEHLO
          "  =stablehlo     - Legalize from StableHLO ops (including VHLO deserialization).\n"
          "  =stablehlo_xla - Legalize from StableHLO ops (including VHLO deserialization and XLA de-tupling).\n"
#endif // IREE_COMPILER_PLUGIN_HAVE_STATIC_INPUT_STABLEHLO
#ifdef IREE_COMPILER_PLUGIN_HAVE_STATIC_INPUT_TOSA
          "  =tosa          - Legalize from TOSA ops.\n"
#endif  // IREE_COMPILER_PLUGIN_HAVE_STATIC_INPUT_TOSA
#ifdef IREE_COMPILER_PLUGIN_HAVE_STATIC_INPUT_TORCH
          "  =tm_tensor     - Legalize a subset of Torch input ops.\n"
          "  =torch         - Legalize from the 'torch' dialect.\n"
#endif  // IREE_COMPILER_PLUGIN_HAVE_STATIC_INPUT_TORCH
          "  =*             - An extensible input type defined in a plugin."
          // clang-format on
          ),
      llvm::cl::cat(category));

  binder.opt<bool>(
      "iree-input-demote-i64-to-i32", demoteI64ToI32,
      llvm::cl::desc("Converts all i64 ops and values into i32 counterparts "
                     "unconditionally before main global optimizations."),
      llvm::cl::cat(category));
  binder.opt<bool>(
      "iree-input-demote-f32-to-f16", demoteF32ToF16,
      llvm::cl::desc("Converts all f32 ops and values into f16 counterparts "
                     "unconditionally before main global optimizations."),
      llvm::cl::cat(category));
  binder.opt<bool>(
      "iree-input-demote-f64-to-f32", demoteF64ToF32,
      llvm::cl::desc("Converts all f64 ops and values into f32 counterparts "
                     "unconditionally before main global optimizations."),
      llvm::cl::cat(category));
  binder.opt<bool>(
      "iree-input-promote-f16-to-f32", promoteF16ToF32,
      llvm::cl::desc("Converts all f16 ops and values into f32 counterparts "
                     "unconditionally before main global optimizations."),
      llvm::cl::cat(category));
  binder.opt<bool>(
      "iree-input-promote-bf16-to-f32", promoteBF16ToF32,
      llvm::cl::desc("Converts all bf16 ops and values into f32 counterparts "
                     "unconditionally before main global optimizations."),
      llvm::cl::cat(category));
}

InputDialectOptions::Type InputDialectOptions::parseInputTypeMnemonic() {
  if (inputTypeMnemonic == "none") {
    return Type::none;
  } else if (inputTypeMnemonic == "auto") {
    return Type::auto_detect;
  } else {
    return Type::plugin;
  }
}

void PreprocessingOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category(
      "IREE options for apply custom preprocessing before normal IREE "
      "compilation flow");

  binder.opt<std::string>(
      "iree-preprocessing-pass-pipeline", preprocessingPassPipeline,
      llvm::cl::desc("Textual description of the pass pipeline to run before "
                     "running normal IREE compilation pipelines."),
      llvm::cl::cat(category));
  binder.opt<std::string>(
      "iree-preprocessing-transform-spec-filename",
      preprocessingTransformSpecFilename,
      llvm::cl::desc(
          "File name of a transform dialect spec to use for preprocessing."),
      llvm::cl::cat(category));
  binder.opt<std::string>(
      "iree-preprocessing-pdl-spec-filename", preprocessingPDLSpecFilename,
      llvm::cl::desc(
          "File name of a transform dialect spec to use for preprocessing."),
      llvm::cl::cat(category));
}

void GlobalOptimizationOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category(
      "IREE options for controlling global optimizations.");
  binder.opt<bool>(
      "iree-opt-aggressively-propagate-transposes",
      aggressiveTransposePropagation,
      llvm::cl::desc(
          "Propagates transposes to named ops even when the resulting op will "
          "be a linalg.generic"),
      llvm::cl::cat(category));
  binder.opt<bool>("iree-opt-outer-dim-concat", outerDimConcat,
                   llvm::cl::desc("Transposes all concatenations to happen"
                                  "along the outer most dimension."),
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
  binder.opt<int64_t>(
      "iree-opt-const-expr-max-size-increase-threshold",
      constExprMaxSizeIncreaseThreshold,
      llvm::cl::desc("Maximum byte size increase allowed for constant expr "
                     "hoisting policy to allow hoisting."),
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
      "iree-opt-import-parameters", parameterImportPaths,
      llvm::cl::desc("File paths to archives to import parameters from with an "
                     "optional `scope=` prefix."),
      llvm::cl::cat(category));
  binder.list<std::string>("iree-opt-import-parameter-keys",
                           parameterImportKeys,
                           llvm::cl::desc("List of parameter keys to import."),
                           llvm::cl::cat(category));
  binder.opt<int64_t>("iree-opt-import-parameter-maximum-size",
                      parameterImportMaximumSize,
                      llvm::cl::desc("Maximum size of parameters to import."),
                      llvm::cl::cat(category));

  binder.opt<std::string>(
      "iree-opt-export-parameters", parameterExportPath,
      llvm::cl::desc("File path to an archive to export parameters to with an "
                     "optional `scope=` prefix."),
      llvm::cl::cat(category));
  binder.opt<int64_t>(
      "iree-opt-export-parameter-minimum-size", parameterExportMinimumSize,
      llvm::cl::desc(
          "Minimum size of constants to export to the archive created in "
          "`iree-opt-export-parameter-archive-export-file`."),
      llvm::cl::cat(category));

  binder.opt<std::string>(
      "iree-opt-splat-parameters", parameterSplatExportFile,
      llvm::cl::desc(
          "File path to create a parameter archive of splat values out of all "
          "parameter backed globals."),
      llvm::cl::cat(category));

  binder.opt<bool>(
      "iree-opt-generalize-matmul", generalizeMatmul,
      llvm::cl::desc("Convert named matmul ops to linalg generic ops during "
                     "global optimization to enable better fusion."),
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

  binder.opt<InitializationMode>(
      "iree-scheduling-initialization-mode", initializationMode,
      llvm::cl::desc(
          "Specifies the initialization mode for parameters and globals."),
      llvm::cl::values(
          clEnumValN(InitializationMode::Synchronous, "sync",
                     "Synchronously initialize all parameters and globals "
                     "prior to returning from the module initializer."),
          clEnumValN(InitializationMode::Asynchronous, "async",
                     "Asynchronously initialize all parameters and globals and "
                     "return immediately from the module initializer without "
                     "waiting for them to complete.")),
      llvm::cl::cat(category));

  binder.opt<bool>(
      "iree-scheduling-optimize-bindings", optimizeBindings,
      llvm::cl::desc(
          "Enables binding fusion and dispatch site specialization."),
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

} // namespace mlir::iree_compiler
