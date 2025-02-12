// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PIPELINES_OPTIONS_H_
#define IREE_COMPILER_PIPELINES_OPTIONS_H_

#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler {

struct BindingOptions {
  // Whether to include runtime support functions for the IREE native ABI.
  bool native = true;
  // Whether to include runtime support functions required for the IREE TFLite
  // API compatibility bindings.
  bool tflite = false;

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<BindingOptions>;
};

// The transformation to apply to the input prior to main compiler execution.
// These input pipelines are purposefully primitive and mainly focused on
// test case/reproducers as opposed to anything that should be coming from
// a user. For user/framework level interfacing, a dedicated importer likely
// needs to be created in order to represent whole-module level framework
// quirks. These are just about the ops in the functions.
struct InputDialectOptions {
  // Built-in input types, represented by an enum.
  enum class Type {
    // Applies no input transformation. Only supported core and extension ops
    // are supported.
    none,
    // Analyses the input to determine what input dialect pipeline to use.
    auto_detect,
    // A named input pipeline from a plugin. If set, then 'pluginInputPipeline'
    // must be set.
    plugin,
  };
  // The flag value is captured into spec by the CL system and it must be
  // interpreted by parseInputTypeSpec.
  std::string inputTypeMnemonic{"auto"};

  // Parses the user-provided inputTypeMnemonic, returning a recognized Type
  // enumeration as appropriate. If the returned type is `plugin`, then it is
  // a custom input type and the raw inputTypeMnemonic should be passed to the
  // plugin system for resolution.
  Type parseInputTypeMnemonic();

  // Gate various type based demotion passes that run before anything else.
  bool demoteI64ToI32 = false;
  bool demoteF32ToF16 = false;
  bool demoteF64ToF32 = true;
  bool promoteF16ToF32 = false;
  bool promoteBF16ToF32 = false;

  // Perfoms early optimizations geared towards optimizing/simplifying the
  // types of integer arithmetic inefficiencies that frontends typically
  // include and which are implicated in blocking downstream optimizations.
  bool optimizeIndexArithmetic = true;

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<InputDialectOptions>;
};

// Allows specifying one of several ways of doing custom transformations at the
// pre-processing phase, multiple ways may be used and they are run in order:
//   1. Through a preprocessing pass pipeline.
//   2. Through a Transform dialect spec file.
//   3. Through a PDL spec file.
struct PreprocessingOptions {
  std::string preprocessingPassPipeline;
  std::string preprocessingTransformSpecFilename;
  std::string preprocessingPDLSpecFilename;

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<PreprocessingOptions>;
};

// Options controlling high level optimizations.
struct GlobalOptimizationOptions {
  // Maximum byte size increase allowed for constant expr hoisting policy to
  // allow hoisting. The threshold is 1MB by default.
  int64_t constExprMaxSizeIncreaseThreshold = 1024 * 1024;

  // File paths to archives to import parameters from with an optional
  // `scope=` prefix.
  std::vector<std::string> parameterImportPaths;
  // List of parameter keys to import. Any matching keys from any scope will be
  // imported.
  std::vector<std::string> parameterImportKeys;
  // Maximum size of parameters to import or 0 to disable automatic import.
  int64_t parameterImportMaximumSize = 0;

  // File path to an archive to export parameters to with an optional
  // `scope=` prefix.
  std::string parameterExportPath;
  // Minimum size of constants to export as parameters.
  int64_t parameterExportMinimumSize = 0;

  // File path to create a splat parameter archive out of all parameters in the
  // module.
  std::string parameterSplatExportFile = "";

  // Enables aggressive propagation of transposes to the inputs of named ops,
  // rewriting named ops as fused generics.
  bool aggressiveTransposePropagation = false;

  // Enables transposing all concatenations to the outer most dimension.
  bool outerDimConcat = false;

  // Enables data tiling.
  bool dataTiling = true;

  // Enables const-expr hoisting into globals.
  bool constExprHoisting = true;

  // Enables recursive evaluation of immutable globals using the compiler
  // and runtime.
  bool constEval = true;

  // Optimizations to reduce numeric precision where it is safe to do so.
  bool numericPrecisionReduction = false;

  // Strips debug assertions after any useful information has been extracted.
  bool stripAssertions = false;

  // Converts linalg named matmul ops to linalg generic ops.
  bool generalizeMatmul = false;

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<GlobalOptimizationOptions>;
};

// Options controlling scheduling across host/device.
struct SchedulingOptions {
  // Defines the execution model used for scheduling work.
  enum class ExecutionModel {
    // Host-local code only that does not need execution scheduling.
    // Disables flow/stream/hal pipelines.
    HostOnly = 0,
    // Full HAL using asynchronous host/device execution internally but
    // exporting functions as if synchronous.
    AsyncInternal = 1,
    // Full HAL using asynchronous host/device execution both internally and
    // externally.
    AsyncExternal = 2,
    // Inline host-local in-process execution with executable code statically
    // linked into the host program.
    // (Currently) only supports the `vmvx-inline` HAL target backend.
    InlineStatic = 3,
    // Inline host-local in-process execution using dynamic executables.
    // Only supports CPU HAL target backends that produce executable libraries.
    InlineDynamic = 4,
  };
  // Program execution model specifying scheduling behavior.
  ExecutionModel executionModel = ExecutionModel::AsyncInternal;

  // Defines the behavior of initialization.
  enum class InitializationMode {
    // Synchronously initialize all parameters and globals prior to returning
    // from the module initializer.
    Synchronous = 0,
    // Asynchronously initialize all parameters and globals and return
    // immediately from the module initializer without waiting for them to
    // complete. Subsequent invocations will queue waiting for any dependencies
    // they have on the initialized values.
    Asynchronous = 1,
  };
  // Initialization mode for parameters and globals.
  InitializationMode initializationMode = InitializationMode::Synchronous;

  // TODO(benvanik): favor size/speed/etc for partitioning.
  // TODO(benvanik): execution model to optimize for (unified/discrete memory,
  //                 single/multiple processors, etc).

  // Enables fusing bindings with the same underlying storage.
  bool optimizeBindings = true;

  // TODO(benvanik): find a way to share this with
  // Stream/Transforms/Passes.h w/o circular deps.
  // Defines the output format of a dump pass.
  enum class DumpOutputFormat {
    // Dumping disabled.
    None = 0,
    // Human-readable pretty printing.
    Pretty = 1,
    // Pretty printing with additional information that can result in large
    // dumps.
    Verbose = 2,
    // Comma separated values for throwing into Sheets.
    CSV = 3,
    // JSON format for better structure and data exchange.
    JSON = 4,
  };
  // Enables and specifies the the format for a stream statistics dump.
  DumpOutputFormat dumpStatisticsFormat = DumpOutputFormat::None;
  // File path to write statistics to; or `` for stderr or `-` for stdout.
  std::string dumpStatisticsFile = "";

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<SchedulingOptions>;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_PIPELINES_OPTIONS_H_
