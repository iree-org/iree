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

  bool demoteI64ToI32 = true;
  bool demoteF64ToF32 = true;
  bool promoteBF16ToF32 = false;

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<InputDialectOptions>;
};

// Options controlling high level optimizations.
struct GlobalOptimizationOptions {
  // Gate various type based demotion passes that run before anything else.
  bool demoteF64ToF32 = true;
  bool demoteF32ToF16 = false;
  bool promoteF16ToF32 = false;
  bool promoteBF16ToF32 = false;
  bool demoteI64ToI32 = false;

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

  // Maximum byte size increase allowed for constant expr hoisting policy to
  // allow hoisting. The threshold is 1MB by default.
  int64_t constExprMaxSizeIncreaseThreshold = 1024 * 1024;

  // File path to create a parameter archive out of global initial values.
  std::string parameterArchivePath = "";

  // Optional namespace to use for the created parameter archive.
  std::string parameterNamespace = "";

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
  // Enables fusing bindings with the same underlying storage.
  bool optimizeBindings = true;

  // TODO(benvanik): favor size/speed/etc for partitioning.
  // TODO(benvanik): execution model to optimize for (unified/discrete memory,
  //                 single/multiple processors, etc).

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<SchedulingOptions>;
};

struct PreprocessingOptions {
  std::string preprocessingPassPipeline;
  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<PreprocessingOptions>;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_PIPELINES_OPTIONS_H_
