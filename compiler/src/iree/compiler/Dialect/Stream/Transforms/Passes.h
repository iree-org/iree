// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_STREAM_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir::func {
class FuncDialect;
} // namespace mlir::func

namespace mlir::iree_compiler::IREE::Stream {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

// TODO(benvanik): find a way to share this with IREEVM.h w/o circular deps.
// Defines the output format of a dump pass.
enum class DumpOutputFormat {
  // Dumping disabled.
  None = 0,
  // Human-readable pretty printing.
  Pretty = 1,
  // Pretty printing with additional information that can result in large dumps.
  Verbose = 2,
  // Comma separated values for throwing into Sheets.
  CSV = 3,
  // JSON format for better structure and data exchange.
  JSON = 4,
};

struct TransformOptions : public PassPipelineOptions<TransformOptions> {
  // TODO(benvanik): options for async/sync overrides.

  Option<bool> optimizeBindings{
      *this,
      "optimize-bindings",
      llvm::cl::desc(
          "Enables binding fusion and dispatch site specialization."),
      llvm::cl::init(true),
  };

  Option<DumpOutputFormat> dumpStatisticsFormat{
      *this,
      "dump-statistics-format",
      llvm::cl::desc("Dumps statistics in the specified output format."),
      llvm::cl::init(DumpOutputFormat::None),
      llvm::cl::values(
          clEnumValN(IREE::Stream::DumpOutputFormat::Pretty, "pretty",
                     "Human-readable pretty printed output."),
          clEnumValN(IREE::Stream::DumpOutputFormat::Verbose, "verbose",
                     "Pretty printed output with additional IR."),
          clEnumValN(IREE::Stream::DumpOutputFormat::CSV, "csv",
                     "Comma separated values.")),
  };
  Option<std::string> dumpStatisticsFile{
      *this,
      "dump-statistics-file",
      llvm::cl::desc(
          "File path to write to; or `` for stderr or `-` for stdout."),
      llvm::cl::init(""),
  };
};

// Adds a set of passes to the given pass manager that run the required flow
// transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   Input legalization by one of:
//     - Directly passing supported flow plus core ops
//   buildStreamTransformPassPipeline
//   <run conversion from flow to sequencer/hal/vm/etc>
//
// This is equivalent to:
//   buildStreamTensorPassPipeline
//   buildStreamAsyncPassPipeline
//   buildStreamCmdPassPipeline
void buildStreamTransformPassPipeline(OpPassManager &passManager,
                                      const TransformOptions &transformOptions);

// Lowers from source dialects into stream.tensor.* IR.
void buildStreamTensorPassPipeline(OpPassManager &passManager,
                                   const TransformOptions &transformOptions);
// Lowers stream.tensor.* IR into stream.async.* IR.
void buildStreamAsyncPassPipeline(OpPassManager &passManager,
                                  const TransformOptions &transformOptions);
// Lowers stream.async.* IR into stream.cmd.* IR.
void buildStreamCmdPassPipeline(OpPassManager &passManager,
                                const TransformOptions &transformOptions);
// Optimizes stream commands (mostly optional).
void buildStreamOptimizationPassPipeline(
    OpPassManager &passManager, const TransformOptions &transformOptions);

void registerStreamTransformPassPipelines();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc" // IWYU pragma: keep

void registerStreamPasses();

} // namespace mlir::iree_compiler::IREE::Stream

#endif // IREE_COMPILER_DIALECT_STREAM_TRANSFORMS_PASSES_H_
