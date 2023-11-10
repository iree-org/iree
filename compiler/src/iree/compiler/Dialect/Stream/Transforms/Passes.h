// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_STREAM_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

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
// Conversion
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertToStreamPass();

//===----------------------------------------------------------------------===//
// Tensor lowering and resource management
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<>> createEncodeHostTensorsPass();
std::unique_ptr<OperationPass<>> createEncodeDeviceTensorsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createMaterializeBuiltinsPass();
std::unique_ptr<OperationPass<>> createMaterializeCopyOnWritePass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createElideAsyncCopiesPass();
std::unique_ptr<OperationPass<>> createEmplaceAllocationsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createRefineUsagePass();

//===----------------------------------------------------------------------===//
// Stream formation and scheduling
//===----------------------------------------------------------------------===//

std::unique_ptr<InterfacePass<CallableOpInterface>>
createScheduleExecutionPass();
std::unique_ptr<InterfacePass<CallableOpInterface>>
createScheduleConcurrencyPass();

std::unique_ptr<OperationPass<mlir::ModuleOp>> createPropagateTimepointsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createElideTimepointsPass();

//===----------------------------------------------------------------------===//
// Allocation and command issuing
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<mlir::ModuleOp>> createScheduleAllocationPass();

std::unique_ptr<InterfacePass<CallableOpInterface>> createPackConstantsPass();
std::unique_ptr<InterfacePass<CallableOpInterface>> createLayoutSlicesPass();

//===----------------------------------------------------------------------===//
// Memoization
//===----------------------------------------------------------------------===//

// TODO(benvanik): outline streams (ala dispatch regions).
// TODO(benvanik): deduplicate outlined streams.

//===----------------------------------------------------------------------===//
// Dispatch optimization
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<mlir::ModuleOp>> createFoldUniformOperandsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createFuseDispatchBindingsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createSpecializeDispatchesPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createAnnotateDispatchArgumentsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createPackDispatchOperandsPass();

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<mlir::ModuleOp>> createDumpStatisticsPass(
    DumpOutputFormat outputFormat = DumpOutputFormat::Pretty,
    std::string outputFile = "");

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createVerifyAsyncAccessRangesPass();

std::unique_ptr<OperationPass<mlir::ModuleOp>> createVerifyInputPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createVerifyLoweringToTensorsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createVerifyLoweringToAsyncPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createVerifyLoweringToCmdPass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerStreamPasses();

} // namespace Stream
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_DIALECT_STREAM_TRANSFORMS_PASSES_H_
