// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_STREAM_PASSES
#define IREE_DIALECT_STREAM_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Optimizations
//===----------------------------------------------------------------------===//

def OutlineConstants :
    Pass<"iree-stream-outline-constants", "mlir::ModuleOp"> {
  let summary = "Outlines tensor constants into util.globals at the module level.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createOutlineConstantsPass()
  }];
}

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

def ConvertToStream :
    Pass<"iree-stream-conversion", "mlir::ModuleOp"> {
  let summary = "Converts from flow/std/etc dialects into the stream dialect.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createConvertToStreamPass()
  }];
}

def EncodeHostTensors :
    Pass<"iree-stream-encode-host-tensors", ""> {
  let summary = "Encodes tensors into binary formats based on affinity and target support.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createEncodeHostTensorsPass()
  }];
}

def EncodeDeviceTensors :
    Pass<"iree-stream-encode-device-tensors", ""> {
  let summary = "Encodes tensors into binary formats based on affinity and target support.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createEncodeDeviceTensorsPass()
  }];
}

def MaterializeBuiltins :
    Pass<"iree-stream-materialize-builtins", "mlir::ModuleOp"> {
  let summary = "Materialize stream.builtin.* executables.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createMaterializeBuiltinsPass()
  }];
}

//===----------------------------------------------------------------------===//
// Placement/affinity management
//===----------------------------------------------------------------------===//

def MaterializeCopyOnWrite :
    Pass<"iree-stream-materialize-copy-on-write", ""> {
  let summary = "Materializes copy-on-write (🐄) behavior as explicit ops.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createMaterializeCopyOnWritePass()
  }];
}

def ElideAsyncCopies :
    Pass<"iree-stream-elide-async-copies", "mlir::ModuleOp"> {
  let summary = "Elides copies when they are not performing meaningful work.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createElideAsyncCopiesPass()
  }];
}

def EmplaceAllocations :
    Pass<"iree-stream-emplace-allocations", ""> {
  let summary = "Emplaces transient tensor allocations to remove copies.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createEmplaceAllocationsPass()
  }];
}

def RefineUsage :
    Pass<"iree-stream-refine-usage", "mlir::ModuleOp"> {
  let summary = "Refines resource usage bits and inserts transfers where appropriate.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createRefineUsagePass()
  }];
}

//===----------------------------------------------------------------------===//
// Stream formation and scheduling
//===----------------------------------------------------------------------===//

def ScheduleExecution :
    InterfacePass<"iree-stream-schedule-execution", "mlir::CallableOpInterface"> {
  let summary = "Identifies and groups asynchronous operations into executable regions within function-like regions.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createScheduleExecutionPass()
  }];
}

def ScheduleConcurrency :
    InterfacePass<"iree-stream-schedule-concurrency", "mlir::CallableOpInterface"> {
  let summary = "Identifies and groups asynchronous operations within executable regions that can run concurrently and groups them into streams.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createScheduleConcurrencyPass()
  }];
}

def PropagateTimepoints :
    Pass<"iree-stream-propagate-timepoints", "mlir::ModuleOp"> {
  let summary = "Materializes timepoints and sinks them to consumers throughout the whole program.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createPropagateTimepointsPass()
  }];
}

def ElideTimepoints :
    Pass<"iree-stream-elide-timepoints", "mlir::ModuleOp"> {
  let summary = "Elides timepoints that are known to be covered by dependent timepoints.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createElideTimepointsPass()
  }];
}

//===----------------------------------------------------------------------===//
// Allocation and command issuing
//===----------------------------------------------------------------------===//

def ScheduleAllocation :
    Pass<"iree-stream-schedule-allocation", "mlir::ModuleOp"> {
  let summary = "Allocates resources and converts to explicit stream commands.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createScheduleAllocationPass()
  }];
}

def PackConstants :
    InterfacePass<"iree-stream-pack-constants", "mlir::CallableOpInterface"> {
  let summary = "Packs and allocate backing storage for fused constant resources.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createPackConstantsPass()
  }];
}

def PackAllocations :
    InterfacePass<"iree-stream-pack-allocations", "mlir::CallableOpInterface"> {
  let summary = "Packs fused allocations based on lifetime.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createPackAllocationsPass()
  }];
}

def LayoutSlices :
    InterfacePass<"iree-stream-layout-slices", "mlir::CallableOpInterface"> {
  let summary = "Lays out packed slices and produces arithmetic required for all offsets.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createLayoutSlicesPass()
  }];
}

//===----------------------------------------------------------------------===//
// Memoization
//===----------------------------------------------------------------------===//

// TODO(benvanik): outline streams (ala dispatch regions).
// TODO(benvanik): deduplicate outlined streams.

//===----------------------------------------------------------------------===//
// Dispatch optimization
//===----------------------------------------------------------------------===//

def FoldUniformOperands :
    Pass<"iree-stream-fold-uniform-operands", "mlir::ModuleOp"> {
  let summary = "Folds redundant and uniformly constant dispatch operands.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createFoldUniformOperandsPass()
  }];
}

def FuseDispatchBindings :
    Pass<"iree-stream-fuse-dispatch-bindings", "mlir::ModuleOp"> {
  let summary = "Fuses bindings to the same underlying storage to reduce binding count.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createFuseDispatchBindingsPass()
  }];
  let options = [
    Option<"aliasMutableBindings", "alias-mutable-bindings",
           "bool", /*default=*/"false",
           "Fuses bindings that are mutable instead of leaving them split.">
  ];
}

def SpecializeDispatches :
    Pass<"iree-stream-specialize-dispatches", "mlir::ModuleOp"> {
  let summary = "Specializes executables by inlining/fusing operands based on dispatch sites.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createSpecializeDispatchesPass()
  }];
}

def AnnotateDispatchArguments :
    Pass<"iree-stream-annotate-dispatch-arguments", "mlir::ModuleOp"> {
  let summary = "Annotates dispatch arguments with potential values derived from dispatch sites.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createAnnotateDispatchArgumentsPass()
  }];
}

def PackDispatchOperands :
    Pass<"iree-stream-pack-dispatch-operands", "mlir::ModuleOp"> {
  let summary = "Packs stream dispatch operands into i32 push constants.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createPackDispatchOperandsPass()
  }];
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

def DumpStatistics :
    Pass<"iree-stream-dump-statistics", "mlir::ModuleOp"> {
  let summary = "Dumps stream dialect usage information to a file.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createDumpStatisticsPass()
  }];
  let options = [
    Option<"outputFormat", "output-format", "IREE::Stream::DumpOutputFormat",
           "IREE::Stream::DumpOutputFormat::Pretty",
           "Specifies the output format to produce.",
           [{::llvm::cl::values(
             clEnumValN(IREE::Stream::DumpOutputFormat::Pretty, "pretty", "Human-readable pretty printed output."),
             clEnumValN(IREE::Stream::DumpOutputFormat::Verbose, "verbose", "Pretty printed output with additional IR."),
             clEnumValN(IREE::Stream::DumpOutputFormat::CSV, "csv", "Comma separated values.")
           )}]>,
    Option<"outputFile", "output-file",
           "std::string", /*default=*/"std::string()",
           "File path to write to; or `` for stderr or `-` for stdout.">
  ];
}

def VerifyInput :
    Pass<"iree-stream-verify-input", "mlir::ModuleOp"> {
  let summary = "Verifies that input dialects are supported by the streams dialect.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createVerifyInputPass()
  }];
}

def VerifyLoweringToTensors :
    Pass<"iree-stream-verify-lowering-to-tensors", "mlir::ModuleOp"> {
  let summary = "Verifies that input dialects are converted to stream.tensor.* ops.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createVerifyLoweringToTensorsPass()
  }];
}

def VerifyLoweringToAsync :
    Pass<"iree-stream-verify-lowering-to-async", "mlir::ModuleOp"> {
  let summary = "Verifies that all stream.tensor.* ops and types are fully lowered to stream.async.* ops.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createVerifyLoweringToAsyncPass()
  }];
}

def VerifyLoweringToCmd :
    Pass<"iree-stream-verify-lowering-to-cmd", "mlir::ModuleOp"> {
  let summary = "Verifies that all stream.async.* ops and types are fully lowered to stream.cmd.* ops.";
  let constructor = [{
    mlir::iree_compiler::IREE::Stream::createVerifyLoweringToCmdPass()
  }];
}

#endif  // IREE_DIALECT_STREAM_PASSES
