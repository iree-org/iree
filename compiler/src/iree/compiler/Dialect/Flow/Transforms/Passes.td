// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_FLOW_PASSES
#define IREE_DIALECT_FLOW_PASSES

include "mlir/Pass/PassBase.td"

// File organization:
// Groups passes that are related under one banner //===....===//. For any new
// pass add it to the relevant group and keep them alphabetical within a group.

//===---------------------------------------------------------------------===//
// General Flow passes
//===---------------------------------------------------------------------===//

def AnnotateDispatchesPass :
    Pass<"iree-flow-annotate-dispatches", "mlir::ModuleOp"> {
  let summary = "Annotates executable dispatches based on their contents.";
}

def CanonicalizePass :
    Pass<"iree-flow-canonicalize", ""> {
  let summary = "Flow specific canonicalization pass";
  let options = [
    Option<"cseConstants", "cse-constants", "bool",
            /*default=*/"true", "Do Not CSE constants on canonicalization">,
    Option<"testConvergence", "test-convergence", "bool",
           /*default=*/"false", "Fails if the patterns fail to converge">
  ];
}

def CaptureDynamicDimsPass :
    Pass<"iree-flow-capture-dynamic-dims", ""> {
  let summary = "Captures dynamic shape dimensions required by dispatch operands/results and control flow operations.";
  let dependentDialects = [
    "IREE::Flow::FlowDialect",
  ];
}

def CleanupTensorShapesPass :
    Pass<"iree-flow-cleanup-tensor-shapes", ""> {
  let summary = "Cleans up any remaining tensor shape metadata after lowering.";
}

def ConvertMeshToFlowPass :
    Pass<"iree-convert-mesh-to-flow", "mlir::ModuleOp"> {
  let summary = "Convert Mesh dialect operations to flow.";
  let description = [{
    Each mesh corresponds to a default !flow.channel with the same group name.
    ```
    mesh.mesh @mesh_1(shape = 2x3)
    ```
    ```
    %channel = flow.channel.default "mesh_1" : !flow.channel
    ```
    If there is onl one mesh in the program than the name is omitted and the
    ```
    %channel = flow.channel.default : !flow.channel
    ```

    Each (mesh, mesh_axes) pair partitions and orders the devices into disjoint
    groups, each corresponding to a !flow.channel to perform a collective
    operation.
    For example
    ```
    mesh.mesh @mesh(shape = 2x3x4x5)
    ...
    %1 = mesh.all_reduce on @mesh mesh_axes = [2, 0] : tensor<10x20xf32>
    ```
    For more information see
    [Mesh dialect](https://mlir.llvm.org/docs/Dialects/Mesh/#device-groups).

    The mesh partition and device ordering determines the values for the
    `color` and `key` in the corresponding `flow.channel.split` operation used
    to create the channel.
    For more information on the meaning of `color` and `key` see
    [MPI_Comm_split](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node188.htm#Node188)
    in the MPI standard.

    Each !flow.channel is wrapped in an IREE `util.global` and its construction
    is done only once with `util.initializer`.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::mesh::MeshDialect",
    "mlir::tensor::TensorDialect",
    "IREE::Flow::FlowDialect",
  ];
}

def ConvertToFlowPass :
    Pass<"iree-flow-convert-to-flow", ""> {
  let summary = "Convert operations to flow. Currently just a test pass.";
  let description = [{
    Convert operations to `flow` ops. This pass is currently only used for
    testing, since the conversion to Flow ops happens within dispatch region
    formation.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "IREE::Flow::FlowDialect",
  ];
}

def DeduplicateExecutablesPass :
    Pass<"iree-flow-deduplicate-executables", "mlir::ModuleOp"> {
  let summary = "Deduplicates executables that are identical.";
}

def DumpDispatchGraphPass : Pass<"iree-flow-dump-dispatch-graph-pass"> {
  let summary = "Dump visualization of dispatches within the program.";
  let options = [
    Option<
      "outputFile", "output-file",
      "std::string",
      /*default=*/"std::string()",
      "File path to write to; or `` for stderr or `-` for stdout."
    >,
    Option<"maxLabelLen", "max-label-len", "unsigned",
            /*default=*/"20", "Limit attribute/type length to number of chars">,
    Option<"printAttrs", "print-attrs", "bool",
           /*default=*/"true", "Print attributes of operations">,
    Option<"printControlFlowEdges", "print-control-flow-edges", "bool",
           /*default=*/"false", "Print control flow edges">,
    Option<"printDataFlowEdges", "print-data-flow-edges", "bool",
           /*default=*/"true", "Print data flow edges">,
    Option<"printResultTypes", "print-result-types", "bool",
            /*default=*/"true", "Print result types of operations">,
    Option<"emitDispatchBody", "emit-dispatch-body", "bool",
            /*default=*/"false", "Emit dispatch body in label">,
    Option<"emitInitializers", "emit-initializers", "bool",
            /*default=*/"false", "Emit initializers">
  ];
}

def ExportBenchmarkFuncsPass :
    Pass<"iree-flow-export-benchmark-funcs-pass", "mlir::ModuleOp"> {
  let summary = "Exports benchmark functions.";
  let description = [{
    Exports all functions and dispatch executables as `() -> ()` benchmark funcs.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "IREE::Flow::FlowDialect",
    "IREE::HAL::HALDialect",
    "IREE::Util::UtilDialect",
  ];
}

def InitializeEmptyTensorsPass :
    Pass<"iree-flow-initialize-empty-tensors", ""> {
  let summary = "Initialize empty tensors.";
  let options = [
    Option<"zeroFill", "zero-fill", "bool", /*default=*/"false",
           "Fills empty tensors with zeros.">,
  ];
  let description = [{
    Initialize all empty tensors after dispatch formation to zero or
    uninitialized allocations.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "IREE::Flow::FlowDialect",
  ];
}

def InjectDispatchTracingPass :
    InterfacePass<"iree-flow-inject-dispatch-tracing", "mlir::FunctionOpInterface"> {
  let summary = "Injects tracing markers for dispatch operation tensor inputs and outputs.";
}

def InjectTensorTracingPass :
    InterfacePass<"iree-flow-inject-tensor-tracing", "mlir::FunctionOpInterface"> {
  let summary = "Injects tensor tracing on ops annotated with `iree.tensor.trace`.";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "IREE::Flow::FlowDialect",
  ];
}

def InsertDebugTargetAtSymbolPass :
    Pass<"iree-flow-insert-debug-target-at-symbol", "mlir::ModuleOp"> {
  let summary = "Crops and/or traces the program at the specified symbol.";
  let options = [
    Option<"breakDebugTarget", "break-debug-target", "std::string",
           /*default=*/"",
           "Symbol at which to insert a break in the program.">,
    Option<"traceDebugTarget", "trace-debug-target", "std::string",
           /*default=*/"",
           "Symbol to insert iree.flow.trace ops around.">
  ];
  let dependentDialects = [
    "IREE::Flow::FlowDialect",
    "IREE::HAL::HALDialect",
    "IREE::Util::UtilDialect",
  ];
}

def InsertDebugTargetAtOrdinalPass :
    Pass<"iree-flow-insert-debug-target-at-ordinal", "mlir::ModuleOp"> {
  let summary = "Crops and/or traces the program at the specified ordinal.";
  let options = [
    Option<"breakDebugTarget", "break-debug-target", "std::string",
           /*default=*/"",
           "Ordinal at which to insert a break in the program.">,
    Option<"traceDebugTarget", "trace-debug-target", "std::string",
           /*default=*/"",
           "Ordinal to insert iree.flow.trace ops around.">
  ];
  let dependentDialects = [
    "IREE::Flow::FlowDialect",
    "IREE::HAL::HALDialect",
  ];
}

def OutlineConstantsPass :
    Pass<"iree-flow-outline-constants", "mlir::ModuleOp"> {
  let summary = "Outlines tensor constants into util.globals at the module level.";
  let description = [{
    Outlines tensor constants throughout the program into globals initialized
    with stream ops.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "IREE::Flow::FlowDialect",
    "IREE::Util::UtilDialect",
  ];
}

def OutlineDispatchExternsPass :
    Pass<"iree-flow-outline-dispatch-externs", "mlir::ModuleOp"> {
  let summary = "Outlines external dispatches into executables.";
  let dependentDialects = [
    "IREE::Flow::FlowDialect",
    "IREE::HAL::HALDialect",
  ];
}

def OutlineDispatchRegionsPass :
    Pass<"iree-flow-outline-dispatch-regions", "mlir::ModuleOp"> {
  let summary = "Outlines dispatch regions into executables.";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "IREE::Flow::FlowDialect",
  ];
}

def TopLevelSCFToCFGPass :
    InterfacePass<"iree-top-level-scf-to-cfg", "mlir::FunctionOpInterface"> {
  let summary = "Converts non-nested SCF constructs to CFG (not traversing into opaque operations).";
}

def VerifyInputLegalityPass: Pass<"iree-verify-input-legality", ""> {
  let summary = "Checks the legality of the IR at the start of IREE flow transformation pipeline.";
  let description = [{
    Verifies that the input to the Flow transformation pipeline is legal.
    This includes checking for operations from dialects that are expected
    to be legalized before this pass.
  }];
}

#endif  // IREE_DIALECT_FLOW_PASSES
