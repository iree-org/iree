// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_FLOW_PASSES
#define IREE_DIALECT_FLOW_PASSES

include "mlir/Pass/PassBase.td"

def AnnotateDispatchesPass :
    Pass<"iree-flow-annotate-dispatches", "mlir::ModuleOp"> {
  let summary = "Annotates executable dispatches based on their contents.";
}

def BubbleUpExpandShapesPass :
    Pass<"iree-flow-bubble-up-expand-shapes"> {
  let summary = "Propagate expand_shapes up the program (and collapse_shapes down).";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
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

def CloneProducersIntoDispatchRegionsPass :
    InterfacePass<"iree-flow-clone-producers-into-dispatch-regions", "mlir::FunctionOpInterface"> {
  let summary = "Clone producers into dispatch regions to be isolated above.";
  let description = [{
    Pass to clone into dispatch regions producers of values used in the dispatch
    regions but defined in the above. This prepares the dispatch regions for
    converting to dispatch workgroups with explicit captures.
  }];
}

def CollapseReductionDimensionsPass :
    Pass<"iree-flow-collapse-reduction-dimensions", ""> {
  let summary = "Collapse reduction dimensions when possible.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
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

def ConvertRegionToWorkgroupsPass :
    Pass<"iree-flow-convert-region-to-workgroups", ""> {
  let summary = "Convert dispatch.region ops to dispatch.workgroups ops.";
  let dependentDialects = [
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

def FoldUnitExtentDimsPass :
    InterfacePass<"iree-flow-fold-unit-extent-dims", "mlir::FunctionOpInterface"> {
  let summary = "Fold unit extent dimension of operations.";
  let description = [{
    Imports upstream patterns to fold unit extent dims but with IREE control.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def FormDispatchRegionsPass :
    InterfacePass<"iree-flow-form-dispatch-regions", "mlir::FunctionOpInterface"> {
  let summary = "Form Dispatch Region Ops from Linalg operations on tensors to form dispatch.regions.";
  let options = [
    Option<"aggressiveFusion", "aggressive-fusion", "bool",
           /*default=*/"false", "Aggressive mode enabling fusions not ready for all backends">,
    Option<"fusePadWithConsumers", "fuse-pad-with-consumers", "bool",
           /*default=*/"false", "Enable fusing pad with consumer">,
    Option<"fusePadWithProducers", "fuse-pad-with-producers", "bool",
           /*default=*/"false", "Enable fusion of pad with producers">
  ];
  let description = [{
    Pass to form dispatch.region ops from Linalg on tensor ops. A dispatch region
    is created for each tiled loop nest. This pass only moves the root compute op
    into the dispatch region, allowing producers to be outside.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "IREE::Flow::FlowDialect",
    "IREE::LinalgExt::IREELinalgExtDialect",
  ];
}

def ConvertDispatchRegionsToWorkgroupsPass :
    InterfacePass<"iree-flow-convert-dispatch-regions-to-workgroups", "mlir::FunctionOpInterface"> {
  let summary = "Convert dispatch regions to dispatch workgroups.";
  let description = [{
    Pass to convert dispatch regions to dispatch workgroups. This pass is
    intended to be used after dispatch regions have been formed.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "IREE::Flow::FlowDialect",
  ];
}

def ConvertTensorToFlowPass :
    InterfacePass<"iree-flow-convert-tensor-to-flow", "mlir::FunctionOpInterface"> {
    let summary = "Convert tensor operations to flow";
    let description = [{
      Pass to convert tensor operations to flow.tensor.* operations.
    }];
    let dependentDialects = [
      "mlir::affine::AffineDialect",
      "mlir::arith::ArithDialect",
      "mlir::linalg::LinalgDialect",
      "mlir::tensor::TensorDialect",
      "IREE::Flow::FlowDialect",
    ];
}

def MaterializeDefaultWorkgroupCountRegionPass:
    InterfacePass<"iree-flow-materialize-default-workgroup-count-region", "mlir::FunctionOpInterface"> {
      let summary = "Canonicalize dispatch workgroups ops.";
      let description = [{
        Apply dispatch workgroups canonicalization patterns.
      }];
      let dependentDialects = [
        "mlir::affine::AffineDialect",
        "mlir::arith::ArithDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::scf::SCFDialect",
        "IREE::Flow::FlowDialect",
      ];
}

def FormScalarDispatchesPass :
    InterfacePass<"iree-flow-form-scalar-dispatches", "mlir::FunctionOpInterface"> {
  let summary = "Form Dispatch Regions for scalar computations.";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "IREE::Flow::FlowDialect",
  ];
}

def CollapseDimensionsPass :
    InterfacePass<"iree-flow-collapse-dimensions", "mlir::FunctionOpInterface"> {
  let summary = "Collapse dimensions of Linalg Ops on tensor ops.";
  let description = [{
    Collapse dimensions of Linalg Ops on tensor ops inside dispatch.region ops
    and hoist the reshaping operations out of the dispatch.
  }];
}

def DispatchWithTransformDialectPass : Pass<"iree-flow-dispatch-with-transform-dialect"> {
  let summary = "Dispatch Linalg operations on tensors by using the transform dialect interpreter.";
  let description = [{
    Pass to perform dispatch of Linalg on tensor ops by using the transform
    dialect. Dispatch regions are created as specified by the transform module
    that is parsed from `transformSpecPath`.

    TODO: Drop this pass in favor of the one upstream. The one upstream requires
    separate loading of the module and thus isn't suited for single-use
    transform scripts.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::transform::TransformDialect",
    "IREE::Flow::FlowDialect",
    "IREE::LinalgExt::IREELinalgExtDialect",
  ];
  let options = [
    Option<"disableExpensiveChecks", "disable-expensive-checks", "bool",
           "false",
           "Disable expensive checks in the interpreter for a faster run.">,
    Option<"transformSpecPath", "transform-spec-path", "std::string",
           /*default=*/"", "File path to the transform spec to use.">,
  ];
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
            /*default=*/"true", "Print result types of operations">
  ];
}

def ElementwiseOpFusionPass :
    Pass<"iree-flow-elementwise-op-fusion", ""> {
  let summary = "Fuse elementwise operations.";
  let options = [
    Option<"fuseMultiReduction", "fuse-multi-reduction", "bool",
           /*default=*/"true", "Fuse ops that have multiple reduction iterators">
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
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

def FusionOfTensorOpsPass :
    InterfacePass<"iree-flow-fusion-of-tensor-ops", "mlir::FunctionOpInterface"> {
  let summary = "Fuse Linalg operations on tensors.";
  let options = [
    Option<"fuseMultiUse", "fuse-multi-use", "bool",
           /*default=*/"false", "Fuse ops with multiuse">,
    Option<"fuseMultiReduction", "fuse-multi-reduction", "bool",
           /*default=*/"true", "Fuse ops that have multiple reduction iterators">,
    Option<"multiUseFusionIteration", "multi-use-fusion-iteration", "unsigned",
           /*default=*/"2", "Number of iterations to fuse multiuse ops">
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::math::MathDialect",
  ];
}

def FusionPreprocessingPass :
    Pass<"iree-flow-fusion-preprocessing", ""> {
  let summary = "Run useful preprocessing patterns that help with fusion.";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
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

def InterchangeTransposeGenericOpsPass :
    Pass<"iree-flow-interchange-transpose-generic-ops", ""> {
  let summary = "Interchange transpose generic op loops to make the input indeximg map indentity.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
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

def SetEncodingPass :
    InterfacePass<"iree-flow-set-encoding", "mlir::FunctionOpInterface"> {
  let summary = "Introduces tensor encoding for flow dispatch regions.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "IREE::Flow::FlowDialect",
    "IREE::Encoding::IREEEncodingDialect",
  ];
}

def SinkReshapesPass :
    Pass<"iree-flow-sink-reshapes", ""> {
  let summary = "Sink reshapes to allow for compute op -> consumer fusion.";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
  ];
}

def SplitReductionPass :
    Pass<"iree-flow-split-reduction-ops", ""> {
  let summary = "Split reduction dimension to increase parallelism.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

def TensorPadToTensorInsertSlicePass :
    Pass<"iree-flow-tensor-pad-to-tensor-insert-slice", ""> {
  let summary = "Convert tensor.pad into linalg.fill + tensor.insert_slice.";
  let options = [
    Option<"skipSingleLinalgOpUses", "skip-one-linalg-use-case", "bool",
           /*default=*/"false",
           "Skip the op that has only one use which is used"
           "by a Linalg op">,
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::math::MathDialect",
    "mlir::memref::MemRefDialect",
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
