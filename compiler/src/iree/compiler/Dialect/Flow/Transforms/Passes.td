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
  let summary = "Clone producers into dispatch regions to be isolated above";
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

def ConvertRegionToWorkgroupsPass :
    Pass<"iree-flow-convert-region-to-workgroups", ""> {
  let summary = "Convert dispatch.region ops to dispatch.workgroups ops.";
  let dependentDialects = [
    "IREE::Flow::FlowDialect",
    "mlir::tensor::TensorDialect",
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
    "IREE::Flow::FlowDialect",
    "mlir::affine::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def DeduplicateExecutablesPass :
    Pass<"iree-flow-deduplicate-executables", "mlir::ModuleOp"> {
  let summary = "Deduplicates executables that are identical";
}

def FoldUnitExtentDimsPass :
    InterfacePass<"iree-flow-fold-unit-extent-dims", "mlir::FunctionOpInterface"> {
  let summary = "Fold unit extent dimension of operations";
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
  let summary = "Form Dispatch Region Ops from Linalg operations on tensors to form dispatch.regions";
  let options = [
    Option<"fuseMultiUse", "fuse-multi-use", "bool",
           /*default=*/"false", "Fuse multi-use ops">,
    Option<"generateWorkloadRegion", "genereate-workload-region", "bool",
           /*default=*/"true", "Generate workload regions of WorkgroupOps">,
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
    "IREE::Flow::FlowDialect",
    "mlir::affine::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def FormDispatchWorkgroupsPass :
    InterfacePass<"iree-flow-form-dispatch-workgroups", "mlir::FunctionOpInterface"> {
  let summary = "Form Dispatch Workgroup Ops from Dispatch Region Ops that contain Linalg on tensor ops by tiling and distribution.";
  let options = [
    Option<"generateWorkloadRegion", "generate-workload-region", "bool",
           /*default=*/"true", "Generate workload regions of WorkgroupOps">,
  ];
  let description = [{
    Pass to perform dispatch of dispatch.region ops that contain Linalg on tensor
    ops by tiling and distribution. A dispatch region is created for each tiled
    loop nest.
  }];
  let dependentDialects = [
    "IREE::Flow::FlowDialect",
    "mlir::affine::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def FormScalarDispatchesPass :
    InterfacePass<"iree-flow-form-scalar-dispatches", "mlir::FunctionOpInterface"> {
  let summary = "Form Dispatch Regions for scalar computations.";
  let dependentDialects = [
    "IREE::Flow::FlowDialect",
    "mlir::affine::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
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
  let summary = "Dispatch Linalg operations on tensors by using the transform dialect interpreter";
  let description = [{
    Pass to perform dispatch of Linalg on tensor ops by using the transform
    dialect. Dispatch regions are created as specified by the transform module
    that is parsed from `transformSpecPath`.

    TODO: Drop this pass in favor of the one upstream. The one upstream requires
    separate loading of the module and thus isn't suited for single-use
    transform scripts.
  }];
  let dependentDialects = [
    "IREE::Flow::FlowDialect",
    "mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect",
    "mlir::transform::TransformDialect",
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
  let summary = "Print visualization of dispatches";
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

def ExportBenchmarkFuncsPass :
    Pass<"iree-flow-export-benchmark-funcs-pass", "mlir::ModuleOp"> {
  let summary = "Exports benchmark functions";
  let description = [{
    Exports all functions and dispatch executables as `() -> ()` benchmark funcs.
  }];
  let dependentDialects = [
    "IREE::Flow::FlowDialect",
    "IREE::HAL::HALDialect",
    "IREE::Util::UtilDialect",
    "mlir::arith::ArithDialect",
  ];
}

def FusionOfTensorOpsPass :
    InterfacePass<"iree-flow-fusion-of-tensor-ops", "mlir::FunctionOpInterface"> {
  let summary = "Fuse Linalg operations on tensors";
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

def InitializeEmptyTensorsPass :
    Pass<"iree-flow-initialize-empty-tensors", ""> {
  let summary = "Initialize empty tensors";
  let options = [
    Option<"zeroFill", "zero-fill", "bool", /*default=*/"false",
           "Fills empty tensors with zeros.">,
  ];
  let description = [{
    Initialize all empty tensors after dispatch formation to zero or
    uninitialized allocations.
  }];
  let dependentDialects = [
    "IREE::Flow::FlowDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
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
    "IREE::Flow::FlowDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def InsertDebugTargetAtSymbolPass :
    Pass<"iree-flow-insert-debug-target-at-symbol", "mlir::ModuleOp"> {
  let summary = "Crops and/or traces the program at the specified symbol";
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
  let summary = "Crops and/or traces the program at the specified ordinal";
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

def InterchangeGenericOpsPass :
    Pass<"iree-flow-interchange-generic-ops", ""> {
  let summary = "Interchange generic op loops to have all the reduction loops to be inner loops.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

def InterchangeTransposeGenericOpsPass :
    Pass<"iree-flow-interchange-transpose-generic-ops", ""> {
  let summary = "Interchange transpose generic op loops to make the input indeximg map indentity.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

def OutlineDispatchExternsPass :
    Pass<"iree-flow-outline-dispatch-externs", "mlir::ModuleOp"> {
  let summary = "Outlines external dispatches into executables";
  let dependentDialects = [
    "IREE::Flow::FlowDialect",
    "IREE::HAL::HALDialect",
  ];
}

def OutlineDispatchRegionsPass :
    Pass<"iree-flow-outline-dispatch-regions", "mlir::ModuleOp"> {
  let summary = "Outlines dispatch regions into executables";
  let dependentDialects = [
    "IREE::Flow::FlowDialect",
    "mlir::func::FuncDialect",
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
  let summary = "Convert tensor.pad into linalg.fill + tensor.insert_slice";
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
