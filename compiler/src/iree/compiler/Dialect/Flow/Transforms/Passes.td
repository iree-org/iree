// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_FLOW_PASSES
#define IREE_DIALECT_FLOW_PASSES

include "mlir/Pass/PassBase.td"

def AnnotateDispatches :
    Pass<"iree-flow-annotate-dispatches", "mlir::ModuleOp"> {
  let summary = "Annotates executable dispatches based on their contents.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createAnnotateDispatchesPass()";
}

def CaptureDispatchDynamicDims :
    Pass<"iree-flow-capture-dispatch-dynamic-dims", ""> {
  let summary = "Captures dynamic shape dimensions required by dispatch operands/results.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createCaptureDispatchDynamicDimsPass()";
}

def CleanupTensorShapes :
    Pass<"iree-flow-cleanup-tensor-shapes", ""> {
  let summary = "Cleans up any remaining tensor shape metadata after lowering.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createCleanupTensorShapesPass()";
}

def CleanupNumericNarrowing :
    Pass<"iree-flow-cleanup-numeric-narrowing", ""> {
  let summary = "Cleans up any numeric narrowing ops inserted by iree-flow-infer-numeric-narrowing";
  let constructor = "mlir::iree_compiler::IREE::Flow::createCleanupNumericNarrowingPass()";
}

def CloneProducersIntoDispatchRegions :
    InterfacePass<"iree-flow-clone-producers-into-dispatch-regions", "mlir::FunctionOpInterface"> {
  let summary = "Clone producers into dispatch regions to be isolated above";
  let constructor = "mlir::iree_compiler::IREE::Flow::createCloneProducersIntoDispatchRegionsPass()";
}

def CollapseDims :
    Pass<"iree-flow-collapse-dims", ""> {
  let summary = "Collapse reduction dimensions when possible.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createCollapseDimsPass()";
}

def ConvertRegionToWorkgroups :
    Pass<"iree-flow-convert-region-to-workgroups", ""> {
  let summary = "Convert DispatchRegion ops to Workgroups ops.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createConvertRegionToWorkgroupsPass()";
}

def ConvertToFlow :
    Pass<"iree-flow-convert-to-flow", ""> {
  let summary = "Convert operations to flow. Currently just a test pass.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createConvertToFlowPass()";
}

def DeduplicateExecutables :
    Pass<"iree-flow-deduplicate-executables", "mlir::ModuleOp"> {
  let summary = "Deduplicates executables that are identical";
  let constructor = "mlir::iree_compiler::IREE::Flow::createDeduplicateExecutablesPass()";
}

def FoldUnitExtentDims :
    InterfacePass<"iree-flow-fold-unit-extent-dims", "mlir::FunctionOpInterface"> {
  let summary = "Fold unit extent dimension of operations";
  let constructor = "mlir::iree_compiler::IREE::Flow::createFoldUnitExtentDimsPass()";
}

def FormDispatchRegions :
    InterfacePass<"iree-flow-form-dispatch-regions", "mlir::FunctionOpInterface"> {
  let summary = "Form Dispatch Region Ops from Linalg operations on tensors to form dispatch.regions";
  let constructor = "mlir::iree_compiler::IREE::Flow::createFormDispatchRegionsPass()";
  let options = [
    Option<"fuseMultiUse", "fuse-multi-use", "bool",
           /*default=*/"false", "Fuse multi-use ops">,
    Option<"generateWorkloadRegion", "genereate-workload-region", "bool",
           /*default=*/"true", "Generate workload regions of WorkgroupOps">,
    Option<"fusePadWithConsumers", "fuse-pad-with-consumers", "bool",
           /*default=*/"false", "Enalbe fusing pad with consumer">,
    Option<"fusePadWithProducers", "fuse-pad-with-producers", "bool",
           /*default=*/"false", "Enable fusion of pad with producers">
  ];
}

def FormDispatchWorkgroups :
    InterfacePass<"iree-flow-form-dispatch-workgroups", "mlir::FunctionOpInterface"> {
  let summary = "Form Dispatch Workgroup Ops from Dispatch Region Ops that contain Linalg on tensor ops by tiling and distribution.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createFormDispatchWorkgroupsPass()";
  let options = [
    Option<"generateWorkloadRegion", "genereate-workload-region", "bool",
           /*default=*/"true", "Generate workload regions of WorkgroupOps">,
  ];
}

def FormScalarDispatches :
    InterfacePass<"iree-flow-form-scalar-dispatches", "mlir::FunctionOpInterface"> {
  let summary = "Form Dispatch Regions for scalar computations.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createFormScalarDispatchesPass()";
}

def FuseDequantizationMatmul:
    Pass<"iree-flow-fuse-dequantization-matmul", ""> {
  let summary = "Fuse dequantization and matmul linalg.generic ops";
  let constructor = "mlir::iree_compiler::IREE::Flow::createFuseDequantizationMatmulPass()";
}

def CollapseDimensions :
    InterfacePass<"iree-flow-collapse-dimensions", "mlir::FunctionOpInterface"> {
  let summary = "Collapse dimensions of Linalg Ops on tensor ops.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createCollapseDimensionsPass()";
}

def DispatchWithTransformDialect :
    InterfacePass<"iree-flow-dispatch-with-transform-dialect", "mlir::FunctionOpInterface"> {
  let summary = "Dispatch Linalg operations on tensors by using the transform dialect interpreter";
  let constructor = "mlir::iree_compiler::IREE::Flow::createDispatchWithTransformDialect()";

  let options = [
    Option<"transformFileName", "transform-file-name", "std::string",
            /*default=*/"\"\"",
            "Optional filename containing a transform dialect specification to "
            "apply. If left empty, the IR is assumed to contain one top-level "
            "transform dialect operation somewhere in the module.">,
    ListOption<"transformLibraryPaths",
           "transform-library-paths",
           "std::string",
           "If non-empty, the paths to files containing definitions of "
           "external symbols referenced in the transform script. "
           "These definitions will be used to replace declarations.">,
    Option<"debugPayloadRootTag", "debug-payload-root-tag", "std::string",
            /*default=*/"\"\"",
            "Select the operation with 'transform.target_tag' attribute having "
            "the given value as payload IR root. This allows user control on "
            "what operation to transform in debug mode, without requiring "
            "intimate knowledge of the IREE nested pass pipeline.\\n"
            "If empty (normal operation mode), select the pass anchor "
            "operation in the IREE pipeline, as the payload IR root.">,
    Option<"debugTransformRootTag", "debug-transform-root-tag", "std::string",
            /*default=*/"\"\"",
            "Select the operation with 'transform.target_tag' attribute having "
            "the given value as container IR for top-level transform ops. This "
            "allows user control on what transformation to apply in debug "
            "mode, without requiring intimate knowledge of the IREE nested "
            "pass pipeline.\\n"
            "If empty (normal operation mode), select the container of the "
            "top-level transform op.">
  ];
}

def DumpDispatchGraph : Pass<"iree-flow-dump-dispatch-graph-pass"> {
  let summary = "Print visualization of dispatches";
  let options = [
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
  let constructor = "mlir::iree_compiler::IREE::Flow::createDumpDispatchGraphPass()";
}

def ExpandTensorShapes :
    Pass<"iree-flow-expand-tensor-shapes", "mlir::ModuleOp"> {
  let summary = "Expands tensor shape dimensions into SSA values across the program.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createExpandTensorShapesPass()";
}

def ExportBenchmarkFuncs :
    Pass<"iree-flow-export-benchmark-funcs-pass", "mlir::ModuleOp"> {
  let summary = "Exports benchmark functions";
  let constructor = "mlir::iree_compiler::IREE::Flow::createExportBenchmarkFuncsPass()";
}

def FusionOfTensorOps :
    InterfacePass<"iree-flow-fusion-of-tensor-ops", "mlir::FunctionOpInterface"> {
  let summary = "Fuse operations on tensors";
  let constructor = "mlir::iree_compiler::IREE::Flow::createFusionOfTensorOpsPass()";
  let options = [
    Option<"fuseMultiUse", "fuse-multi-use", "bool",
           /*default=*/"false", "Fuse ops with multiuse">,
    Option<"multiUseFusionIteration", "multi-use-fusion-iteration", "unsigned",
           /*default=*/"2", "Number of iterations to fuse multiuse ops">
  ];
}

def GeneralizeLinalgNamedOps :
    InterfacePass<"iree-flow-generalize-linalg-named-ops", "mlir::FunctionOpInterface"> {
  let summary = "Convert some Linalg named ops into linalg.generics";
  let constructor = "mlir::iree_compiler::IREE::Flow::createGeneralizeLinalgNamedOpsPass()";
}

def InferNumericNarrowing :
    Pass<"iree-flow-infer-numeric-narrowing", ""> {
  let summary = "Infers and inserts util.numeric.optional_narrow ops at points that may be beneficial";
  let constructor = "mlir::iree_compiler::IREE::Flow::createInferNumericNarrowingPass()";
}

def InitializeEmptyTensors :
    Pass<"iree-flow-initialize-empty-tensors", ""> {
  let summary = "Initialize empty tensors";
  let options = [
    Option<"zeroFill", "zero-fill", "bool", /*default=*/"false",
           "Fills empty tensors with zeros.">,
  ];
  let constructor = "mlir::iree_compiler::IREE::Flow::createInitializeEmptyTensorsPass()";
}

def InjectDispatchTracing :
    InterfacePass<"iree-flow-inject-dispatch-tracing", "mlir::FunctionOpInterface"> {
  let summary = "Injects dispatch region tracing.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createInjectDispatchTracingPass()";
}

def InsertDebugTargetAtSymbol :
    Pass<"iree-flow-insert-debug-target-at-symbol", "mlir::ModuleOp"> {
  let summary = "Crops and/or traces the program at the specified symbol";
  let constructor = "mlir::iree_compiler::IREE::Flow::createInsertDebugTargetAtSymbolPass()";
  let options = [
    Option<"breakDebugTarget", "break-debug-target", "std::string",
           /*default=*/"",
           "Symbol at which to insert a break in the program.">,
    Option<"traceDebugTarget", "trace-debug-target", "std::string",
           /*default=*/"",
           "Symbol to insert iree.flow.trace ops around.">
  ];
}

def InsertDebugTargetAtOrdinal :
    Pass<"iree-flow-insert-debug-target-at-ordinal", "mlir::ModuleOp"> {
  let summary = "Crops and/or traces the program at the specified ordinal";
  let constructor = "mlir::iree_compiler::IREE::Flow::createInsertDebugTargetAtOrdinalPass()";
  let options = [
    Option<"breakDebugTarget", "break-debug-target", "std::string",
           /*default=*/"",
           "Ordinal at which to insert a break in the program.">,
    Option<"traceDebugTarget", "trace-debug-target", "std::string",
           /*default=*/"",
           "Ordinal to insert iree.flow.trace ops around.">
  ];
}

def InterchangeGenericOps :
    Pass<"iree-flow-interchange-generic-ops", ""> {
  let summary = "Interchange generic op loops to have all the reduction loops to be inner loops.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createInterchangeGenericOpsPass()";
}

def InterchangeTransposeGenericOps :
    Pass<"iree-flow-interchange-transpose-generic-ops", ""> {
  let summary = "Interchange transpose generic op loops to make the input indeximg map indentity.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createInterchangeTransposeGenericOpsPass()";
}

def OptimizeNumerics :
    Pass<"iree-flow-optimize-numerics", ""> {
  let summary = "Optimizes numerics given annotations added via iree-flow-infer-numeric-narrowing";
  let constructor = "mlir::iree_compiler::IREE::Flow::createOptimizeNumericsPass()";
}

def OutlineDispatchRegions :
    Pass<"iree-flow-outline-dispatch-regions", "mlir::ModuleOp"> {
  let summary = "Outlines dispatch regions into executables";
  let constructor = "mlir::iree_compiler::IREE::Flow::createOutlineDispatchRegionsPass()";
}

def RaiseSpecialOps :
    Pass<"iree-flow-raise-special-ops", ""> {
  let summary = "raise special ops like softmax to the high level linalg.ext representation";
  let constructor = "mlir::iree_compiler::IREE::Flow::createRaiseSpecialOps()";
}

def SplitReduction :
    Pass<"iree-flow-split-reduction-ops", ""> {
  let summary = "Split reduction dimension to increase parallelism.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createSplitReductionPass()";
}

def StripSignedness :
    InterfacePass<"iree-flow-strip-signedness", "mlir::FunctionOpInterface"> {
  let summary = "Legalizes ui tensors constants to uis";
  let constructor = "mlir::iree_compiler::IREE::Flow::createStripSignednessPass()";
}

def TensorPadToTensorInsertSlice :
    Pass<"iree-flow-tensor-pad-to-tensor-insert-slice", ""> {
  let summary = "Convert tensor.pad into linalg.fill + tensor.insert_slice";
  let constructor = "mlir::iree_compiler::IREE::Flow::createTensorPadToTensorInsertSlicePass()";
  let options = [
    Option<"optionSkipSingleLinalgOpUses", "skip-one-linalg-use-case", "bool",
           /*default=*/"false",
           "Skip the op that has only one use which is used"
           "by a Linalg op">,
  ];
}

def TopLevelSCFToCFG :
    InterfacePass<"iree-top-level-scf-to-cfg", "mlir::FunctionOpInterface"> {
  let summary = "Converts non-nested SCF constructs to CFG (not traversing into opaque operations).";
  let constructor = "mlir::iree_compiler::IREE::Flow::createTopLevelSCFToCFGPass()";
}

def VerifyInputLegality: Pass<"iree-verify-input-legality", ""> {
  let summary = "Checks the legality of the IR at the start of IREE flow transformation pipeline.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createVerifyInputLegalityPass()";
}

#endif  // IREE_DIALECT_FLOW_PASSES
