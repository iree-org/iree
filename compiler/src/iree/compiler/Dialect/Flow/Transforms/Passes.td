// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_FLOW_PASSES
#define IREE_DIALECT_FLOW_PASSES

include "mlir/Pass/PassBase.td"

def CaptureDispatchDynamicDims : Pass<"iree-flow-capture-dispatch-dynamic-dims", ""> {
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

def ConvertConv2D1x1ConvToMatmul :
    Pass<"iree-flow-convert-conv2d-1x1-to-matmul", ""> {
  let summary = "Convert linalg convolution ops with 1x1 kernels into linalg matrix multiplication ops.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createConvertConv2D1x1ToMatmulPass()";
}

def ConvertConv2DToImg2Col :
    Pass<"iree-flow-convert-conv2d-to-img2col", ""> {
  let summary = "Convert linalg convolution ops to matmul img2col based implementation";
  let constructor = "mlir::iree_compiler::IREE::Flow::createConvertConv2DToImg2ColPass()";
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

def DetachElementwiseFromNamedOps :
    Pass<"iree-flow-detach-elementwise-from-named-ops", ""> {
  let summary = "Detaches elementwise ops from named Linalg ops";
  let constructor = "mlir::iree_compiler::IREE::Flow::createDetachElementwiseFromNamedOpsPass()";
}

def DispatchLinalgOnTensors :
    InterfacePass<"iree-flow-dispatch-linalg-on-tensors-pass", "mlir::FunctionOpInterface"> {
  let summary = "Dispatch Linalg operations on tensors by using tile and distribute";
  let constructor = "mlir::iree_compiler::IREE::Flow::createDispatchLinalgOnTensorsPass()";
}

def DispatchWithTransformDialect :
    InterfacePass<"iree-flow-dispatch-with-transform-dialect", "mlir::FunctionOpInterface"> {
  let summary = "Dispatch Linalg operations on tensors by using the transform dialect interpreter";
  let constructor = "mlir::iree_compiler::IREE::Flow::createDispatchWithTransformDialect()";

  let options = [
    Option<"transformFileName", "transform-file-name", "std::string", /*default=*/"",
           "File name containing a transform dialect specification to apply.">
  ];
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
    Pass<"iree-flow-fusion-of-tensor-ops", ""> {
  let summary = "Fuse operations on tensors";
  let constructor = "mlir::iree_compiler::IREE::Flow::createFusionOfTensorOpsPass()";
}

def InferNumericNarrowing :
    Pass<"iree-flow-infer-numeric-narrowing", ""> {
  let summary = "Infers and inserts util.numeric.optional_narrow ops at points that may be beneficial";
  let constructor = "mlir::iree_compiler::IREE::Flow::createInferNumericNarrowingPass()";
}

def InitializeEmptyTensors : Pass<"iree-flow-initialize-empty-tensors", ""> {
  let summary = "Initialize empty tensors";
  let constructor = "mlir::iree_compiler::IREE::Flow::createInitializeEmptyTensorsPass()";
}

def InjectDispatchTracing :
    InterfacePass<"iree-flow-inject-dispatch-tracing", "mlir::FunctionOpInterface"> {
  let summary = "Injects dispatch region tracing.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createInjectDispatchTracingPass()";
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

def PadLinalgOps :
    Pass<"iree-flow-pad-linalg-ops", ""> {
  let summary = "Pad linalg ops to the next integer multiple of paddingSize.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createPadLinalgOpsToIntegerMultiplePass()";
}

def ConvertLinalgMatmulToMmt4D :
    Pass<"iree-flow-convert-linalg-matmul-to-mmt4d", ""> {
  let summary = "Convert linalg.matmul to linalg.mmt4d";
  let constructor = "mlir::iree_compiler::IREE::Flow::createConvertLinalgMatmulToMmt4DPass()";
  let options = [
    Option<"arch", "arch", "std::string",
           /*default=*/"",
           "Target architecture, e.g. aarch64">,
    Option<"features", "features", "std::string",
           /*default=*/"",
           "Additional CPU feature flags, e.g. +dotprod">,
    Option<"enableGenericSlow", "enable_generic_slow", "bool",
           /*default=*/"false",
           "For tests only. Use mmt4d even for cases that are not expected to compile to efficient code by using some arbitrary generic tile shape.">,
  ];
}

def PadTensorToSubTensorInsert :
    Pass<"iree-flow-pad-tensor-to-subtensor-insert", ""> {
  let summary = "Convert linalg.pad_tensor into linalg.fill + subtensor_insert";
  let constructor = "mlir::iree_compiler::IREE::Flow::createPadTensorToSubTensorInsertPass()";
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

def StripAndSplatConstantVariables :
    Pass<"iree-flow-strip-and-splat-constant-variables", "mlir::ModuleOp"> {
  let summary = "Strips constant util.globals and replaces them with splats.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createStripAndSplatConstantVariablesPass()";
}

def VerifyInputLegality: Pass<"iree-verify-input-legality", ""> {
  let summary = "Checks the legality of the IR at the start of IREE flow transformation pipeline.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createVerifyInputLegalityPass()";
}

#endif  // IREE_DIALECT_FLOW_PASSES
