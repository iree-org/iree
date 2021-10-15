// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_FLOW_PASSES
#define IREE_DIALECT_FLOW_PASSES

include "mlir/Pass/PassBase.td"

def ConvertConv2D1x1ConvToMatmul :
    Pass<"iree-flow-convert-conv2d-1x1-to-matmul", "mlir::FuncOp"> {
  let summary = "Convert linalg convolution ops with 1x1 kernels into linalg matrix multiplication ops.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createConvertConv2D1x1ToMatmulPass()";
}

def ConvertConv2DToImg2Col :
    Pass<"iree-flow-convert-conv2d-to-img2col", "mlir::FuncOp"> {
  let summary = "Convert linalg convolution ops to matmul img2col based implementation";
  let constructor = "mlir::iree_compiler::IREE::Flow::createConvertConv2DToImg2ColPass()";
}

def ConvertToFlowBeforeDispatchFormation :
    Pass<"iree-flow-convert-to-flow-before-dispatch-formation", "mlir::FuncOp"> {
  let summary = "Convert operations to flow before dispatch formation";
  let constructor = "mlir::iree_compiler::IREE::Flow::createConvertToFlowBeforeDispatchFormation()";
}

def ConvertToFlowAfterDispatchFormation :
    Pass<"iree-flow-convert-to-flow-after-dispatch-formation", "mlir::FuncOp"> {
  let summary = "Convert operations to flow after dispatch formation";
  let constructor = "mlir::iree_compiler::IREE::Flow::createConvertToFlowAfterDispatchFormation()";
}

def DeduplicateExecutables :
    Pass<"iree-flow-deduplicate-executables", "mlir::ModuleOp"> {
  let summary = "Deduplicates executables that are identical";
  let constructor = "mlir::iree_compiler::IREE::Flow::createDeduplicateExecutablesPass()";
}

def DispatchLinalgOnTensors :
    Pass<"iree-flow-dispatch-linalg-on-tensors-pass", "mlir::FuncOp"> {
  let summary = "Dispatch Linalg operations on tensors by using tile and distribute";
  let constructor = "mlir::iree_compiler::IREE::Flow::createDispatchLinalgOnTensorsPass()";
}

def ExpandGlobalDynamicDims :
    Pass<"iree-flow-expand-global-dynamic-dims", "mlir::ModuleOp"> {
  let summary = "Expands !shapex.ranked_shape dynamic dimensions stored in variables.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createExpandGlobalDynamicDimsPass()";
}

def ExportBenchmarkFuncs :
    Pass<"iree-flow-export-benchmark-funcs-pass", "mlir::ModuleOp"> {
  let summary = "Exports benchmark functions";
  let constructor = "mlir::iree_compiler::IREE::Flow::createExportBenchmarkFuncsPass()";
}

def FormStreams :
    Pass<"iree-flow-form-streams", "mlir::FuncOp"> {
  let summary = "Identifies dispatches that can be grouped into streams within functions.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createFormStreamsPass()";
}

def FusionOfTensorOps :
    Pass<"iree-flow-fusion-of-tensor-ops", ""> {
  let summary = "Fuse operations on tensors";
  let constructor = "mlir::iree_compiler::IREE::Flow::createFusionOfTensorOpsPass()";
}

def HoistUnstreamableOps :
    Pass<"iree-flow-hoist-unstreamable-ops", "mlir::FuncOp"> {
  let summary = "Hoist ops that cannot be captured in streams to the top of their block.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createHoistUnstreamableOpsPass()";
}

def InjectDispatchTracing :
    Pass<"iree-flow-inject-dispatch-tracing", "mlir::FuncOp"> {
  let summary = "Injects dispatch region tracing.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createInjectDispatchTracingPass()";
}

def InsertConstantClones :
    Pass<"iree-flow-insert-constant-clones", "mlir::FuncOp"> {
  let summary = "Inserts clones of constant values where they may be required";
  let constructor = "mlir::iree_compiler::IREE::Flow::createInsertConstantClonesPass()";
}

def InterchangeGenericOps :
    Pass<"iree-flow-interchange-generic-ops", "mlir::FuncOp"> {
  let summary = "Interchange generic op loops to have all the reduction loops to be inner loops.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createInterchangeGenericOpsPass()";
}

def OutlineDispatchRegions :
    Pass<"iree-flow-outline-dispatch-regions2", "mlir::ModuleOp"> {
  let summary = "Outlines dispatch regions into executables";
  let constructor = "mlir::iree_compiler::IREE::Flow::createOutlineDispatchRegionsPass()";
}

def OutlineLargeConstants :
    Pass<"iree-flow-outline-large-constants", "mlir::ModuleOp"> {
  let summary = "Outlines large tensor constants into util.globals at the module level.";
  // TODO(#5493): add a flag for this.
  let constructor = "mlir::iree_compiler::IREE::Flow::createOutlineLargeConstantsPass(25)";
}

def PadLinalgOps :
    Pass<"iree-flow-pad-linalg-ops", "mlir::FuncOp"> {
  let summary = "Pad linalg ops to the next integer multiple of paddingSize.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createPadLinalgOpsToIntegerMultiplePass()";
}

def ConvertLinalgMatmulToMmt4D :
    Pass<"iree-flow-convert-linalg-matmul-to-mmt4d", "FuncOp"> {
  let summary = "Convert linalg.matmul to linalg.mmt4d";
  let constructor = "mlir::iree_compiler::IREE::Flow::createConvertLinalgMatmulToMmt4DPass()";
  let options = [
    Option<"M0", "M0", "int", /*default=*/"mlir::ShapedType::kDynamicSize",
           "Specifies an explicit M-axis tile size, overriding the default heuristic.">,
    Option<"K0", "K0", "int", /*default=*/"mlir::ShapedType::kDynamicSize",
           "Specifies an explicit K-axis tile size, overriding the default heuristic.">,
    Option<"N0", "N0", "int", /*default=*/"mlir::ShapedType::kDynamicSize",
           "Specifies an explicit N-axis tile size, overriding the default heuristic.">,
  ];
}

def PadTensorToSubTensorInsert :
    Pass<"iree-flow-pad-tensor-to-subtensor-insert", ""> {
  let summary = "Convert linalg.pad_tensor into linalg.fill + subtensor_insert";
  let constructor = "mlir::iree_compiler::IREE::Flow::createPadTensorToSubTensorInsertPass()";
}

def PromoteI1ToI8 :
    Pass<"iree-flow-promote-i1-to-i8", "mlir::FuncOp"> {
  let summary = "Legalizes i1 tensor constants to i8s";
  let constructor = "mlir::iree_compiler::IREE::Flow::createPromoteI1ToI8Pass()";
}

def StripSignedness :
    Pass<"iree-flow-strip-signedness", "mlir::FuncOp"> {
  let summary = "Legalizes ui tensors constants to uis";
  let constructor = "mlir::iree_compiler::IREE::Flow::createStripSignednessPass()";
}

def StripAndSplatConstantVariables :
    Pass<"iree-flow-strip-and-splat-constant-variables", "mlir::ModuleOp"> {
  let summary = "Strips constant util.globals and replaces them with splats.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createStripAndSplatConstantVariablesPass()";
}

def VerifyInputLegality: Pass<"iree-verify-input-legality", "mlir::FuncOp"> {
  let summary = "Checks the legality of the IR at the start of IREE flow transformation pipeline.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createVerifyInputLegalityPass()";
}

#endif  // IREE_DIALECT_FLOW_PASSES
