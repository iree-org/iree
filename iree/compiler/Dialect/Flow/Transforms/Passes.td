// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_FLOW_PASSES
#define IREE_DIALECT_FLOW_PASSES

include "mlir/Pass/PassBase.td"

def ConvertConv2D1x1ConvToMatmul :
    Pass<"iree-flow-convert-conv2d-1x1-to-matmul", "FuncOp"> {
  let summary = "Convert linalg convolution ops with 1x1 kernels into linalg matrix multiplication ops.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createConvertConv2D1x1ToMatmulPass()";
}

def ConvertConv2DToImg2Col :
    Pass<"iree-flow-convert-conv2d-to-img2col", "FuncOp"> {
  let summary = "Convert linalg convolution ops to matmul img2col based implementation";
  let constructor = "mlir::iree_compiler::IREE::Flow::createConvertConv2DToImg2ColPass()";
}

def ConvertToFlowTensorOps :
    Pass<"iree-flow-convert-to-flow-tensor-ops-pass", "FuncOp"> {
  let summary = "Convert operations to equivalent flow.tensor.* operations";
  let constructor = "mlir::iree_compiler::IREE::Flow::createConvertToFlowTensorOpsPass()";
}

def DeduplicateExecutables :
    Pass<"iree-flow-deduplicate-executables", "ModuleOp"> {
  let summary = "Deduplicates executables that are identical";
  let constructor = "mlir::iree_compiler::IREE::Flow::createDeduplicateExecutablesPass()";
}

def DispatchLinalgOnTensors :
    Pass<"iree-flow-dispatch-linalg-on-tensors-pass", "FuncOp"> {
  let summary = "Dispatch Linalg operations on tensors by using tile and distribute";
  let constructor = "mlir::iree_compiler::IREE::Flow::createDispatchLinalgOnTensorsPass()";
}

def ExpandVariableDynamicDims :
    Pass<"iree-flow-expand-variable-dynamic-dims", "ModuleOp"> {
  let summary = "Expands !shapex.ranked_shape dynamic dimensions stored in variables.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createExpandVariableDynamicDimsPass()";
}

def ExportBenchmarkFuncs :
    Pass<"iree-flow-export-benchmark-funcs-pass", "ModuleOp"> {
  let summary = "Exports benchmark functions";
  let constructor = "mlir::iree_compiler::IREE::Flow::createExportBenchmarkFuncsPass()";
}

def FormStreams :
    Pass<"iree-flow-form-streams", "FuncOp"> {
  let summary = "Identifies dispatches that can be grouped into streams within functions";
  let constructor = "mlir::iree_compiler::IREE::Flow::createFormStreamsPass()";
}

def FusionOfTensorOps :
    Pass<"iree-flow-fusion-of-tensor-ops", ""> {
  let summary = "Fuse operations on tensors";
  let constructor = "mlir::iree_compiler::IREE::Flow::createFusionOfTensorOpsPass()";
}

def HoistUnstreamableOps :
    Pass<"iree-flow-hoist-unstreamable-ops", "FuncOp"> {
  let summary = "Hoist ops that cannot be captured in streams to the top of their block.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createHoistUnstreamableOpsPass()";
}

def InjectDispatchTracing :
    Pass<"iree-flow-inject-dispatch-tracing", "FuncOp"> {
  let summary = "Injects dispatch region tracing";
  let constructor = "mlir::iree_compiler::IREE::Flow::createInjectDispatchTracingPass()";
}

def InterchangeGenericOps :
    Pass<"iree-flow-interchange-generic-ops", "FuncOp"> {
  let summary = "Interchange generic op loops to have all the reduction loops to be inner loops.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createInterchangeGenericOpsPass()";
}

def OutlineDispatchRegions :
    Pass<"iree-flow-outline-dispatch-regions2", "ModuleOp"> {
  let summary = "Outlines dispatch regions into executables";
  let constructor = "mlir::iree_compiler::IREE::Flow::createOutlineDispatchRegionsPass()";
}

def OutlineLargeConstants :
    Pass<"iree-flow-outline-large-constants", "ModuleOp"> {
  let summary = "Outlines large tensor constants into flow.variables at the module level.";
  // TODO(#5493): add a flag for this.
  let constructor = "mlir::iree_compiler::IREE::Flow::createOutlineLargeConstantsPass(25)";
}

def PadLinalgOps :
    Pass<"iree-flow-pad-linalg-ops", "FuncOp"> {
  let summary = "Pad linalg ops to the next integer multiple of paddingSize.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createPadLinalgOpsToIntegerMultiplePass()";
}

def PadTensorToSubTensorInsert :
    Pass<"iree-flow-pad-tensor-to-subtensor-insert", ""> {
  let summary = "Convert linalg.pad_tensor into linalg.fill + subtensor_insert";
  let constructor = "mlir::iree_compiler::IREE::Flow::createPadTensorToSubTensorInsertPass()";
}

def PromoteI1ToI8 :
    Pass<"iree-flow-promote-i1-to-i8", "FuncOp"> {
  let summary = "Legalizes i1 tensor constants to i8s";
  let constructor = "mlir::iree_compiler::IREE::Flow::createPromoteI1ToI8Pass()";
}

def PromoteTensorLoads :
    Pass<"iree-flow-promote-tensor-loads", "FuncOp"> {
  let summary = "Converts standard ops which match to flow.tensor.load (typically causing a read-back)";
  let constructor = "mlir::iree_compiler::IREE::Flow::createPromoteTensorLoadsPass()";
}

def StripAndSplatConstantVariables :
    Pass<"iree-flow-strip-and-splat-constant-variables", "ModuleOp"> {
  let summary = "Strips constant flow.variables and replaces them with splats.";
  let constructor = "mlir::iree_compiler::IREE::Flow::createStripAndSplatConstantVariablesPass()";
}

#endif  // IREE_DIALECT_FLOW_PASSES
