// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DISPATCHCREATION_PASSES
#define IREE_COMPILER_DISPATCHCREATION_PASSES


include "mlir/Pass/PassBase.td"

// File organization:
// Groups passes that are related under one banner //===....===//. For example
// the dispatch region creation preprocessing passes and dispatch region
// formation passes are a couple of such groups. For any new pass add it to the
// relevant group and keep them alphabetical within a group.

//===---------------------------------------------------------------------===//
// Dispatch region creation preprocessing passes :
// Passes that transform the program before forming dispatches, like
// - Elementwise operation fusion
// - Reshape propagation passes
//===---------------------------------------------------------------------===//

def BubbleUpExpandShapesPass :
    Pass<"iree-dispatch-creation-bubble-up-expand-shapes"> {
  let summary = "Propagate expand_shapes up the program (and collapse_shapes down).";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
  ];
}

def CollapseReductionDimensionsPass :
    Pass<"iree-dispatch-creation-collapse-reduction-dimensions", ""> {
  let summary = "Collapse reduction dimensions when possible.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

def ElementwiseOpFusionPass :
    Pass<"iree-dispatch-creation-elementwise-op-fusion", ""> {
  let summary = "Fuse elementwise operations.";
  let options = [
    Option<"fuseMultiReduction", "fuse-multi-reduction", "bool",
           /*default=*/"true", "Fuse ops that have multiple reduction iterators">
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
  ];
}

def FoldUnitExtentDimsPass :
    Pass<"iree-dispatch-creation-fold-unit-extent-dims", "mlir::ModuleOp"> {
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

def FuseHorizontalContractionsPass:
    InterfacePass<"iree-dispatch-creation-fuse-horizontal-contractions", "mlir::FunctionOpInterface"> {
  let summary = "Fuses horizontal contraction ops without fusions";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
  ];
  let options = [
    Option<"fusionLimit", "fusion-limit", "int",
            /*default=*/"3", "Maximum number of contractions fused into one">
  ];
  let statistics = [
    Statistic<"numFusionGroups", "num-fusion-groups", "Number of fusion groups found">,
    Statistic<"numSize2FusionGroups", "num-size-2-groups", "Number of fusion groups of size 2">,
    Statistic<"numSize3FusionGroups", "num-size-3-groups", "Number of fusion groups of size 3">
  ];
}

def FuseMultiUseElementwiseProducerPass :
    InterfacePass<"iree-dispatch-creation-fuse-multi-use-elementwise-producer",
                   "mlir::FunctionOpInterface"> {
  let summary = "Fuse elementwise linalg operations on tensors when producers have multiple uses.";
  let options = [
    Option<"numIterations", "num-iterations", "unsigned",
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
    Pass<"iree-dispatch-creation-fusion-preprocessing", ""> {
  let summary = "Run useful preprocessing patterns that help with fusion.";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
  ];
}

def SinkReshapesPass :
    Pass<"iree-dispatch-creation-sink-reshapes", ""> {
  let summary = "Sink reshapes to allow for compute op -> consumer fusion.";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
  ];
}

def SplitReductionPass :
    Pass<"iree-dispatch-creation-split-reduction-ops", ""> {
  let summary = "Split reduction dimension to increase parallelism.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

def TensorPadToTensorInsertSlicePass :
    Pass<"iree-dispatch-creation-tensor-pad-to-tensor-insert-slice"> {
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

def TransposeGenericOpsPass :
    Pass<"iree-dispatch-creation-transpose-generic-ops", ""> {
  let summary = "Transpose generic op loops.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

def BubbleUpExtractSlicesPass : Pass<"iree-dispatch-creation-bubble-up-extract-slices"> {
  let summary = "Bubble up `extract_slice` ops through Linalg-like ops";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
  ];
}

//===---------------------------------------------------------------------===//
// Dispatch region creation passes.
//===---------------------------------------------------------------------===//

def CloneProducersIntoDispatchRegionsPass :
    InterfacePass<"iree-dispatch-creation-clone-producers-into-dispatch-regions", "mlir::FunctionOpInterface"> {
  let summary = "Clone producers into dispatch regions to be isolated above.";
  let description = [{
    Pass to clone into dispatch regions producers of values used in the dispatch
    regions but defined in the above. This prepares the dispatch regions for
    converting to dispatch workgroups with explicit captures.
  }];
}

def CollapseDimensionsPass :
    InterfacePass<"iree-dispatch-creation-collapse-dimensions", "mlir::FunctionOpInterface"> {
  let summary = "Collapse dimensions of Linalg Ops on tensor ops.";
  let options = [
    Option<"maxIterations", "max-iterations", "int",
    /*default=*/"10",
    "Maximum number of iterations to wait for collapse dimensions to converge"
    >,
  ];
  let description = [{
    Collapse dimensions of Linalg Ops on tensor ops inside dispatch.region ops
    and hoist the reshaping operations out of the dispatch.
  }];
}

def DispatchWithTransformDialectPass : Pass<"iree-dispatch-creation-dispatch-with-transform-dialect"> {
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

def FormDispatchRegionsPass :
    InterfacePass<"iree-dispatch-creation-form-dispatch-regions", "mlir::FunctionOpInterface"> {
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

def FormScalarDispatchesPass :
    InterfacePass<"iree-dispatch-creation-form-scalar-dispatches", "mlir::FunctionOpInterface"> {
  let summary = "Form Dispatch Regions for scalar computations.";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "IREE::Flow::FlowDialect",
  ];
}

def FuseEncodingOpsIntoDispatchRegionsPass :
    InterfacePass<"iree-dispatch-creation-fuse-encoding-ops-into-dispatch-regions-pass", "mlir::FunctionOpInterface"> {
  let summary = "Fuses set_encoding ops into producer dispatch regions, or forms new dispatches around them.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "IREE::Flow::FlowDialect",
    "IREE::Encoding::IREEEncodingDialect",
  ];
}

def HoistEncodingOpsPass :
    InterfacePass<"iree-dispatch-creation-hoist-encoding-ops", "mlir::FunctionOpInterface"> {
  let summary = "Hoists tensor encoding ops out of flow dispatch regions.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "IREE::Flow::FlowDialect",
    "IREE::Encoding::IREEEncodingDialect",
  ];
}


def SetEncodingPass :
    InterfacePass<"iree-dispatch-creation-set-encoding", "mlir::FunctionOpInterface"> {
  let summary = "Introduces tensor encoding for flow dispatch regions.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "IREE::Flow::FlowDialect",
    "IREE::Encoding::IREEEncodingDialect",
  ];
  let options = [
    Option<"padFactor", "pad-factor", "int64_t", /*default=*/"32",
           "provides padding size hints that will be attached to encodings.">,
  ];
}

//===---------------------------------------------------------------------===//
// Dispatch region to workgroups passes
//===---------------------------------------------------------------------===//

def ConvertDispatchRegionsToWorkgroupsPass :
    InterfacePass<"iree-dispatch-creation-convert-dispatch-regions-to-workgroups", "mlir::FunctionOpInterface"> {
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
  let statistics = [
    Statistic<"numDispatches", "num-dispatches", "Number of dispatches created">
  ];
}

def ConvertTensorToFlowPass :
    InterfacePass<"iree-dispatch-creation-convert-tensor-to-flow", "mlir::FunctionOpInterface"> {
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
    let statistics = [
      Statistic<"numSlowCopyDispatches", "num-slow-copy-dispatches",
                "Number of slow copy dispatches (for handling slices) created">
    ];
}

def MaterializeDefaultWorkgroupCountRegionPass:
    InterfacePass<"iree-dispatch-creation-materialize-default-workgroup-count-region",
                  "mlir::FunctionOpInterface"> {
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


#endif // IREE_COMPILER_DISPATCHCREATION_PASSES
