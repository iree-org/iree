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
  let options = [
    Option<"enableBubbleUpExpandShapesAcrossReductionOps", "enable-bubble-up-expand-shapes-across-reduction-ops", "bool",
           /*default=*/"false", "Enables propagation of 'expand_shape's through 'linalg.generic's with reductions">
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::linalg::LinalgDialect",
    "IREE::LinalgExt::IREELinalgExtDialect",
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

def FoldUnitExtentDimsForFuncPass :
    Pass<"iree-dispatch-creation-fold-unit-extent-dims-for-func", ""> {
  let summary = "Fold unit extent dimension of operations on a function.";
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
  let summary = "Fuses horizontal contraction ops";
  let description = [{
    For cases where multiple contractions
    - that dont have a direct dependence
    - that have the same LHS operand
    - all the N dimensions of the RHS operands used are the same
    Such contractions can be executed as a single contraction, i.e.

    A = matmul(lhs, rhs0);
    B = matmul(lhs, rhs1);
    C = matmul(lhs, rhs2);

    can be combined into
    result = matmul(lhs, concat_along_N(rhs0, rhs1, rhs2));
    A = slice0(result)
    B = slice1(result)
    C = slice2(result)

    Instead of doing an actual concat of the RHS operands,
    and extracting slices of the result, the pass generates a single
    operation with
    - the lhs operands
    - all the rhs operands
    - multiple results representing the individual matmuls
  }];
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
    "IREE::LinalgExt::IREELinalgExtDialect",
  ];
}

def SplitReductionPass :
    Pass<"iree-dispatch-creation-split-reduction-ops", ""> {
  let summary = "Split reduction dimension to increase parallelism.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "IREE::LinalgExt::IREELinalgExtDialect",
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
  let options = [
    Option<"aggressive", "aggressive", "bool",
    /*default=*/"false",
    "Include operations that are cloned only under aggressive fusion mode">,
  ];
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
  let dependentDialects = [
    "IREE::LinalgExt::IREELinalgExtDialect",
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
    "IREE::TensorExt::IREETensorExtDialect",
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
  let options = [
    Option<"hoistEncodingsForConstExpr", "hoist-encodings-for-constexpr", "bool", /*default=*/"true",
           "Enable the hoisting when the source is a ConstExpr.">,
  ];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "IREE::Flow::FlowDialect",
    "IREE::Encoding::IREEEncodingDialect",
  ];
}

def PropagateEncodingsPass :
    InterfacePass<"iree-dispatch-creation-propagate-encodings", "mlir::FunctionOpInterface"> {
  let summary = "Propagate encodings across other operations.";
  let dependentDialects = [
    "mlir::tensor::TensorDialect",
    "IREE::Encoding::IREEEncodingDialect",
  ];
}

def SetEncodingPass : InterfacePass<"iree-dispatch-creation-set-encoding",
                                    "mlir::FunctionOpInterface"> {
  let summary = "Introduces tensor encoding for flow dispatch regions.";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "IREE::Flow::FlowDialect",
    "IREE::Encoding::IREEEncodingDialect",
  ];
  let options = [
    Option<
        "encodingOption", "encoding-option",
        "mlir::iree_compiler::DispatchCreation::EncodingOptions",
        /*default=*/
        "mlir::iree_compiler::DispatchCreation::EncodingOptions::Generic",
        "Select the type of encoding options to add.",
        [{::llvm::cl::values(
            clEnumValN(
                mlir::iree_compiler::DispatchCreation::EncodingOptions::Padding,
                "padding", "Encode tensors that need to be padded."),
            clEnumValN(
                mlir::iree_compiler::DispatchCreation::EncodingOptions::MatmulK,
                "matmulk", "Only encodes reduction dimensions in the encoding."),
            clEnumValN(
                mlir::iree_compiler::DispatchCreation::EncodingOptions::Generic,
                "default", "Uses EncodingAttr which encodes as much information as possible."))}]>,
  ];
}

def ConvertEncodingToFlowPass :
    InterfacePass<"iree-dispatch-creation-convert-encoding-to-flow", "mlir::FunctionOpInterface"> {
  let summary = "Convert top-level Encoding ops to Flow ops.";
  let description = [{
    Pass to convert top-level Encoding ops to Flow ops, which only converts the
    Encoding ops outside flow.dispatch.region to Flow.
  }];
  let dependentDialects = [
    "IREE::Encoding::IREEEncodingDialect",
    "IREE::Flow::FlowDialect",
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
    "IREE::TensorExt::IREETensorExtDialect",
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
        "IREE::TensorExt::IREETensorExtDialect",
        "mlir::affine::AffineDialect",
        "mlir::arith::ArithDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::scf::SCFDialect",
        "IREE::Flow::FlowDialect",

      ];
}


#endif // IREE_COMPILER_DISPATCHCREATION_PASSES
