// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMEXTENSIONS_IREEGPUEXTENSIONS
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMEXTENSIONS_IREEGPUEXTENSIONS

include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def ApplyDropMultiMmaOpUnitDims : Op<Transform_Dialect,
    "apply_patterns.iree.drop_multi_mma_unit_dims",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Populate patterns to drop the unit dims from multi_mma ops with
    only unit iteration bounds.
  }];

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let assemblyFormat = "attr-dict";
}

def ApplyLowerMultiMmaOp : Op<Transform_Dialect,
    "apply_patterns.iree.lower_multi_mma",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Populate patterns to drop the unit dims from multi_mma ops with
    only unit iteration bounds.
  }];

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let assemblyFormat = "attr-dict";
}

def ApplyLowerVectorBarrierOp : Op<Transform_Dialect,
    "apply_patterns.iree.lower_vector_barrier",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Populate patterns to convert vector barriers into gpu.barrier ops.
  }];

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let assemblyFormat = "attr-dict";
}

def ApplyVectorizeMultiMmaOp : Op<Transform_Dialect,
    "apply_patterns.iree.vectorize_multi_mma",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Populate patterns to vectorize static iree_gpu.multi_mma ops.
  }];

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let assemblyFormat = "attr-dict";
}

def ApplyUnrollMultiMmaOp : Op<Transform_Dialect,
    "apply_patterns.iree.unroll_multi_mma",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Populate patterns to unroll iree_gpu.multi_mma ops to a single intrinsic.
  }];

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let assemblyFormat = "attr-dict";
}

def ConvertToMultiMmaOp : Op<Transform_Dialect, "iree.convert_to_multi_mma",
    [TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Converts the target linalg op to a multi_mma operation based on the
    given intrinsic kind.

    #### Return modes
    Emits a definite failure if the target is not a contraction like linalg
    op or the conversion to the target mma fails.
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$target,
    IREEGPU_AnyMmaAttr:$intrinsic_kind);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = [{
    $target `,` `kind` `(` $intrinsic_kind `)` attr-dict
    `:` functional-type(operands, results)
  }];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::linalg::LinalgOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def DistributeMultiMmaOp : Op<Transform_Dialect, "iree.distribute_multi_mma",
    [TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Distributes the target multi_mma op to lanes.

    #### Return modes
    Emits a definite failure if the target is not an iree_gpu.multi_mma op or
    if it fails to distribute.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = [{
    $target attr-dict `:` functional-type(operands, results)
  }];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMEXTENSIONS_IREEGPUEXTENSIONS
