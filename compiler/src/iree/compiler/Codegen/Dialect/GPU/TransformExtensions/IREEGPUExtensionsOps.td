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

def ApplyDropInnerTiledOpUnitDims : Op<Transform_Dialect,
    "apply_patterns.iree.drop_inner_tiled_unit_dims",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Populate patterns to drop the unit dims from inner_tiled ops with
    only unit iteration bounds.
  }];

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let assemblyFormat = "attr-dict";
}

def ApplyLowerInnerTiledOp : Op<Transform_Dialect,
    "apply_patterns.iree.lower_inner_tiled",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Populate patterns to lowering inner_tiled ops to the intrinsic specified by
    the |kind| attribute.
  }];

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let assemblyFormat = "attr-dict";
}

def ApplyLowerBarrierRegionPatternsOp : Op<Transform_Dialect,
    "apply_patterns.iree.lower_barrier_region",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Populate patterns that lowers iree_gpu.barrier_region ops to allocations
    and copies.
  }];

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let assemblyFormat = "attr-dict";
}

def ApplyLowerValueBarrierOp : Op<Transform_Dialect,
    "apply_patterns.iree.lower_value_barrier",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Populate patterns to convert value barriers on vectors into gpu.barrier ops.
    Barriers on tensors are ignored.
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

def ApplyVectorizeIREEGPUOp : Op<Transform_Dialect,
    "apply_patterns.iree.vectorize_iree_gpu",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Populate patterns to vectorize various iree_gpu ops. Expected to run
    before or as a part of a larger vectorization pass.
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

def DistributeInnerTiledOp : Op<Transform_Dialect, "iree.distribute_inner_tiled",
    [TransformEachOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Distributes the target inner_tiled op to lanes.

    #### Return modes
    Emits a definite failure if the target is not an iree_codegen.inner_tiled op or
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

def ForallToLanesOp : Op<Transform_Dialect,
    "iree.forall_to_lanes",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     TransformEachOpTrait,
     TransformOpInterface,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Collect all of the scf.forall ops in the target that are distributed to
    lanes.

    Only scf.forall distributed to exactly a single lane id are currently
    supported.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` functional-type($target, results)";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def FuseForallOp : Op<Transform_Dialect, "iree.fuse_forall",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Fuses a producer-consumer pair of scf.forall ops that share the same
    iterator mapping types and trip counts. An allocation is created to
    bridge the `parallel.insert_slice` of the producer with the per-thread
    `extract_slice` of the consumer. If specified, uses |address_space| for
    the intermediate allocation.

    The mapping attributes of both the producer and consumer `scf.forall` ops
    must be in a relative descending order, for example:
     [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>]
    or
     [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]

    NOTE: This pattern implicitly REQUIRES that the resulting scf.forall
    is capable of synchronizing all threads at the point of fusion (i.e.
    inserting a barrier). This invalidates certain kinds of lowerings of
    scf.forall ops such as lowering it to loops.

    #### Return modes
    Emits a definite failure if either the producer or consumer are not
    scf.forall ops.
  }];

  let arguments = (
      ins TransformHandleTypeInterface:$producer,
          TransformHandleTypeInterface:$consumer,
          OptionalAttr<AnyAttr>:$address_space
  );
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = [{
    $producer `into` $consumer attr-dict
    `:` functional-type(operands, results)
  }];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

def FuseCollapseShapeIntoForallOp : Op<Transform_Dialect, "iree.fuse_collapse_shape_into_forall",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Fuses a consumer tensor.collapse_shape op into a producer scf.forall op.
    The users of the block argument for the corresponding forall output operand
    should be only a tensor.parallel_insert_slice op, and tensor.extract_slice
    ops that extract an equivalent subset. After the fusion, the output of the
    forall will be collapsed, and all users of this block arg will also be
    collapsed. Additional tensor.expand_shape ops will be inserted after any
    tensor.extract_slice users inside the forall so that types match. Similarly,
    a tensor.collapse_shape will be inserted before the
    tensor.parallel_insert_slice.

    #### Return modes
    Emits a definite failure if either the producer is not an scf.forall op or
    if the consumer is not a tensor.collapse_shape op.
  }];

  let arguments = (
      ins TransformHandleTypeInterface:$producer,
          TransformHandleTypeInterface:$consumer
  );
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = [{
    $consumer `into` $producer attr-dict
    `:` functional-type(operands, results)
  }];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

def FuseExtractSliceIntoForallOp : Op<Transform_Dialect, "iree.fuse_extract_slice_into_forall",
    [FunctionalStyleTransformOpTrait,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Fuses a consumer tensor.extract_slice op into a producer scf.forall op.
    This transform is supported if the extract_slice op has all zero offsets,
    and if all the offsets, sizes, and strides dominate the scf.forall op.
    After the transformation, the forall loop output argument corresponding
    to the sliced result will be replaced with a slice of it with the same
    offsets, sizes, and strides as the original extract_slice. The source of
    the corresponding tensor.parallel_insert_slice of the scf.forall will also
    become a slice of the original parallel insert source, clamped to fit within
    the new sliced result tensor.

    #### Return modes
    Emits a definite failure if either the producer is not an scf.forall op or
    if the consumer is not a tensor.extract_slice op.
  }];

  let arguments = (
      ins TransformHandleTypeInterface:$producer,
          TransformHandleTypeInterface:$consumer
  );
  let results = (outs TransformHandleTypeInterface:$result);

  let assemblyFormat = [{
    $consumer `into` $producer attr-dict
    `:` functional-type(operands, results)
  }];
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMEXTENSIONS_IREEGPUEXTENSIONS
