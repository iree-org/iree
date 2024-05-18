// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMEXTENSIONS_IREEGPUEXTENSIONS
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMEXTENSIONS_IREEGPUEXTENSIONS

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
    Populate patterns to lowering multi_mma ops to the intrinsic specified by
    the |kind| attribute.
  }];

  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
  let assemblyFormat = "attr-dict";
}

def ApplyLowerShuffleTensorPatternsOp : Op<Transform_Dialect,
    "apply_patterns.iree.lower_shuffle_tensor",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Populate patterns that lowers iree_gpu.shuffle_tensor ops to allocations
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

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMEXTENSIONS_IREEGPUEXTENSIONS
