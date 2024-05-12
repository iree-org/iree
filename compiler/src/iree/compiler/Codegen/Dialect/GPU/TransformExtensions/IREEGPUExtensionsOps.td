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

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMEXTENSIONS_IREEGPUEXTENSIONS
