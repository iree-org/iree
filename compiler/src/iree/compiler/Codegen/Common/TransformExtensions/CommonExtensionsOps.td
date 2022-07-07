// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONS
#define IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def ApplyPatternsOp : Op<Transform_Dialect, "iree.apply_patterns",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformEachOpTrait, 
     TransformOpInterface]> {
  let description = [{
    Greedily applies patterns as specified by its attributes.
    
    Must be applied to an op with trait IsolatedFromAbove since the 
    GreedyPatternRewriter asserts those.

    Returns the IsolatedFromAbove op whose content it has modified for better
    chaining APIs.

    The following additive attributes can be set, they add patterns in an 
    unspecified order:
      - canonicalization: adds all the canonicalization pattenrs of all 
      registered dialects and ops.
      - rank_reducing: adds patterns that results in rank-reducing behavior on
      subset-based operations.
  }];

  let arguments = (ins PDL_Operation:$target,
                       UnitAttr:$canonicalization,
                       UnitAttr:$rank_reducing);
  let results = (outs PDL_Operation:$result);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::Operation *> applyToOne(
      ::mlir::Operation *target, ::mlir::transform::TransformState &state);
  }];
}

def IREEBufferizeOp : Op<Transform_Dialect, "iree.bufferize",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Call upstream comprehensive bufferize with extra IREE hooks.

    By default, CPU allocations are emitted. This behavior can be modified by 
    using the following attributes:
      - target_gpu: if set, GPU allocations are emitted.
  }];

  let arguments = (ins UnitAttr:$target_gpu);
  let results = (outs);

  let assemblyFormat = "attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::transform_dialect";
}

#endif // IREE_COMPILER_CODEGEN_COMMON_TRANSFORMEXTENSIONS_COMMONEXTENSIONS
