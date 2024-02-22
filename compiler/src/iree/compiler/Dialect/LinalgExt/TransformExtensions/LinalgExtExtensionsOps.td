// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_TRANSFORMOPS
#define IREE_DIALECT_LINALGEXT_TRANSFORMOPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def TileAttentionOp : Op<Transform_Dialect, "iree.tile_attention",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformOpInterface,
     TransformEachOpTrait,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Target iree_linalg_ext.attention ops and tile them.
    This transform consumes the target handle and produces a result handle.
  }];

  let arguments = (
      ins TransformHandleTypeInterface:$target,
          OptionalAttr<I64Attr>:$tile_size
  );
  let results = (outs Variadic<TransformHandleTypeInterface>:$result);

  let assemblyFormat = "attr-dict $target `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";

  let builders = [
    OpBuilder<(ins "Value":$target)>
  ];

  let assemblyFormat = [{
    $target attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::iree_compiler::IREE::LinalgExt::AttentionOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def DecomposeTiledAttentionOp : Op<Transform_Dialect, "iree.decompose_tiled_attention",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformOpInterface,
     TransformEachOpTrait,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Target iree_linalg_ext.attention ops and decompose them.
    This transform consumes the target handle and produces a result handle.
  }];

  let arguments = (
      ins TransformHandleTypeInterface:$target,
          OptionalAttr<I64Attr>:$tile_size
  );
  let results = (outs Variadic<TransformHandleTypeInterface>:$result);

  let assemblyFormat = "attr-dict $target `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";

  let builders = [
    OpBuilder<(ins "Value":$target)>
  ];

  let assemblyFormat = [{
    $target attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::iree_compiler::IREE::LinalgExt::AttentionOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // IREE_DIALECT_LINALGEXT_TRANSFORMOPS
