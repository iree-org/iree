// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_TRANSFORMOPS
#define IREE_DIALECT_LINALGEXT_TRANSFORMOPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def DecomposeOnlineAttention : Op<Transform_Dialect, "iree.decompose_online_attention",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformOpInterface,
     TransformEachOpTrait,
     ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Target iree_linalg_ext.online_attention ops and decompose them.
    This transform consumes the target handle and produces a result handle.

    # TODO: This op should be replaced by upstream
    transform.structured.decompose_interface, which does not support multiple
    results for now.
  }];

  let arguments = ( ins TransformHandleTypeInterface:$target);
  let results = (outs Variadic<TransformHandleTypeInterface>:$result);

  let assemblyFormat = "attr-dict $target `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";

  let assemblyFormat = [{
    $target attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::iree_compiler::IREE::LinalgExt::OnlineAttentionOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def ConvertToOnlineAttention : Op<Transform_Dialect, "iree.convert_to_online_attention",
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
      ins TransformHandleTypeInterface:$target
  );
  let results = (outs Variadic<TransformHandleTypeInterface>:$result);

  let assemblyFormat = "attr-dict $target `:` functional-type(operands, results)";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";

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
