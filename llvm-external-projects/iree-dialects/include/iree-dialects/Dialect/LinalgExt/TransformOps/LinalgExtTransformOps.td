// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_TRANSFORMOPS
#define IREE_DIALECT_LINALGEXT_TRANSFORMOPS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def FuseProducersOp : Op<Transform_Dialect, "fuse_producers",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{Fuses the producers for the operands to fuse.}];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$operands_to_fuse);
  let results = (outs PDL_Operation:$transformed,
                      Variadic<PDL_Operation>:$fused_ops);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";
}

def RewriteForallToAsyncOp :
  Op<Transform_Dialect, "forall_to_async",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {

  let description = [{
    Rewrite a bufferized scf.forall op to the async dialect.

    Return modes:
    =============
    This operation ignores non-Linalg ops and drops them in the return.
    This transform is currently only implemented for 1-D scf.forall that
    have been bufferized and definitely fail for the rest.

    If all the operations referred to by the `target` PDLOperation lower
    properly, the transform succeeds. Otherwise the transform silently fails.

    The returned handle points to only the subset of successfully produced
    async operations, which can all be empty.
  }];
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::scf::ForallOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def RewriteForallToScfForOp :
  Op<Transform_Dialect, "forall_to_scf_for",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {

  let description = [{
    Rewrite a bufferized scf.forall to a sequential scf.for.

    Return modes:
    =============
    This operation ignores non-Linalg ops and drops them in the return.
    This transform is currently only implemented for 1-D scf.forall that
    have been bufferized and definitely fail for the rest.

    If all the operations referred to by the `target` PDLOperation lower
    properly, the transform succeeds. Otherwise the transform silently fails.

    The returned handle points to only the subset of successfully produced
    scf.for operations, which can all be empty.
  }];
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::scf::ForallOp target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // IREE_DIALECT_LINALGEXT_TRANSFORMOPS
