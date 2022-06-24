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

def TileToForeachOp :
    Op<Transform_Dialect, "tile_to_foreach_thread_op",
      [FunctionalStyleTransformOpTrait,
       MemoryEffectsOpInterface,
       TransformEachOpTrait,
       TransformOpInterface]> {
  let description = [{Tile a linalg op to `scf.foreach_thread`.}];

  let summary = [{
    0 should be used as a tile size to skip tiling a particular dimension.
    This is a commonly used convention in Linalg.
    }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$num_threads);
  let results = (outs PDL_Operation:$tiled_op,
                      PDL_Operation:$tile_op);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::llvm::SmallVector<::mlir::Operation *>> applyToOne(
      ::mlir::linalg::LinalgOp target, transform::TransformState &state);
  }];
}

def FuseIntoContainingOp :
    Op<Transform_Dialect, "fuse_into_containing_op",
      [FunctionalStyleTransformOpTrait,
       MemoryEffectsOpInterface,
       DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{Fuse a producer into a containing operation.}];

  let summary = [{
    Search the body of the containing operation for all producer uses and
    compute the accessed producer slices on-the-fly.
    }];

  let arguments = (ins PDL_Operation:$producer_op,
                       PDL_Operation:$containing_op);

  let assemblyFormat = "$producer_op `into` $containing_op attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";
}

def RewriteForeachThreadToAsyncOp :
  Op<Transform_Dialect, "foreach_thread_to_async",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {

  let description = [{
    Rewrite a bufferized scf.foreach_thread op to the async dialect.
  }];
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::Operation *> applyToOne(
      ::mlir::scf::ForeachThreadOp target, transform::TransformState &state);
  }];
}

def RewriteForeachThreadToScfForOp :
  Op<Transform_Dialect, "foreach_thread_to_scf_for",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {

  let description = [{
    Rewrite a bufferized scf.foreach_thread to a sequential scf.for.
  }];
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::scf::ForOp> applyToOne(
        ::mlir::scf::ForeachThreadOp target, transform::TransformState &state);
  }];
}

#endif // IREE_DIALECT_LINALGEXT_TRANSFORMOPS
