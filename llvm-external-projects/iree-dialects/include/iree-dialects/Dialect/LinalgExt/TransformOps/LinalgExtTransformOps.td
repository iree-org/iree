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


def FunctionalStyleMultiOperandMultiResultTransformOpTrait
    : NativeOpTrait<"FunctionalStyleMultiOperandMultiResultTransformOpTrait"> {
  let cppNamespace = "::mlir::transform";
}

def FuseProducersOp : Op<Transform_Dialect, "fuse_producers",
    [FunctionalStyleMultiOperandMultiResultTransformOpTrait, 
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

def TileToLinalgExtTileOp :
    Op<Transform_Dialect, "tile_to_iree_linalg_ext_tile_op",
      [FunctionalStyleMultiOperandMultiResultTransformOpTrait,
       MemoryEffectsOpInterface,
       DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{Tile a linalg op with linalg_ext.tile op along a single
  dimension.}];

  let summary = [{
    0 should be used as a tile size to skip tiling a particular dimension.
    This is a commonly used convention in Linalg.
    For the purpose of this transformation, a single non-zero positive tile
    size is allowed.
    }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$sizes);
  let results = (outs PDL_Operation:$tiled_op,
                      PDL_Operation:$tile_op);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";
}

def FuseIntoContainingOp :
    Op<Transform_Dialect, "fuse_into_containing_op",
      [FunctionalStyleMultiOperandMultiResultTransformOpTrait,
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

def RewriteLinalgExtTileToScfForOp :
  Op<Transform_Dialect, "rewrite_iree_linalg_ext_tile_to_scf_for",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformOpInterface,
     TransformEachOpTrait]> {

  let description = [{Rewrite linalg_ext.tile op to scf.for.}];
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::scf::ForOp> applyToOne(
        ::mlir::iree_compiler::IREE::LinalgExt::TileOp target);
  }];
}

def RewriteLinalgExtTileToInParallelOp :
  Op<Transform_Dialect, "rewrite_iree_linalg_ext_tile_to_in_parallel",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {

  let description = [{Rewrite linalg_ext.tile op to linalg_ext.in_parallel.}];
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::iree_compiler::IREE::LinalgExt::InParallelOp> applyToOne(
        ::mlir::iree_compiler::IREE::LinalgExt::TileOp target);
  }];
}

def RewriteLinalgExtInParallelToAsyncOp :
  Op<Transform_Dialect, "rewrite_iree_linalg_ext_in_parallel_to_async",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {

  let description = [{Rewrite linalg_ext.in_parallel op to the async dialect.}];
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::Operation *> applyToOne(
        ::mlir::iree_compiler::IREE::LinalgExt::InParallelOp target);
  }];
}

def RewriteLinalgExtInParallelToHALOp :
  Op<Transform_Dialect, "rewrite_iree_linalg_ext_in_parallel_to_hal",
    // Cannot use FunctionalStyleTransformOpTrait: this returns 0 results atm.
    [FunctionalStyleMultiOperandMultiResultTransformOpTrait,
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {

  let description = [{Rewrite linalg_ext.in_parallel op to use HAL ops.}];
  let arguments = (ins PDL_Operation:$target);
  // TODO: Determine whether we want to return something here, the only natural
  // results would be the resulting insertTensorOps.
  // let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";
}

def RewriteLinalgExtInParallelToScfForOp :
  Op<Transform_Dialect, "rewrite_iree_linalg_ext_in_parallel_to_scf_for",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {

  let description = [{Rewrite linalg_ext.in_parallel to a sequential scf.for.

    Warning: when the linalg_ext.in_parallel terminator operates on tensors,
    this is a potentially dangerous transformation under the current semantics.
    In order for this transformation to be semantics-preserving, 2 conditions need
    to come together that are currently not checked and the subject of future analyses:
      1. The terminator of linalg_ext.in_parallel may compose the output tensor in
      potentially undefined ways: if the linalg_ext.parallel_insert_slice regions overlap, they
      may occur in any order and the result is unspecified. A future overlap/intersection
      analysis will be needed to guard against this case.
      2. Even when condition 1. has well-defined behavior, semantics altering behavior may
      still be introduced to simplify inplace bufferization. In the current implementation,
      all linalg_ext.parallel_insert_slice dest operands are optimistically turned into scf.for
      iter_args. This is optimistic because any use of a tensor inside linalg_ext.in_parallel
      is guaranteed to see the value before the start of the op; whereas the same use may
      see a partially updated sequential result in the scf.for op.
      An extra analysis is needed to ensure that a particular read of a result tensor would
      see the initial value and not a partial update. This is guaranteed by construction when
      the linalg_ext.in_parallel op is produced by lowering a linalg_ext.tile operation but
      is not something that is generally enforced in the IR.
      For the moment we perform the replacement of the use with the scf.for iter_arg to be
      able to connect pieces inplace with bufferization. In the future an analysis will be
      needed to ensure correctness of this lowering to sequential scf.for + iter_args.
  }];
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::scf::ForOp> applyToOne(
        ::mlir::iree_compiler::IREE::LinalgExt::InParallelOp target);
  }];
}

#endif // IREE_DIALECT_LINALGEXT_TRANSFORMOPS
