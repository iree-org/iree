// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_TRANSFORMOPS
#define IREE_DIALECT_LINALGEXT_TRANSFORMOPS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformEffects.td"
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
  let description = [{
    Tile a linalg op to `scf.foreach_thread`.
    
    0 should be used as a tile size to skip tiling a particular dimension.
    This is a commonly used convention in Linalg.
    
    Return modes:
    =============
    This operation ignores non-Linalg ops and drops them in the return.

    If all the operations referred to by the `target` PDLOperation tile
    properly, the transform succeeds. Otherwise the transform silently fails.
    
    The 2 returned handles points to only the subset of successfully produced 
    tiled operations, which can all be empty.
    This 2 returned handles point to:
      - the new scf.foreach_thread op,
      - the tiled linalg op used.
  }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$num_threads,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$thread_dim_mapping);
  let results = (outs PDL_Operation:$tile_op,
                      PDL_Operation:$tiled_op);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "mlir::iree_compiler::IREE::LinalgExt";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::linalg::LinalgOp target, 
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
  }];
}

def FuseIntoContainingOp :
    Op<Transform_Dialect, "fuse_into_containing_op",
      [DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{Fuse a producer into a containing operation.}];

  let summary = [{
    Fuse the `producer_op` into the `containing_op`. Only scf.foreach_thread
    ops are currently supported as the containing op.

    The producer is typically a slice of a tileable op (i.e., implements
    TilingInterface). In that case, this transform computes the accessed
    producer slice inside of the containing op ("tile and fuse"). Otherwise,
    the entire producer is cloned inside the containing op ("clone and fuse").

    Producers and containing ops are matched pairwise when multiple payload ops
    are provided (batched execution). The number of producers and containing
    ops must be the same.

    #### Return modes

    This operation fails definitely when a containing op is not an
    scf.foreach_thread op. If at least one producer could not be fused, this
    operation fails silently.

    This operation reads and frees the producer handle. It reads the containing
    op handle.
  }];

  let arguments = (ins Arg<PDL_Operation, "",
                           [TransformMappingRead,
                            TransformMappingFree]>:$producer_op,
                       Arg<PDL_Operation, "",
                           [TransformMappingRead]>:$containing_op);

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
        
    Return modes:
    =============
    This operation ignores non-Linalg ops and drops them in the return.
    This transform is currently only implemented for 1-D scf.foreach_thread that
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
        ::mlir::scf::ForeachThreadOp target, 
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
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
            
    Return modes:
    =============
    This operation ignores non-Linalg ops and drops them in the return.
    This transform is currently only implemented for 1-D scf.foreach_thread that
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
        ::mlir::scf::ForeachThreadOp target, 
        ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
        ::mlir::transform::TransformState &state);
  }];
}

#endif // IREE_DIALECT_LINALGEXT_TRANSFORMOPS
