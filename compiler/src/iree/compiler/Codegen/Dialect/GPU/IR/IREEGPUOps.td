// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_IREEGPUOPS
#define IREE_CODEGEN_DIALECT_IREEGPUOPS

include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

def MemrefTypeAttr : TypeAttrBase<"MemRefType", "Memref type attribute">;

//===----------------------------------------------------------------------===//
// ShuffleTensorOp
//===----------------------------------------------------------------------===//

def IREEGPU_ShuffleTensorOp : Op<IREEGPU_Dialect, "shuffle_tensor", [
    Pure,
    AttrSizedOperandSegments
    ]> {
  let summary = "Shuffles a private tensor across a shared allocation";
  let description = [{
    This op is designed to represent a shuffle of private tensor data
    collectively held across a set of workers. This operation naturally arises
    when combining the regions of producer-consumer `scf.forall` operations
    that share a mapping type and worker count.

    For example, consider the following pair of parallel loops.
    ```mlir
      %0 = scf.forall (%idy, %idx) in (2, 32) shared_outs(%init = %empty) -> (tensor<4x128xf32>) {
        %in = ...
        %2 = affine.apply #affine_map<(d0) -> (d0 * 2)> (%idy)
        %3 = affine.apply #affine_map<(d0) -> (d0 * 4)> (%idx)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %in into %init[%2, %3] [2, 4] [1, 1]
            : tensor<2x4xf32> into tensor<4x128xf32>
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      %1 = scf.forall (%idy, %idx) in (8, 8) -> (tensor<128x128xf32>) {
        %4 = affine.apply #affine_map<(d0) -> (d0 * 16)> (%idx)
        %extracted_slice = tensor.extract_slice %0[0, %4] [4, 16] [1, 1]
          : tensor<4x128xf32> to tensor<4x16xf32>
        ...
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    ```

    Because these loops share the same worker type and total count, the bodies
    of these two loops can be merged with a barrier and a shuffle where the
    boundary of the loops currently is.

    ```mlir
      %0 = scf.forall (%idy, %idx) in (8, 8) -> (tensor<4x128xf32>) {
        %ids = affine.delinearize_index %idy * 8 + %idx to (2, 32) : index
        %in = ...
        %2 = affine.apply #affine_map<(d0) -> (d0 * 2)> (%ids#0)
        %3 = affine.apply #affine_map<(d0) -> (d0 * 4)> (%ids#1)
        %4 = affine.apply #affine_map<(d0) -> (d0 * 16)> (%idx)
        %slice = iree_gpu.shuffle_tensor %in[%2, %3] [2, 4] [1, 1] to %empty[0, %4] [4, 16] [1, 1]
          : tensor<2x4xf32> -> tensor<4x128xf32> -> tensor<4x16xf32>
        ...
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    ```

    A shuffle can be lowered to a shared allocation with a write of the source
    slice, a barrier, and a read of the result slice. Note that to avoid both
    conflicting writes, and to execute the barrier, this renders any lowerings
    of the enclosing `scf.forall` to serial loops invalid. In other words, the
    lowerings/hardware must provide the number of workers requested by the loop.

    It is undefined behavior if the source or result tensor slices are out of
    bounds of the intermediate allocation.

    Movtivation and Intended Use Cases:

    The primary way this op is generated is when fusing parallel loops with
    tensor results. This operation helps to make lowerings more progressive
    and flexible.
      - Rather than lowering straight to vector ops for the reads/writes
        for the shuffle, this allows separating out the vectorization of the
        shared memory accesses from earlier tiling steps.
      - Lowering directly to an alloc + reads and writes breaks the dependency
        chain making transformations like barrier placement and pipelining
        potentially more difficult.
      - Allows the option of non-vector based lowering paths.
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    Variadic<Index>:$source_offsets,
    Variadic<Index>:$source_sizes,
    Variadic<Index>:$source_strides,
    DenseI64ArrayAttr:$static_source_offsets,
    DenseI64ArrayAttr:$static_source_sizes,
    DenseI64ArrayAttr:$static_source_strides,
    AnyRankedTensor:$shared_alloc,
    Variadic<Index>:$result_offsets,
    Variadic<Index>:$result_sizes,
    Variadic<Index>:$result_strides,
    DenseI64ArrayAttr:$static_result_offsets,
    DenseI64ArrayAttr:$static_result_sizes,
    DenseI64ArrayAttr:$static_result_strides
  );
  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($source_offsets, $static_source_offsets)
    custom<DynamicIndexList>($source_sizes, $static_source_sizes)
    custom<DynamicIndexList>($source_strides, $static_source_strides)
    `to` $shared_alloc
    custom<DynamicIndexList>($result_offsets, $static_result_offsets)
    custom<DynamicIndexList>($result_sizes, $static_result_sizes)
    custom<DynamicIndexList>($result_strides, $static_result_strides)
    attr-dict `:` type($source) `->` type($shared_alloc) `->` type($result)
  }];

  let extraClassDeclaration = [{
    RankedTensorType getSourceType() {
      return getSource().getType();
    }

    RankedTensorType getSharedAllocType() {
      return getSharedAlloc().getType();
    }

    // Because we have two sets of offsets/sizes/strides, we cannot use
    // interface boilerplate and instead redefine it.

    // Source slice view-like getters.
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedSourceOffsets() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticSourceOffsets(),
                                    getSourceOffsets(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedSourceSizes() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticSourceSizes(),
                                    getSourceSizes(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedSourceStrides() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticSourceStrides(),
                                    getSourceStrides(), b);
    }

    // Result slice view-like getters.
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedResultOffsets() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticResultOffsets(),
                                    getResultOffsets(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedResultSizes() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticResultSizes(),
                                    getResultSizes(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedResultStrides() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticResultStrides(),
                                    getResultStrides(), b);
    }
  }];

  let hasVerifier = 1;
}

#endif // IREE_CODEGEN_DIALECT_IREEGPUOPS
