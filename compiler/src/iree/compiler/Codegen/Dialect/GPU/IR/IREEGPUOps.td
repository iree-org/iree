// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_IREEGPUOPS
#define IREE_CODEGEN_DIALECT_IREEGPUOPS

include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ShuffleTensorOp
//===----------------------------------------------------------------------===//

def IREEGPU_ShuffleTensorOp : Op<IREEGPU_Dialect, "shuffle_tensor", [
    Pure,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"mlir::iree_compiler::IREE::GPU::YieldOp">
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
        %slice = iree_gpu.shuffle_tensor %in[%2, %3] [2, 4] [1, 1] to %empty {
        ^bb0(%intermediate: tensor<4x128xf32>):
          %slice = tensor.extract_slice %intermediate[0, %4] [4, 16] [1, 1] : tensor<4x128xf32> to tensor<4x16xf32>
          iree_gpu.yield %slice : tensor<4x16xf32>
        } : tensor<2x4xf32> -> tensor<4x128xf32> -> tensor<4x16xf32>
        ...
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    ```

    A shuffle can be lowered to a shared allocation with a write of the source
    slice, a barrier, inlining the body of the shuffle op (the read), and then
    a barrier to synchronize all workers on the result of the read. Note that
    it is undefined behavior if there are any conflicting writes to the
    intermediate. Also to execute the barrier, any lowerings of the enclosing
    `scf.forall` to serial loops is invalid. In other words, the lowerings must
    provide the number of workers requested by the loop.

    This op takes an input |source| tensor to represent the slice held by this
    worker before the shuffle, an intermediate tensor |dest| that all workers
    insert into, and performs a synchronized read from that intermediate
    tensor.

    It is undefined behavior if the source tensor is out of bounds of the
    intermediate allocation.

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
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides,
    AnyRankedTensor:$dest
  );
  let regions = (region SizedRegion<1>:$region);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    `to` $dest $region attr-dict
    `:` type($source) `->` type($dest) `->` type($result)
  }];

  let extraClassDeclaration = [{
    RankedTensorType getSourceType() {
      return getSource().getType();
    }

    RankedTensorType getDestType() {
      return getDest().getType();
    }

    // Source slice view-like getters.
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedOffsets() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticOffsets(),
                                    getOffsets(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedSizes() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticSizes(),
                                    getSizes(), b);
    }
    ::llvm::SmallVector<::mlir::OpFoldResult, 4> getMixedStrides() {
      Builder b(getContext());
      return ::mlir::getMixedValues(getStaticStrides(),
                                    getStrides(), b);
    }
  }];

  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def IREEGPU_YieldOp : Op<IREEGPU_Dialect, "yield", [
    Pure, ReturnLike, Terminator,
    HasParent<"::mlir::iree_compiler::IREE::GPU::ShuffleTensorOp">]> {
  let summary = "Yield a value from a region";
  let description = [{
     This operation is used to yield a single value from a within a region.
  }];

  let arguments = (ins AnyType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

#endif // IREE_CODEGEN_DIALECT_IREEGPUOPS
