// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_IREEGPUOPS
#define IREE_CODEGEN_DIALECT_IREEGPUOPS

include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.td"
include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.td"
include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MultiMmaOp
//===----------------------------------------------------------------------===//

def IREEGPU_MultiMmaOp : Op<IREEGPU_Dialect, "multi_mma", [
    Pure,
    AllTypesMatch<["acc", "result"]>,
    ]> {
  let summary = "Models a contraction of multiple mma operations";
  let description = [{
    Computes the sum of inner MMA operations along a set of outer dimensions.
    Logically matches closely with a `vector.contraction` operation, however
    the combiner type is a specific intrinsic rather than a generic combiner
    type.

    Similar to `vector.contraction`, an iterator type attribute list must be
    specified, where each element of the list represents an iterator over one
    of the outer dimensions. Iteration of inner dimensions is defined solely by
    the intrinsic and may be opaque.

    An indexing map attribute list must be specified with an entry for lhs, rhs
    and acc arguments. An indexing map attribute specifies a mapping from each
    outer loop iterator in the iterator type list, to each dimension of each
    operand.

    The combiner type is defined by the intrinsic.

    Example:

    ```mlir
    #contraction_accesses = [
     affine_map<(i, j, k) -> (i, k)>,
     affine_map<(i, j, k) -> (k, j)>,
     affine_map<(i, j, k) -> (i, j)>
    ]
    #contraction_trait = {
      indexing_maps = #contraction_accesses,
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
    }
    %3 = iree_gpu.multi_mma %0, %1, %2 #contraction_trait
      : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>

    // Takes tensors as well, however the inner dimensions must always be
    // static.
    %7 = iree_gpu.multi_mma %4, %5, %6 #contraction_trait
      : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
    ```

    The example above can be logically lowered directly to loops like this
    (ignoring type conversions from tensor to vector needed for the mfma).
    ```
    %outer_m = tensor.dim %6, %c0 : index
    %outer_n = tensor.dim %6, %c1 : index
    %outer_k = tensor.dim %4, %c1 : index
    %7 = scf.for %i = %c0 to %outer_m iter_args(%arg0 = %6) {
      %8 = scf.for %j = %c0 to %outer_n iter_args(%arg1 = %arg0) {
        %9 = scf.for %k = %c0 to %outer_k iter_args(%arg2 = %arg1) {
          %lhs = tensor.extract_slice %4 [%i, %k, 0] [1, 1, 4] [1, 1, 1] : tensor<4xf16>
          %rhs = tensor.extract_slice %5 [%k, %j, 0] [1, 1, 4] [1, 1, 1] : tensor<4xf16>
          %acc = tensor.extract_slice %arg2 [%i, %j, 0] [1, 1, 4] [1, 1, 1] : tensor<4xf32>
          %res = amdgpu.mfma %lhs, %rhs, %acc : tensor<4xf32>
          %ret = tensor.insert_slice %acc into %arg2 [%i, %j, 0] [1, 1, 4] [1, 1, 1] : tensor<?x?x4xf32>
          scf.yield %ret : tensor<?x?x4xf32>
        }
        scf.yield %9 : tensor<?x?x4xf32>
      }
      scf.yield %8 : tensor<?x?x4xf32>
    }
    ```

    Or alternatively unrolled to a single intrinsic when operation on vectors.
    ```mlir
    #contraction_accesses = [
     affine_map<() -> ()>,
     affine_map<() -> ()>,
     affine_map<() -> ()>
    ]
    #contraction_trait = {
      indexing_maps = #contraction_accesses,
      iterator_types = [],
      kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
    }
    %3 = iree_gpu.multi_mma %0, %1, %2 #contraction_trait
      : vector<4xf16>, vector<4xf16> into vector<4xf32>
    ```

    ## Motivation, Design Choices, and Pitfalls

    The idea behind this operation is to decouple the layout setting/tiling
    required to target certain intrinsics from the lowering to them. Because
    typically tiling of this sort happens on tensor operands, however the target
    intrinsics operate on vectors, we use this operation to bridge the gap. The
    choice for a shared operation is intended to ease the lowering process and
    allow for different transformations at different stages of the pipeline
    without needing to essentially clone this op.

    The choice to let the inner dimensions required to compute the intrinsic be
    implicit based on the indexing maps was made to make this operation easier
    to generate and to skip the need for type conversion ops. However this comes
    at the expense of ease of verification for the operation. It is also
    implicitly linked to a lane-level parent `scf.forall` operation.
  }];

  let arguments = (ins
    AnyRankedTensorOrVector:$lhs,
    AnyRankedTensorOrVector:$rhs,
    AnyRankedTensorOrVector:$acc,
    ArrayAttr:$indexing_maps,
    IREEGPU_IteratorTypeArrayAttr:$iterator_types,
    IREEGPU_AnyMmaAttr:$kind
  );
  let results = (outs
    AnyRankedTensorOrVector:$result
  );

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc attr-dict
    `:` type($lhs) `,` type($rhs) `into` type($acc)
  }];

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$acc,
      "ArrayAttr":$indexingMaps, "ArrayAttr":$iteratorTypes,
      "MmaInterfaceAttr":$intrinsic)>,
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$acc,
      "ArrayRef<ArrayRef<AffineExpr>>":$indexingExprs,
      "ArrayRef<utils::IteratorType>":$iteratorTypes,
      "MmaInterfaceAttr":$intrinsic)>
  ];
  let extraClassDeclaration = [{
    ::mlir::ShapedType getLhsType() {
      return ::llvm::cast<::mlir::ShapedType>(getLhs().getType());
    }
    ::mlir::ShapedType getRhsType() {
      return ::llvm::cast<::mlir::ShapedType>(getRhs().getType());
    }
    ::mlir::ShapedType getAccType() {
      return ::llvm::cast<::mlir::ShapedType>(getAcc().getType());
    }
    ::mlir::ShapedType getResultType() {
      return ::llvm::cast<::mlir::ShapedType>(getResult().getType());
    }

    llvm::SmallVector<::mlir::AffineMap, 4> getIndexingMapsArray() {
      return llvm::to_vector<4>(getIndexingMaps().getAsValueRange<::mlir::AffineMapAttr>());
    }

    // Returns the bounds of each dimension in the iteration space spanned
    // by the iterator types of this operation.
    void getIterationBounds(SmallVectorImpl<int64_t> &iterationBounds);

    // Returns a list of index maps, where there is a list entry for each
    // op indexing map attribute (i.e. one for each input and output, with
    // the output listed last). Each index map, maps from this operations
    // iteration space, to vector dimensions of the maps input/output.
    void getIterationIndexMap(
      std::vector<DenseMap<int64_t, int64_t>> &iterationIndexMap);

    SmallVector<utils::IteratorType> getIteratorTypesArray() {
      auto range =
          getIteratorTypes()
              .template getAsValueRange<IteratorTypeAttr, utils::IteratorType>();
      return {range.begin(), range.end()};
    }
  }];

  let hasVerifier = 1;
}

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
