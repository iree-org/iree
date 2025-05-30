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
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// BarrierRegionOp
//===----------------------------------------------------------------------===//

def IREEGPU_BarrierRegionOp : Op<IREEGPU_Dialect, "barrier_region", [
    Pure,
    SingleBlockImplicitTerminator<"mlir::iree_compiler::IREE::GPU::YieldOp">
    ]> {
  let summary = [{Synchronizes workers on a region of shared code.}];
  let description = [{
    This op is designed to represent synchronization of workers on the operands
    and results of the given region. This operation naturally arises when combining
    the regions of producer-consumer `scf.forall` operations that share a
    mapping type.

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
    of these two loops can be merged with a barrier an insert_slice and a
    shuffle where the boundary of the loops currently is.

    ```mlir
      %0 = scf.forall (%idy, %idx) in (8, 8) -> (tensor<4x128xf32>) {
        %alloc = bufferization.alloc_tensor {memory_space = #gpu.address_space<workgroup>}
          : tensor<4x128xf32>
        %barrier = iree_gpu.barrier_region %alloc {
        ^bb0(%shared: tensor<4x128xf32>):
          %ids = affine.delinearize_index %idy * 8 + %idx to (2, 32) : index
          %in = ...
          %2 = affine.apply #affine_map<(d0) -> (d0 * 2)> (%ids#0)
          %3 = affine.apply #affine_map<(d0) -> (d0 * 4)> (%ids#1)
          %inserted_slice = tensor.insert_slice %in into %shared[%2, %3] [2, 4] [1, 1]
            : tensor<2x4xf32> to tensor<4x128xf32>
          iree_gpu.yield %slice : tensor<4x16xf32>
        } : tensor<4x128xf32> -> tensor<4x16xf32>
        %4 = affine.apply #affine_map<(d0) -> (d0 * 16)> (%idx)
        %slice = tensor.extract_slice %barrier[0, %4] [4, 16] [1, 1] : tensor<4x128xf32> to tensor<4x16xf32>
        ...
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    ```

    A barrier_region can be lowered to two barriers, one on the input operands
    and a second one on the results.

    Movtivation and Intended Use Cases:

    The primary way this op is generated is when fusing parallel loops with
    tensor results. This operation helps to make lowerings more progressive
    and flexible.
      - Lowering directly to an alloc + reads and writes breaks the dependency
        chain making transformations like barrier placement and pipelining
        potentially more difficult.
      - Allows the option of non-vector based lowering paths.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs
  );
  let regions = (region SizedRegion<1>:$region);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    (`ins` `(` $inputs^ `:` type($inputs) `)` )?
    $region attr-dict `:` type($results)
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$result_types, "ValueRange":$inputs)>
  ];

  let skipDefaultBuilders = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MultiMmaOp
//===----------------------------------------------------------------------===//

def IREEGPU_MultiMmaOp : Op<IREEGPU_Dialect, "multi_mma", [
    Pure,
    AllTypesMatch<["acc", "result"]>,
    DestinationStyleOpInterface,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface, ["getShapeForUnroll"]>,
    DeclareOpInterfaceMethods<TilingInterface,
       ["getIterationDomain",
        "getLoopIteratorTypes",
        "getTiledImplementation",
        "getResultTilePosition"]>
    ]> {
  let summary = [{Models a contraction of multiple mma operations.}];
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
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
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
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
    }
    %3 = iree_gpu.multi_mma %0, %1, %2 #contraction_trait
      : vector<4xf16>, vector<4xf16> into vector<4xf32>
    ```

    This operation can represent an intrinsic both in subgroup/warp and
    distributed (thread) abstractions through the intrinsic attribute interface.
    It does so semi-opaquely by including optional permutations of each MMA
    fragment with respect to the "canonical" MNK row major matrix multiply.

    The permutations represents the permutation that takes the canonical
    (row major) layout and converts it to the current layout.

    Since the canonical dimensionality of the inner dimensions are somewhat
    intrinsic specific, verification of this op requires only that element
    counts of the inner dimensions match the intrinsic.

    For example, an MMT product of inner dimensions with warp semantics can be
    represented with the following. Permutations are only allowed for ops with
    subgroup semantics and must be resolved before distribution.

    ```mlir
    #contraction_accesses = [
     affine_map<(i, j, k) -> (i, k)>,
     affine_map<(i, j, k) -> (k, j)>,
     affine_map<(i, j, k) -> (i, j)>
    ]
    #contraction_trait = {
      indexing_maps = #contraction_accesses,
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
      rhs_permutation = [1, 0]
    }
    %7 = iree_gpu.multi_mma %4, %5, %6 #contraction_trait
      : tensor<?x?x16x16xf16>, tensor<?x?x16x16xf16> into tensor<?x?x16x16xf32>
    ```

    #### Motivation, Design Choices, and Pitfalls

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
    IREEGPU_AnyMmaAttr:$kind,
    OptionalAttr<DenseI64ArrayAttr>:$lhs_permutation,
    OptionalAttr<DenseI64ArrayAttr>:$rhs_permutation,
    OptionalAttr<DenseI64ArrayAttr>:$acc_permutation
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
      "MmaInterfaceAttr":$intrinsic,
      CArg<"std::optional<DenseI64ArrayAttr>", "std::nullopt">:$lhsPerm,
      CArg<"std::optional<DenseI64ArrayAttr>", "std::nullopt">:$rhsPerm,
      CArg<"std::optional<DenseI64ArrayAttr>", "std::nullopt">:$accPerm)>,
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$acc,
      "ArrayRef<ArrayRef<AffineExpr>>":$indexingExprs,
      "ArrayRef<utils::IteratorType>":$iteratorTypes,
      "MmaInterfaceAttr":$intrinsic,
      CArg<"std::optional<SmallVector<int64_t>>", "std::nullopt">:$lhsPerm,
      CArg<"std::optional<SmallVector<int64_t>>", "std::nullopt">:$rhsPerm,
      CArg<"std::optional<SmallVector<int64_t>>", "std::nullopt">:$accPerm)>,
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$acc,
      "ArrayRef<AffineMap>":$indexingMaps,
      "ArrayRef<utils::IteratorType>":$iteratorTypes,
      "MmaInterfaceAttr":$intrinsic,
      CArg<"std::optional<SmallVector<int64_t>>", "std::nullopt">:$lhsPerm,
      CArg<"std::optional<SmallVector<int64_t>>", "std::nullopt">:$rhsPerm,
      CArg<"std::optional<SmallVector<int64_t>>", "std::nullopt">:$accPerm)>
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

    bool hasTensorSemantics() {
      return isa<RankedTensorType>(getResultType());
    }

    bool hasThreadSemantics();

    llvm::SmallVector<::mlir::AffineMap, 4> getIndexingMapsArray() {
      return llvm::to_vector<4>(getIndexingMaps().getAsValueRange<::mlir::AffineMapAttr>());
    }

    // Returns the bounds of each dimension in the iteration space spanned
    // by the iterator types of this operation.
    void getIterationBounds(SmallVectorImpl<int64_t> &iterationBounds);

    SmallVector<utils::IteratorType> getIteratorTypesArray() {
      auto range =
          getIteratorTypes()
              .template getAsValueRange<IteratorTypeAttr, utils::IteratorType>();
      return {range.begin(), range.end()};
    }

    ArrayRef<int64_t> getOperandInnerShape(uint32_t index) {
      auto opType = ::llvm::cast<::mlir::ShapedType>(getOperandTypes()[index]);
      int64_t innerDimRank = opType.getRank() - getIndexingMapsArray()[index].getNumResults();
      return opType.getShape().take_back(innerDimRank);
    }

    ArrayRef<int64_t> getLhsInnerShape() {
      return getOperandInnerShape(0);
    }

    ArrayRef<int64_t> getRhsInnerShape() {
      return getOperandInnerShape(1);
    }

    ArrayRef<int64_t> getAccInnerShape() {
      return getOperandInnerShape(2);
    }

    int64_t getOperandOuterRank(int64_t index) {
      return getIndexingMapsArray()[index].getNumResults();
    }

    int64_t getLhsOuterRank() {
      return getOperandOuterRank(0);
    }

    int64_t getRhsOuterRank() {
      return getOperandOuterRank(1);
    }

    int64_t getAccOuterRank() {
      return getOperandOuterRank(2);
    }

    // Method to implement for specifying output range for
    // DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() {
      if (hasTensorSemantics()) {
        return getAccMutable();
      }
      // There are no destinations with vector semantics.
      return MutableOperandRange(*this, 0, 0);
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ValueBarrierOp
//===----------------------------------------------------------------------===//

def IREEGPU_ValueBarrierOp : Op<IREEGPU_Dialect, "value_barrier", [
  Pure,
  AllTypesMatch<["inputs", "results"]>]> {
  let summary = [{Synchronizes workers on a value semantic tensor or vector.}];
  let description = [{
    This operation acts as a barrier on a value semantic SSA values (tensor or
    vector). It takes multiple operands and produces a value equivalent to each
    input. This does not have copy and/or data movement semantics and simply
    represents a barrier on all writes in the tensor case, and a barrier until
    all threads acquire the input vector in the vector case.

    The inputs must be either all tensors, or all vectors.

    This operation is a no-op when not present in a parallel context. This
    operation is pure as it only requires synchronization for the value it
    produces.
  }];

  let arguments = (ins  Variadic<AnyRankedTensorOrVector>:$inputs);
  let results   = (outs Variadic<AnyRankedTensorOrVector>:$results);

  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs)
  }];

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs)>
  ];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool hasTensorSemantics() {
      return isa<::mlir::RankedTensorType>(getOperand(0).getType());
    }
    ::mlir::ShapedType getInputType(int operandNum) {
      return ::llvm::cast<::mlir::ShapedType>(
          getInputs()[operandNum].getType());
    }
    SmallVector<::mlir::ShapedType> getInputTypes() {
      return llvm::map_to_vector(
          getInputs(),
          [](Value v) {
            return ::llvm::cast<::mlir::ShapedType>(v.getType());
          });
    }
  }];
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def IREEGPU_YieldOp : Op<IREEGPU_Dialect, "yield", [
    Pure, ReturnLike, Terminator,
    HasParent<"::mlir::iree_compiler::IREE::GPU::BarrierRegionOp">]> {
  let summary = [{Yield values from a iree_gpu region.}];
  let description = [{
     This operation is used to yield values from a within a region.
  }];

  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($values^ `:` type($values))? }];
}

//===----------------------------------------------------------------------===//
//
// AMD Specific Operations
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BufferResourceCastOp
//===----------------------------------------------------------------------===//

def IREEGPU_BufferResourceCastOp : Op<IREEGPU_Dialect, "buffer_resource_cast", [
  Pure,
  AllTypesMatch<["input", "result"]>]> {
  let summary = [{Represents a cast to addr_space<7> (buffer resource) before bufferization.}];
  let description = [{
    Nominal cast of a tensor to AMDGPU buffer resource memory space before
    bufferization. This op takes the parameters with which to perform the cast
    if |input| bufferizes to `storage_buffer` memory space. If |input| resolves
    to any other memory space this op is silently dropped and has no effect.

    If |cache_swizzle_stride| is present, there is verification before
    bufferization that all producers of |input| are view-like and single source
    and user (i.e. trivially no alias). In all other cases this op is best
    effort and has no verification or failure modes.

    // TODO: Add other parameters for casting as needed.
  }];

  let arguments = (ins  AnyRankedTensor:$input,
                        Optional<Index>:$cache_swizzle_stride);
  let results   = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input oilist (`cacheSwizzleStride` `(` $cache_swizzle_stride `)` )
    attr-dict `:` type($result)
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// GlobalLoadDMAOp
//===----------------------------------------------------------------------===//

def IREEGPU_GlobalLoadDMAOp : Op<IREEGPU_Dialect, "global_load_dma", [
    SameVariadicOperandSize]> {
  let summary = "Does a global load DMA operation";
  let description = [{
    This operation represents a subgroup-level global load DMA operation.
    It is used to represent a direct gathering operation from global memory to workgroup.
    To be specific, the thread gathers data from the global memoryspace at the designated
    indices, and stores it to the thread's lane-offset of the workgroup memref at the
    designated indices.

    Specifically, if the thread's subgroup lane id is `lane_id`, the thread will load the data
    from `$source[sourceIndices]` and store it to `$target[targetIndices] + lane_id`.
    Collectively, all threads in the subgroup orchestrate the load DMA operation.

    Note: each gather has a load width is 32bit.
  }];

  let arguments = (ins AnyMemRef:$source, Variadic<Index>:$sourceIndices,
                       AnyMemRef:$target, Variadic<Index>:$targetIndices);
  let results   = (outs);

  let assemblyFormat = [{
    $source`[` $sourceIndices `]` `->` $target `[` $targetIndices `]` attr-dict
      `:` type($source) `->` type($target)
  }];
}

#endif // IREE_CODEGEN_DIALECT_IREEGPUOPS
