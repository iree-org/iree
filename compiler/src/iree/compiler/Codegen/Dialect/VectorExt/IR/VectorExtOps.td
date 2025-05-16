// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VECTOREXT_OPS
#define IREE_DIALECT_VECTOREXT_OPS

include "mlir/Interfaces/VectorInterfaces.td"
include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtBase.td"
include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtAttrs.td"
include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base class.
//===----------------------------------------------------------------------===//

class IREEVectorExt_PureOp<string mnemonic, list<Trait> traits = []> :
    Op<IREEVectorExt_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Layout ops.
//===----------------------------------------------------------------------===//

def IREEVectorExt_ToLayoutOp : IREEVectorExt_PureOp<"to_layout", [
  Pure,
  AllTypesMatch<["input", "output"]>
  ]> {
  let summary = [{Layout conversion operator.}];
  let description = [{
    The layout conversion operator takes a shaped value and a layout and
    transforms the value to have that layout.

    If the "shared_memory_conversion" attribute is set, then this layout
    change has to be materialized through shared memory.
  }];
  let arguments = (ins
    AnyShaped:$input,
    VectorLayoutInterface:$layout,
    DefaultValuedAttr<UnitAttr, "false">:$shared_memory_conversion,
    // TODO: Solve cmake IREEGPU and VectorExt cyclic dependency to
    // change mma_Kind type to be of MMAInterfaceAttr.
    OptionalAttr<AnyAttr>:$mma_kind
  );
  let results = (outs
    AnyShaped:$output
  );
  let builders = [
    OpBuilder<(ins "Value":$input,
                   "VectorLayoutInterface":$layout,
                   "Attribute":$mma_kind_attr,
                   CArg<"bool", "false">:$shared_memory_conversion), [{
      UnitAttr defaultSharedMemoryConversion;
      if (shared_memory_conversion) {
        defaultSharedMemoryConversion = UnitAttr::get(input.getContext());
      }
      build($_builder, $_state, input.getType(), input, layout, defaultSharedMemoryConversion, mma_kind_attr);
    }]>,
  OpBuilder<(ins "Value":$input,
                "VectorLayoutInterface":$layout), [{
      UnitAttr defaultSharedMemoryConversion;
      Attribute emptyIntrinsic;
      build($_builder, $_state, input.getType(), input, layout, defaultSharedMemoryConversion, emptyIntrinsic);
    }]>,
  ];
  let extraClassDeclaration = [{
    bool hasTensorSemantics() {
      return isa<RankedTensorType>(getOutput().getType());
    }
  }];
  let assemblyFormat = "$input `to` `layout` `(` $layout `)` attr-dict `:` type($input)";
  let hasVerifier = 1;
}

def IREEVectorExt_ToSIMDOp : IREEVectorExt_PureOp<"to_simd",
    [SameOperandsAndResultElementType, Pure]> {
  let summary = [{SIMT to SIMD conversion operation.}];
  let description = [{
    This operation is a temporary operation useful for source/target
    materializations when doing type conversions between distributed and not
    distributed vectors.
  }];
  let arguments = (ins
    AnyVectorOfAnyRank:$input
  );
  let results = (outs
    AnyVectorOfAnyRank:$output
  );
  let extraClassDeclaration = [{}];
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasFolder = 1;
}

def IREEVectorExt_ToSIMTOp : IREEVectorExt_PureOp<"to_simt",
    [SameOperandsAndResultElementType, Pure]> {
  let summary = [{SIMD to SIMT conversion operation.}];
  let description = [{
    This operation is a temporary operation useful for source/target
    materializations when doing type conversions between distributed and not
    distributed vectors.
  }];
  let arguments = (ins
    AnyVectorOfAnyRank:$input
  );
  let results = (outs
    AnyVectorOfAnyRank:$output
  );
  let extraClassDeclaration = [{}];
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasFolder = 1;
}

def IREEVectorExt_TransferGatherOp : IREEVectorExt_PureOp<"transfer_gather", [
    DeclareOpInterfaceMethods<VectorTransferOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<ConditionallySpeculatable>,
    AttrSizedOperandSegments
  ]> {
  let arguments = (ins AnyShaped:$base,
                       Variadic<Index>:$indices,
                       Variadic<VectorOfAnyRankOf<[Index]>>:$index_vecs,
                       BoolArrayAttr:$indexed,
                       AffineMapArrayAttr:$indexed_maps,
                       AffineMapAttr:$permutation_map,
                       AnyType:$padding,
                       Optional<VectorOfNonZeroRankOf<[I1]>>:$mask,
                       BoolArrayAttr:$in_bounds);
  let results = (outs AnyVectorOfAnyRank:$vector);

  let summary = [{Gathers a supervector from memory into an SSA vector value.}];

  let description = [{
    The iree_vector_ext.transfer_gather operation provides a structured
    abstraction for gathers, by preserving the iteration space mapping between
    the result vector and the memory dimensions being indexed.

    The operation is a generalization of `vector.transfer_read` op, where the
    slice from which the read is performed is not guranteed to be contiguous,
    and instead how the slice is gathered is defined explicitly in the
    operation.

    The operation can be thought of as:
      1. A contiguous slice gathered from the base as described by the operation
      2. A `vector.transfer_read` on the contiguous slice

    The operation defines `permutation_map`, `padding`, `mask`, `in_bounds` in
    the same way as `vector.transfer_read` defines, but on the inferred
    contiguous slice.

    The other parameters of the operation define how the contiguous slice is
    gathered from the source. `indices` define a base to offset the source by.
    `indexed` defines for each dimension if the dimension is gathered or
    contiguous.

    The `indices` contains a base to offset the source by. The `indexed` array
    defines if a dimension is gathered or not. For example, for the following
    gather:

    ```
    slice[i, j, k] = base[i + i_offset][j][indices[i][j][k]]
    ```

    The operation would represent this as:

    ```
    indices = %i_offset, 0, 0
    indexed = [False, False, True]
    ```

    For every dimension that is gathered, the operation defines how it is
    gathered. For each gathered dimension, the operation expects a vector of
    indices in `index_vecs` to act as a source of indices for that dimension
    and an AffineMap in `index_maps` describing how this source of indices is
    indexed. For example, for the following gather:

    ```
    slice[i, j, k] = base[i][indices0[i] + offset][indices1[j, k]]
    ```

    The indexing would be described by:

    ```
    indices      = 0, %offset, 0
    indexed      = [False, True, True]
    index_vecs   = %index_vec1, %index_vec2
    index_maps = [
      affine_map<(i, j, k) -> (i),
      affine_map<(i, j, k) -> (j, k)
    ]
    ```

    With these additional parameters, the operation can define a supervector
    read from a non-contiguous slice. For example:

    ```
    base: memref<8192x8x16xf32>
    indices0 : vector<2xindex>
    indices1 : vector<4x8xindex>

    slice[i, j, k] = base[indices0[k]][j][indices1[i, j]]
    vector = read(slice) : memref<8192x8x16xf32> -> vector<16x8x2xf32>
    ```

    Can be represented by:

    ```
    %vector = vector.transfer_gather %base[0, 0, 0](%indices0, %indices1) {
      gather_dims = [0, 2],
      index_maps = [
        affine_map<(i, j, k) -> (k)>,
        affine_map<(i, j, k) -> (i, j)>
      ],
      in_bounds = [true, true, true],
      permutation_map = affine_map<(i, j, k) -> (k, j, i)>
    } : memref<8192x8x16xf32> -> vector<16x8x2xf32>
    ```

    The crucial structure of the operation relies on the index_vec and
    the result vector's indexing being defined based on the dimensions of the
    memory. This mapping can be exploited to simplify gathered dimensions
    to contiguous dimensions.
  }];

  let extraClassDeclaration = [{
    // MaskableOpInterface methods.
    bool supportsPassthru() { return true; }

    SmallVector<AffineMap> getIndexedMapsArray() {
      return llvm::to_vector(getIndexedMaps().getAsValueRange<AffineMapAttr>());
    }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

#endif  // IREE_DIALECT_VECTOREXT_OPS
