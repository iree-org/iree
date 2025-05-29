// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_IREECODEGENOPS
#define IREE_CODEGEN_DIALECT_IREECODEGENOPS

include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.td"
include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.td"
include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def TensorTypeAttr : TypeAttrBase<"TensorType", "Tensor type attribute">;

def IREECodegen_QueryTileSizesOp :
    Op<IREECodegen_Dialect, "query_tile_sizes", [Pure]> {
  let summary = [{Yields tile sizes for the specified tensor type.}];

  let description = [{
    For targets where tile sizes can't be resolved at compile time, this
    operation allows querying the sizes at runtime. Today this only applies
    to VMVX.
  }];

  let arguments = (ins TensorTypeAttr:$tensor_type);
  let results = (outs Variadic<Index>:$results);
  let assemblyFormat = [{
    attr-dict $tensor_type `->` type($results)
  }];
}

//===----------------------------------------------------------------------===//
// ExtractStridedMetadataOp
//===----------------------------------------------------------------------===//

def IREECodegen_ExtractStridedMetadataOp : Op<IREECodegen_Dialect, "extract_strided_metadata", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    Pure,
    SameVariadicResultSize,
    ViewLikeOpInterface,
    InferTypeOpAdaptor]> {
  let summary = [{Extracts a buffer base with offset and strides.}];
  let description = [{
    This op is implemented similarly to the upstream MemRef::ExtractStridedMetadataOp
    with the following differences.

    1. It does not fold away static offset/stride information.
    Hence unlike the upstream Op the link between the memref and consumers of the
    metadata is not broken when later passes change this information. A common
    example in IREE of this is buffer binding optimizations.

    2. Helper functions getConstifiedMixed{Offset|Strides|Sizes} are not implemented
    as the expectation is you should lower to the upstream op before using those
    functions if you need them.

    Copy of MemRef::ExtractStridedMetadataOp description for reference below.
    Extracts a base buffer, offset and strides. This op allows additional layers
    of transformations and foldings to be added as lowering progresses from
    higher-level dialect to lower-level dialects such as the LLVM dialect.

    The op requires a strided memref source operand. If the source operand is not
    a strided memref, then verification fails.

    This operation is also useful for completeness to the existing memref.dim op.
    While accessing strides, offsets and the base pointer independently is not
    available, this is useful for composing with its natural complement op:
    `memref.reinterpret_cast`.

    Intended Use Cases:

    The main use case is to expose the logic for manipulate memref metadata at a
    higher level than the LLVM dialect.
    This makes lowering more progressive and brings the following benefits:
      - not all users of MLIR want to lower to LLVM and the information to e.g.
        lower to library calls---like libxsmm---or to SPIR-V was not available.
      - foldings and canonicalizations can happen at a higher level in MLIR:
        before this op existed, lowering to LLVM would create large amounts of
        LLVMIR. Even when LLVM does a good job at folding the low-level IR from
        a performance perspective, it is unnecessarily opaque and inefficient to
        send unkempt IR to LLVM.
  }];

  let arguments = (ins
    AnyStridedMemRef:$source
  );
  let results = (outs
    AnyStridedMemRefOfRank<0>:$base_buffer,
    Index:$offset,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides
  );

  let assemblyFormat = [{
    $source `:` type($source) `->` type(results) attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}

//===----------------------------------------------------------------------===//
// SwizzleHintOp
//===----------------------------------------------------------------------===//

def IREECodegen_SwizzleHintOp : Op<IREECodegen_Dialect, "swizzle_hint", [
    SameOperandsAndResultType, Pure]> {
  let summary = [{Hint to swizzle accesses according to an access pattern.}];
  let description = [{
    Optimization hint to swizzle all accesses to the memref this takes a view
    of. This only affects reads/writes immediately consuming this operation and
    is best effort. If the desired swizzling is not apparently possible, this
    op will no-op. As a result, it should not be relied on for correctness.

    Any subviews on this operation will cause swizzling to fail. The expectation
    is for all view like operations to fold into the accessing ops
    (loads/stores) before this op takes effect.

    Note that this only rewrites direct users. If there are any aliased loads
    or stores of the data from/to the |src| memref of a hintOp, those accesses
    will not be swizzled. This allows reusing an allocation with different
    swizzled access patterns as long as there is no data dependency between
    memory with different layouts. For example:

    ```
    %0 = alloc()
    %1 = iree_codegen.swizzle_hint %0, #layout_0
    %2 = iree_codegen.swizzle_hint %0, #layout_1
    {
       vector.store %1
       vector.load %1
         ^
         |
        unrelated
         |
         v
       vector.store %2
       vector.load %2
    }
    ```

    If there is a data dependency between the accesses of %1 and %2, for example
    a value stored to %1 is loaded from %2, this is undefined behavior. Aliasing
    is otherwise perfectly legal.
  }];

  let arguments = (ins MemRefRankOf<[AnyType], [1]>:$operand,
                       IREECodegen_AnySwizzleAttr:$swizzle);
  let results = (outs MemRefRankOf<[AnyType], [1]>:$result);

  let assemblyFormat = [{
    $operand `[` $swizzle attr-dict `]` `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// NullPointerOp
//===----------------------------------------------------------------------===//

def IREECodegen_NullPointerOp :
     Op<IREECodegen_Dialect, "null_pointer", [Pure]> {
  let summary = [{Returns a null_pointer value.}];
  let description = [{
    This is meant to be used only as arguments to microkernels.
  }];
  let results = (outs NullPointer:$result);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// LoadFrom/StoreToBuffer Ops
//===----------------------------------------------------------------------===//

def IREECodegen_LoadFromBufferOp : Op<IREECodegen_Dialect, "load_from_buffer",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>]> {
  let summary = [{Loads a tensor from a memref.}];
  let description = [{
    Loads a tensor from a memref with a compatible shape and the same element
    type.
  }];

  let arguments = (ins
    AnyStridedMemRef:$buffer
  );
  let results = (outs
    AnyRankedTensor:$tensor
  );

  let assemblyFormat = [{
    $buffer attr-dict `:` type($buffer) `->` type($tensor)
  }];

  let hasVerifier = 1;
}

def IREECodegen_StoreToBufferOp : Op<IREECodegen_Dialect, "store_to_buffer",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = [{Stores a tensor into a memref.}];
  let description = [{
    Stores a tensor into a memref with a compatible shape and the same element
    type.
  }];

  let arguments = (ins
    AnyRankedTensor:$tensor,
    AnyStridedMemRef:$buffer
  );
  let results = (outs);

  let assemblyFormat = [{
    $tensor `,` $buffer
    attr-dict `:` type($tensor) `into` type($buffer)
  }];

  let hasVerifier = 1;
}

#endif // IREE_CODEGEN_DIALECT_IREECODEGENOPS
