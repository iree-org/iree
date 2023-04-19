// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_VMVX_OPS
#define IREE_DIALECT_MODULES_VMVX_OPS

include "iree/compiler/Dialect/VMVX/IR/VMVXBase.td"
include "iree/compiler/Dialect/VMVX/IR/VMVXInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class VMVX_PureOp<string mnemonic, list<Trait> traits = []> :
    VMVX_Op<mnemonic, !listconcat(traits, [Pure])>;

//===----------------------------------------------------------------------===//
// Utility ops
//===----------------------------------------------------------------------===//

def VMVX_GetBufferDescriptorOp : VMVX_PureOp<"get_buffer_descriptor", [
    SameVariadicResultSize
  ]> {
  let summary = "Late binds a base buffer/offset/strides";
  let description = [{
    Queries a base buffer, offset and strides. This op is late bound to its
    source (alloca, binding, etc), allowing additional layers of
    transformations to be added as lowering progresses (or for buffers to be
    combined).

    This op has canonicalization rules which will bubble it up through the
    view stack. A final reconciliation pass is used explicitly to bind it to
    concrete sources.
  }];

  let arguments = (ins
    AnyMemRef:$source
  );
  let results = (outs
    Util_BufferType:$base_buffer,
    Index:$offset,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides
  );

  let assemblyFormat = [{
    $source `:` type($source) `->` type(results) attr-dict
  }];
}

def VMVX_GetRawInterfaceBindingBufferOp : VMVX_PureOp<
    "get_raw_interface_binding_buffer"> {
  let summary = "Gets the raw buffer associated with a binding";
  let description = [{
    Normally, a slice of a binding buffer is returned via
    hal.interface.binding.subspan. However, the normal VMVX lowering flow for
    this presumes that the result is a memref, and upon final conversion, it
    will offset the memref automatically to make it consistent.

    This op is used in situations where earlier in a lowering, we have fully
    resolved the binding to a buffer and would just like the raw backing
    buffer as passed to the interface.
  }];
  let arguments = (ins
    IndexAttr:$set,
    IndexAttr:$binding
  );
  let results = (outs
    Util_BufferType:$buffer
  );
  let assemblyFormat = [{
    `set` `(` $set `)` `binding` `(` $binding `)` attr-dict
  }];
}

def VMVX_QueryTileSizesOp : VMVX_PureOp<"query_tile_sizes", [Pure]> {
  let summary = "Returns tile sizes to use to materialize the given encoding, based on runtime CPU capabilities.";
  let description = [{
    TODO write me
  }];

  let arguments = (ins
    Variadic<VMVX_Index>:$sizes,
    I32Attr:$flags
  );
  let results = (outs
    Variadic<VMVX_Index>:$tileSizes
  );

  let assemblyFormat = [{
    `sizes` `` `(` $sizes `)`
    `flags` `` `(` $flags `)`
    `->` type(results) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// VMVX Ops: ABI
//===----------------------------------------------------------------------===//

def VMVX_BinaryOp : VMVX_Op<"binary", [SameVariadicOperandSize]> {
  let summary = "Performs a strided elementwise operation on two same-rank buffers";
  let description = [{
    Performs the operation in-place as if:
    ```
      OUT = OP(LHS, RHS)
    ```

    Where `OP` is a concrete operation name as defined in ukernel/elementwise.h
  }];
  let arguments = (ins
    // Corresponds to lower-cased opcode suffix of a ukernel binary op.
    StrAttr:$opcode,
    // LHS.
    VMVX_Buffer:$lhs_buffer,
    VMVX_Index:$lhs_offset,
    Variadic<VMVX_Index>:$lhs_strides,
    // RHS.
    VMVX_Buffer:$rhs_buffer,
    VMVX_Index:$rhs_offset,
    Variadic<VMVX_Index>:$rhs_strides,
    // OUT.
    VMVX_Buffer:$out_buffer,
    VMVX_Index:$out_offset,
    Variadic<VMVX_Index>:$out_strides,

    // Dimensions.
    Variadic<VMVX_Index>:$sizes,

    // Attributes.
    VMVX_ElementTypeAttr:$element_type
  );

  let assemblyFormat = [{
    `op` `` `(` $opcode `:` $element_type `)`
    `lhs` `` `(` $lhs_buffer `offset` $lhs_offset `strides` `[` $lhs_strides `]` `:` type($lhs_buffer) `)`
    `rhs` `` `(` $rhs_buffer `offset` $rhs_offset `strides` `[` $rhs_strides `]` `:` type($rhs_buffer) `)`
    `out` `` `(` $out_buffer `offset` $out_offset `strides` `[` $out_strides `]` `:` type($out_buffer) `)`
    `sizes` `` `(` $sizes `)`
    attr-dict
  }];
}

def VMVX_CopyOp : VMVX_Op<"copy", [SameVariadicOperandSize]> {
  let summary = "Copy from one buffer to another";
  let arguments = (ins
    // LHS.
    VMVX_Buffer:$in_buffer,
    VMVX_Index:$in_offset,
    Variadic<VMVX_Index>:$in_strides,
    // OUT.
    VMVX_Buffer:$out_buffer,
    VMVX_Index:$out_offset,
    Variadic<VMVX_Index>:$out_strides,

    // Dimensions.
    Variadic<VMVX_Index>:$sizes,

    // Attributes.
    VMVX_ElementTypeAttr:$element_type
  );
  let assemblyFormat = [{
    `in` `` `(` $in_buffer `offset` $in_offset `strides` `[` $in_strides `]` `:` type($in_buffer) `)`
    `out` `` `(` $out_buffer `offset` $out_offset `strides` `[` $out_strides `]` `:` type($out_buffer) `)`
    `sizes` `` `(` $sizes `)`
    `:` $element_type
    attr-dict
  }];
}

def VMVX_Fill2DOp : VMVX_Op<"fill2d"> {
  let summary = "Fill a tile with a scalar";
  let description = [{
    Fills a tile with dimensions [m, n] with a scalar.
  }];
  let arguments = (ins
    VMVX_ElementType:$scalar,
    VMVX_Buffer:$out_buffer,
    VMVX_Index:$out_offset,
    VMVX_Index:$out_row_stride,

    // Dimensions.
    VMVX_Index:$m,
    VMVX_Index:$n
  );

  let assemblyFormat = [{
    `scalar` `` `(` $scalar `:` type($scalar) `)`
    `out` `` `(` $out_buffer `offset` $out_offset `row_stride` $out_row_stride `:` type($out_buffer) `)`
    `sizes` `` `(` $m `,` $n `)`
    attr-dict
  }];
}

def VMVX_MatmulOp : VMVX_Op<"matmul"> {
  let summary = "Matmul";
  let description = [{
    General matrix-multiply of the form:

      OUT = alpha * (LHS * RHS) + beta * OUT
  }];
  let arguments = (ins
    // Lhs buffer.
    VMVX_Buffer:$lhs_buffer,
    VMVX_Index:$lhs_offset,
    VMVX_Index:$lhs_row_stride,
    // Rhs buffer.
    VMVX_Buffer:$rhs_buffer,
    VMVX_Index:$rhs_offset,
    VMVX_Index:$rhs_row_stride,
    // Out buffer.
    VMVX_Buffer:$out_buffer,
    VMVX_Index:$out_offset,
    VMVX_Index:$out_row_stride,

    // Dimensions.
    VMVX_Index:$m,
    VMVX_Index:$n,
    VMVX_Index:$k,

    // Type and flag attributes.
    VMVX_ElementTypeAttr:$lhs_type,
    VMVX_ElementTypeAttr:$rhs_type,
    VMVX_ElementTypeAttr:$out_type,
    I32Attr:$flags
  );

  let assemblyFormat = [{
    `lhs` `` `(` $lhs_buffer `offset` $lhs_offset `row_stride` $lhs_row_stride `:` type($lhs_buffer) `)`
    `rhs` `` `(` $rhs_buffer `offset` $rhs_offset `row_stride` $rhs_row_stride `:` type($rhs_buffer)`)`
    `out` `` `(` $out_buffer `offset` $out_offset `row_stride` $out_row_stride `:` type($out_buffer) `)`
    `mnk` `` `(` $m `,` $n `,` $k `)`
    `flags` `` `(` $flags `)`
    `:` `(` $lhs_type `,` $rhs_type `,` $out_type `)`
    attr-dict
  }];
}

def VMVX_UnaryOp : VMVX_Op<"unary", [SameVariadicOperandSize]> {
  let summary = "Performs a strided elementwise unary operation";
  let description = [{
    Performs the operation in-place as if:
    ```
      OUT = OP(IN)
    ```

    Where `OP` is a concrete operation name as defined in ukernel/elementwise.h
  }];
  let arguments = (ins
    // Corresponds to lower-cased opcode suffix of a ukernel unary op.
    StrAttr:$opcode,
    // IN.
    VMVX_Buffer:$in_buffer,
    VMVX_Index:$in_offset,
    Variadic<VMVX_Index>:$in_strides,
    // OUT.
    VMVX_Buffer:$out_buffer,
    VMVX_Index:$out_offset,
    Variadic<VMVX_Index>:$out_strides,

    // Dimensions.
    Variadic<VMVX_Index>:$sizes,

    // Attributes.
    VMVX_ElementTypeAttr:$element_type
  );

  let assemblyFormat = [{
    `op` `` `(` $opcode `:` $element_type `)`
    `in` `` `(` $in_buffer `offset` $in_offset `strides` `[` $in_strides `]` `:` type($in_buffer) `)`
    `out` `` `(` $out_buffer `offset` $out_offset `strides` `[` $out_strides `]` `:` type($out_buffer) `)`
    `sizes` `` `(` $sizes `)`
    attr-dict
  }];
}

#endif  // IREE_DIALECT_MODULES_VMVX_OPS
