// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_VMVX_OPS
#define IREE_DIALECT_MODULES_VMVX_OPS

include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXBase.td"
include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VMVX_PureOp<string mnemonic, list<Trait> traits = []> :
    VMVX_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

//===----------------------------------------------------------------------===//
// VMVX Ops: ABI
//===----------------------------------------------------------------------===//

def VMVX_AddOp : VMVX_Op<"add", [SameVariadicOperandSize]> {
  let summary = "Performs a strided elementwise add of two same-rank buffers";
  let description = [{
    Performs addition in-place as if:
      OUT = LHS + RHS

    All operands have the same rank.
  }];
  let arguments = (ins
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
    Variadic<VMVX_Index>:$sizes
  );

  let assemblyFormat = [{
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
    Variadic<VMVX_Index>:$sizes
  );
  let assemblyFormat = [{
    `in` `` `(` $in_buffer `offset` $in_offset `strides` `[` $in_strides `]` `:` type($in_buffer) `)`
    `out` `` `(` $out_buffer `offset` $out_offset `strides` `[` $out_strides `]` `:` type($out_buffer) `)`
    `sizes` `` `(` $sizes `)`
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

    // Scale factors.
    VMVX_ElementType:$alpha,
    VMVX_ElementType:$beta,

    // Execution flags.
    I32Attr:$flags
  );

  let assemblyFormat = [{
    `lhs` `` `(` $lhs_buffer `offset` $lhs_offset `row_stride` $lhs_row_stride `:` type($lhs_buffer) `)`
    `rhs` `` `(` $rhs_buffer `offset` $rhs_offset `row_stride` $rhs_row_stride `:` type($rhs_buffer)`)`
    `out` `` `(` $out_buffer `offset` $out_offset `row_stride` $out_row_stride `:` type($out_buffer) `)`
    `mnk` `` `(` $m `,` $n `,` $k `)`
    `scale` `` `(` $alpha `:` type($alpha) `,` $beta `:` type($beta) `)`
    `flags` `` `(` $flags `)`
    attr-dict
  }];
}

#endif  // IREE_DIALECT_MODULES_VMVX_OPS
