// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_DIALECT_VMLA_OPS
#define IREE_DIALECT_VMLA_OPS

include "iree/compiler/Dialect/VMLA/IR/VMLABase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VMLA_PureOp<string mnemonic, list<OpTrait> traits = []> :
    VMLA_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

//===----------------------------------------------------------------------===//
// VMLA Ops: pseudo ops
//===----------------------------------------------------------------------===//

def VMLA_ConstantOp : VMLA_PureOp<"constant"> {
  let summary = [{constant buffer declaration}];
  let description = [{
    A pseudo-op used to represent a buffer with constant contents. This is later
    expanded into VM ops and the vmla.buffer.const op.
  }];

  let arguments = (ins
    ElementsAttr:$value
  );
  let results = (outs
    VMLA_Buffer:$result
  );

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, ElementsAttr value",
    [{
      build(builder, result, IREE::VMLA::BufferType::get(builder.getContext()),
            value);
    }]>,
  ];

  let assemblyFormat = "attr-dict $value `->` type($result)";
}

//===----------------------------------------------------------------------===//
// VMLA Ops: buffer manipulation
//===----------------------------------------------------------------------===//

def VMLA_BufferConstOp : VMLA_PureOp<"buffer.const"> {
  let arguments = (ins
    VMLA_HostBuffer:$value
  );
  let results = (outs
    VMLA_Buffer:$result
  );

  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def VMLA_BufferAllocOp : VMLA_Op<"buffer.alloc"> {
  let arguments = (ins
    VMLA_DeviceSize:$byte_length
  );
  let results = (outs
    VMLA_Buffer:$result
  );

  let assemblyFormat = [{
    `byte_length` `=` $byte_length attr-dict `:` type($result)
  }];
}

def VMLA_BufferCloneOp : VMLA_Op<"buffer.clone"> {
  let arguments = (ins
    VMLA_Buffer:$src
  );
  let results = (outs
    VMLA_Buffer:$result
  );

  let assemblyFormat = "$src attr-dict `:` type($result)";
}

def VMLA_BufferByteLengthOp : VMLA_PureOp<"buffer.byte_length"> {
  let arguments = (ins
    VMLA_Buffer:$value
  );
  let results = (outs
    VMLA_DeviceSize:$result
  );

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def VMLA_BufferViewOp : VMLA_PureOp<"buffer.view"> {
  let arguments = (ins
    VMLA_Buffer:$src,
    VMLA_DeviceSize:$byte_offset,
    VMLA_DeviceSize:$byte_length
  );
  let results = (outs
    VMLA_Buffer:$result
  );

  let assemblyFormat = [{
    $src`[`$byte_offset`]``,` `byte_length` `=` $byte_length
    attr-dict `:` type($result)
  }];
}

def VMLA_BufferCopyOp : VMLA_Op<"buffer.copy"> {
  let arguments = (ins
    VMLA_Buffer:$src,
    VMLA_DeviceSize:$src_byte_offset,
    VMLA_Buffer:$dst,
    VMLA_DeviceSize:$dst_byte_offset,
    VMLA_DeviceSize:$byte_length
  );

  let assemblyFormat = [{
    $src`[`$src_byte_offset`]``,`
    `out` $dst`[`$dst_byte_offset`]``,` `byte_length` `=` $byte_length
    attr-dict
  }];
}

def VMLA_BufferFillOp : VMLA_Op<"buffer.fill"> {
  let arguments = (ins
    VMLA_Buffer:$value,
    VMLA_Buffer:$dst
  );

  let assemblyFormat = "$value`,` `out` $dst attr-dict";
}

def VMLA_BufferLoadI32Op : VMLA_PureOp<"buffer.load.i32"> {
  let arguments = (ins
    VMLA_Buffer:$src,
    VMLA_DeviceSize:$byte_offset
  );
  let results = (outs
    I32:$result
  );

  let assemblyFormat = "$src`[`$byte_offset`]` attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// VMLA Ops: comparison
//===----------------------------------------------------------------------===//

def VMLA_CmpOp : VMLA_ElementTypeOp<"cmp"> {
  let arguments = (ins
    VMLA_CmpPredicateAttr:$predicate,
    VMLA_Buffer:$lhs,
    VMLA_Buffer:$rhs,
    VMLA_Buffer:$dst,
    VMLA_AnyTypeAttr:$element_type
  );

  let assemblyFormat = [{
    $predicate`,` $lhs`,` $rhs`,` `out` $dst attr-dict `:` $element_type
  }];
}

def VMLA_SelectOp : VMLA_ElementTypeOp<"select"> {
  let arguments = (ins
    VMLA_Buffer:$cond,
    VMLA_Buffer:$lhs,
    VMLA_Buffer:$rhs,
    VMLA_Buffer:$dst,
    VMLA_AnyTypeAttr:$element_type
  );

  let assemblyFormat = [{
    $cond`,` $lhs`,` $rhs`,` `out` $dst attr-dict `:` $element_type
  }];
}

//===----------------------------------------------------------------------===//
// VMLA Ops: shape/structure
//===----------------------------------------------------------------------===//

def VMLA_CopyOp : VMLA_ElementTypeOp<"copy", [
    VMLA_IncludeShapes,
    SameVariadicOperandSize,
  ]> {
  let arguments = (ins
    VMLA_Buffer:$src,
    VMLA_Shape:$src_shape,
    Variadic<VMLA_Index>:$src_indices,
    VMLA_Buffer:$dst,
    VMLA_Shape:$dst_shape,
    Variadic<VMLA_Index>:$dst_indices,
    Variadic<VMLA_Index>:$lengths,
    VMLA_AnyTypeAttr:$element_type
  );

  let assemblyFormat = [{
    $src`(`$src_shape `:` type($src_shape)`)``,`
    (`src_indices` `=` `[` $src_indices^ `]``,`)?
    `out` $dst`(`$dst_shape `:` type($dst_shape)`)`
    (`,` `dst_indices` `=` `[` $dst_indices^ `]`)?
    (`,` `lengths` `=` `[` $lengths^ `]`)? attr-dict `:` $element_type
  }];
}

def VMLA_TransposeOp : VMLA_ElementTypeOp<"transpose", [VMLA_IncludeShapes]> {
  let arguments = (ins
    VMLA_Buffer:$src,
    VMLA_Shape:$src_shape,
    ElementsAttr:$permutation,
    VMLA_Buffer:$dst,
    VMLA_Shape:$dst_shape,
    VMLA_AnyTypeAttr:$element_type
  );

  let assemblyFormat = [{
    $src`(`$src_shape `:` type($src_shape)`)``,`
    `out` $dst`(`$dst_shape `:` type($dst_shape)`)` attr-dict `:` $element_type
  }];
}

def VMLA_ReverseOp : VMLA_ElementTypeOp<"reverse", [VMLA_IncludeShapes]> {
  let arguments = (ins
    VMLA_Buffer:$src,
    VMLA_Shape:$src_shape,
    ElementsAttr:$dimensions,
    VMLA_Buffer:$dst,
    VMLA_Shape:$dst_shape,
    VMLA_AnyTypeAttr:$element_type
  );

  let assemblyFormat = [{
    $src`(`$src_shape `:` type($src_shape)`)``,`
    `out` $dst`(`$dst_shape `:` type($dst_shape)`)` attr-dict `:` $element_type
  }];
}

def VMLA_PadOp : VMLA_ElementTypeOp<"pad", [VMLA_IncludeShapes]> {
  let arguments = (ins
    VMLA_Buffer:$src,
    VMLA_Shape:$src_shape,
    VMLA_Buffer:$value,
    VMLA_Shape:$value_shape,
    VMLA_Buffer:$dst,
    VMLA_Shape:$dst_shape,
    ElementsAttr:$edge_padding_low,
    ElementsAttr:$edge_padding_high,
    ElementsAttr:$interior_padding,
    VMLA_AnyTypeAttr:$element_type
  );

  let assemblyFormat = [{
    $src`(`$src_shape `:` type($src_shape)`)``,`
    $value`(`$value_shape `:` type($value_shape)`)``,`
    `out` $dst`(`$dst_shape `:` type($dst_shape)`)` attr-dict `:` $element_type
  }];
}

def VMLA_BroadcastOp : VMLA_ElementTypeOp<"broadcast", [VMLA_IncludeShapes]> {
  let arguments = (ins
    VMLA_Buffer:$src,
    VMLA_Shape:$src_shape,
    VMLA_Buffer:$dst,
    VMLA_Shape:$dst_shape,
    VMLA_AnyTypeAttr:$element_type
  );

  let assemblyFormat = [{
    $src`(`$src_shape `:` type($src_shape)`)``,`
    `out` $dst`(`$dst_shape `:` type($dst_shape)`)` attr-dict `:` $element_type
  }];
}

def VMLA_TileOp : VMLA_ElementTypeOp<"tile", [VMLA_IncludeShapes]> {
  let arguments = (ins
    VMLA_Buffer:$src,
    VMLA_Shape:$src_shape,
    VMLA_Buffer:$dst,
    VMLA_Shape:$dst_shape,
    VMLA_AnyTypeAttr:$element_type
  );

  let assemblyFormat = [{
    $src`(`$src_shape `:` type($src_shape)`)``,`
    `out` $dst`(`$dst_shape `:` type($dst_shape)`)` attr-dict `:` $element_type
  }];
}

def VMLA_GatherOp : VMLA_ElementTypeOp<"gather", [VMLA_IncludeShapes]> {
  let arguments = (ins
    VMLA_Buffer:$src,
    VMLA_Shape:$src_shape,
    VMLA_Buffer:$indices,
    VMLA_Shape:$indices_shape,
    VMLA_Buffer:$dst,
    VMLA_Shape:$dst_shape,
    I64Attr:$dim,
    I64Attr:$batch_dims,
    VMLA_AnyTypeAttr:$element_type
  );

  let assemblyFormat = [{
    $src`(`$src_shape `:` type($src_shape)`)``,`
    $indices`(`$indices_shape `:` type($indices_shape)`)``,`
    `out` $dst`(`$dst_shape `:` type($dst_shape)`)` attr-dict `:` $element_type
  }];
}

def VMLA_ScatterOp : VMLA_ElementTypeOp<"scatter", [VMLA_IncludeShapes]> {
  let arguments = (ins
    VMLA_Buffer:$src,
    VMLA_Shape:$src_shape,
    VMLA_Buffer:$indices,
    VMLA_Shape:$indices_shape,
    VMLA_Buffer:$dst,
    VMLA_Shape:$dst_shape,
    VMLA_AnyTypeAttr:$element_type
  );

  let assemblyFormat = [{
    $src`(`$src_shape `:` type($src_shape)`)``,`
    $indices`(`$indices_shape `:` type($indices_shape)`)``,`
    `out` $dst`(`$dst_shape `:` type($dst_shape)`)` attr-dict `:` $element_type
  }];
}

//===----------------------------------------------------------------------===//
// VMLA Ops: bit manipulation
//===----------------------------------------------------------------------===//

def VMLA_NotOp : VMLA_UnaryOp<"not", VMLA_AnyTypeAttr>;
def VMLA_AndOp : VMLA_BinaryOp<"and", VMLA_AnyTypeAttr>;
def VMLA_OrOp : VMLA_BinaryOp<"or", VMLA_AnyTypeAttr>;
def VMLA_XorOp : VMLA_BinaryOp<"xor", VMLA_AnyTypeAttr>;
def VMLA_ShlOp : VMLA_BinaryOp<"shl", VMLA_AnyTypeAttr>;
def VMLA_ShrOp : VMLA_BinaryOp<"shr", VMLA_AnyTypeAttr>;

//===----------------------------------------------------------------------===//
// VMLA Ops: arithmetic
//===----------------------------------------------------------------------===//

def VMLA_AddOp : VMLA_BinaryOp<"add", VMLA_AnyTypeAttr>;
def VMLA_SubOp : VMLA_BinaryOp<"sub", VMLA_AnyTypeAttr>;
def VMLA_AbsOp : VMLA_UnaryOp<"abs", VMLA_AnyTypeAttr>;
def VMLA_NegOp : VMLA_UnaryOp<"neg", VMLA_AnyTypeAttr>;
def VMLA_MulOp : VMLA_BinaryOp<"mul", VMLA_AnyTypeAttr>;
def VMLA_DivOp : VMLA_BinaryOp<"div", VMLA_AnyTypeAttr>;
def VMLA_RemOp : VMLA_BinaryOp<"rem", VMLA_AnyTypeAttr>;
def VMLA_PowOp : VMLA_BinaryOp<"pow", VMLA_FloatTypeAttr>;
def VMLA_ExpOp : VMLA_UnaryOp<"exp", VMLA_FloatTypeAttr>;
def VMLA_LogOp : VMLA_UnaryOp<"log", VMLA_FloatTypeAttr>;
def VMLA_RsqrtOp : VMLA_UnaryOp<"rsqrt", VMLA_FloatTypeAttr>;
def VMLA_SqrtOp : VMLA_UnaryOp<"sqrt", VMLA_FloatTypeAttr>;
def VMLA_CosOp : VMLA_UnaryOp<"cos", VMLA_FloatTypeAttr>;
def VMLA_SinOp : VMLA_UnaryOp<"sin", VMLA_FloatTypeAttr>;
def VMLA_TanhOp : VMLA_UnaryOp<"tanh", VMLA_FloatTypeAttr>;
def VMLA_Atan2Op : VMLA_BinaryOp<"atan2", VMLA_FloatTypeAttr>;

def VMLA_MinOp : VMLA_BinaryOp<"min", VMLA_AnyTypeAttr>;
def VMLA_MaxOp : VMLA_BinaryOp<"max", VMLA_AnyTypeAttr>;
def VMLA_ClampOp : VMLA_TernaryOp<"clamp", VMLA_AnyTypeAttr>;
def VMLA_FloorOp : VMLA_UnaryOp<"floor", VMLA_FloatTypeAttr>;
def VMLA_CeilOp : VMLA_UnaryOp<"ceil", VMLA_FloatTypeAttr>;

//===----------------------------------------------------------------------===//
// VMLA Ops: conversion
//===----------------------------------------------------------------------===//

def VMLA_ConvertOp : VMLA_Op<"convert", [VMLA_OpInterface]> {
  let arguments = (ins
    VMLA_Buffer:$src,
    VMLA_Buffer:$dst,
    VMLA_AnyTypeAttr:$src_type,
    VMLA_AnyTypeAttr:$dst_type
  );

  let extraClassDeclaration = [{
    static void extractTypeAttributes(OperationState &state, ArrayRef<Type> operandTypes, ArrayRef<Type> resultTypes) {
      state.addAttribute("src_type", TypeAttr::get(operandTypes[0]));
      state.addAttribute("dst_type", TypeAttr::get(resultTypes[0]));
    }
  }];

  let assemblyFormat = [{
    $src`,` `out` $dst attr-dict `:` $src_type `->` $dst_type
  }];
}

//===----------------------------------------------------------------------===//
// VMLA Ops: Convultion
//===----------------------------------------------------------------------===//

def VLMA_ConvOp : VMLA_Op<"conv", [VMLA_IncludeShapes]> {
  let arguments = (ins
    VMLA_Buffer:$input,
    VMLA_Shape:$input_shape,
    VMLA_Buffer:$filter,
    VMLA_Shape:$filter_shape,
    VMLA_Buffer:$dst,
    VMLA_Shape:$dst_shape,
    I32ElementsAttr:$window_strides,
    I32ElementsAttr:$padding,
    I32ElementsAttr:$lhs_dilation,
    I32ElementsAttr:$rhs_dilation,
    I32Attr:$feature_group_count,
    I32Attr:$batch_group_count,
    VMLA_FloatTypeAttr:$input_type,
    VMLA_FloatTypeAttr:$filter_type,
    VMLA_FloatTypeAttr:$dst_type
  );

  let extraClassDeclaration = [{
    static void extractTypeAttributes(OperationState &state, ArrayRef<Type> operandTypes, ArrayRef<Type> resultTypes) {
      state.addAttribute("input_type", TypeAttr::get(operandTypes[0]));
      state.addAttribute("filter_type", TypeAttr::get(operandTypes[1]));
      state.addAttribute("dst_type", TypeAttr::get(resultTypes[0]));
    }
  }];

  let assemblyFormat = [{
    $input`(`$input_shape `:` type($input_shape)`)` `:` $input_type`,`
    $filter`(`$filter_shape `:` type($filter_shape)`)` `:` $filter_type`,`
    `out` $dst`(`$dst_shape `:` type($dst_shape)`)` `:` $dst_type attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// VMLA Ops: GEMM/GEMV
//===----------------------------------------------------------------------===//

def VMLA_BatchMatMulPseudoOp : VMLA_Op<"batch.matmul.pseudo"> {
  let summary = "Tensor-level pseudo-op of VMLA::BatchMatMulOp.";
  let description = [{
    This is a tensor-level version of VMLA::BatchMatMulOp, to facilitate
    the lowering process.

    All operands are rank-3 with the following dimension structure:
    - lhs = [B, FLHS, C]
    - rhs = [B, FRHS, C]
    - dst = [B, FRHS, FLHS]
    Where:
    - B = batch dimension
    - C = contracting dimension
    - FLHS and FRHS are the free dimensions of each operand

    To put this in terms closer to the mathematics of matrix multiplication,
    if we ignore the leading B dimension and focus on what is mathematically an
    MxKxN matmul, then this corresponds to:
    - lhs = [M, K] = [LHSROWS, K]
    - rhs = [N, K] = [RHSCOLS, K]
    - dst = [N, M] = [RHSCOLS, LHSROWS]
    Note that dst is transposed from what one would expect.
    This is due to an implementation detail of this op in the runtime.
    This op is backed by an invocation of the Ruy matrix multiplication library,
    which prefers its matrices in this layout (in matrix terminology:
    lhs = row-major, rhs = column-major, dst = column-major).
    We insert the relevant transposes as needed in the compiler.
  }];
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  let results = (outs
    AnyTensor:$dst
  );

  let assemblyFormat = [{
    $lhs`,` $rhs attr-dict `:`
    `(`type($lhs)`,` type($rhs)`)` `->` type($dst)
  }];
}

def VMLA_BatchMatMulOp : VMLA_Op<"batch.matmul", [VMLA_OpInterface, VMLA_IncludeShapes]> {
  let arguments = (ins
    VMLA_Buffer:$lhs,
    VMLA_Shape:$lhs_shape,
    VMLA_Buffer:$rhs,
    VMLA_Shape:$rhs_shape,
    VMLA_Buffer:$dst,
    VMLA_Shape:$dst_shape,
    VMLA_FloatTypeAttr:$lhs_type,
    VMLA_FloatTypeAttr:$rhs_type,
    VMLA_FloatTypeAttr:$dst_type
  );

  let extraClassDeclaration = [{
    static void extractTypeAttributes(OperationState &state, ArrayRef<Type> operandTypes, ArrayRef<Type> resultTypes) {
      state.addAttribute("lhs_type", TypeAttr::get(operandTypes[0]));
      state.addAttribute("rhs_type", TypeAttr::get(operandTypes[1]));
      state.addAttribute("dst_type", TypeAttr::get(resultTypes[0]));
    }
  }];

  let assemblyFormat = [{
    $lhs`(`$lhs_shape `:` type($lhs_shape)`)` `:` $lhs_type`,`
    $rhs`(`$rhs_shape `:` type($rhs_shape)`)` `:` $rhs_type`,`
    `out` $dst`(`$dst_shape `:` type($dst_shape)`)` `:` $dst_type attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// VMLA Ops: reduction
//===----------------------------------------------------------------------===//

class VMLA_ReduceOp<string mnemonic, list<OpTrait> traits = []> :
    VMLA_ElementTypeOp<mnemonic, !listconcat(traits, [VMLA_IncludeShapes])> {
  let arguments = (ins
    VMLA_Buffer:$src,
    VMLA_Shape:$src_shape,
    VMLA_Buffer:$init,
    VMLA_Shape:$init_shape,
    I32Attr:$dimension,
    VMLA_Buffer:$dst,
    VMLA_Shape:$dst_shape,
    VMLA_AnyTypeAttr:$element_type
  );

  let assemblyFormat = [{
    $src`(`$src_shape `:` type($src_shape)`)``,`
    $init`(`$init_shape `:` type($init_shape)`)``,`
    `out` $dst`(`$dst_shape `:` type($dst_shape)`)` attr-dict `:` $element_type
  }];
}

def VMLA_ReduceSumOp : VMLA_ReduceOp<"reduce.sum">;
def VMLA_ReduceMinOp : VMLA_ReduceOp<"reduce.min">;
def VMLA_ReduceMaxOp : VMLA_ReduceOp<"reduce.max">;

class VMLA_PoolingOp<string mnemonic, list<OpTrait> traits = []> :
    VMLA_ElementTypeOp<mnemonic, !listconcat(traits, [VMLA_IncludeShapes])> {
  let arguments = (ins
    VMLA_Buffer:$src,
    VMLA_Shape:$src_shape,
    VMLA_Buffer:$init,
    VMLA_Shape:$init_shape,
    VMLA_Buffer:$dst,
    VMLA_Shape:$dst_shape,
    VMLA_AnyTypeAttr:$element_type,
    I32ElementsAttr:$window_dimensions,
    I32ElementsAttr:$window_strides,
    I32ElementsAttr:$padding
  );

  let assemblyFormat = [{
    $src`(`$src_shape `:` type($src_shape)`)``,`
    $init`(`$init_shape `:` type($init_shape)`)``,`
    `out` $dst`(`$dst_shape `:` type($dst_shape)`)` attr-dict `:` $element_type
  }];
}

def VMLA_PoolingSumOp : VMLA_PoolingOp<"pooling.sum">;
def VMLA_PoolingMinOp : VMLA_PoolingOp<"pooling.min">;
def VMLA_PoolingMaxOp : VMLA_PoolingOp<"pooling.max">;

//===----------------------------------------------------------------------===//
// VMLA Ops: ABI
//===----------------------------------------------------------------------===//

def VMLA_InterfaceConstOp :
    VMLA_PureOp<"interface.const", [VMLA_OpInterface]> {
  let arguments = (ins
    VMLA_Interface:$interface,
    IREE_IndexAttr:$offset
  );
  let results = (outs
    AnyTypeOf<[I32, VMLA_Index]>:$result
  );

  let assemblyFormat = [{
    $interface attr-dict `:` type($result)
  }];
}

def VMLA_InterfaceBindingOp :
    VMLA_PureOp<"interface.binding", [VMLA_OpInterface]> {
  let arguments = (ins
    VMLA_Interface:$interface,
    I32Attr:$set,
    I32Attr:$binding
  );
  let results = (outs
    VMLA_Buffer:$result
  );

  let assemblyFormat = [{
    $interface attr-dict `:` type($result)
  }];
}

#endif  // IREE_DIALECT_VMLA_OPS
