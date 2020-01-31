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

//===----------------------------------------------------------------------===//
// VMLA Ops: buffer manipulation
//===----------------------------------------------------------------------===//

def VMLA_BufferConstOp : VMLA_PureOp<"buffer.const"> {
  let arguments = (ins
    VMLA_HostBufferRef:$value
  );
  let results = (outs
    VMLA_BufferRef:$result
  );
}

def VMLA_BufferAllocOp : VMLA_PureOp<"buffer.alloc"> {
  let arguments = (ins
    VMLA_DeviceSize:$byte_length
  );
  let results = (outs
    VMLA_BufferRef:$result
  );
}

def VMLA_BufferCloneOp : VMLA_PureOp<"buffer.clone"> {
  let arguments = (ins
    VMLA_BufferRef:$src
  );
  let results = (outs
    VMLA_BufferRef:$result
  );
}

def VMLA_BufferViewOp : VMLA_PureOp<"buffer.view"> {
  let arguments = (ins
    VMLA_BufferRef:$src,
    VMLA_DeviceSize:$byte_offset,
    VMLA_DeviceSize:$byte_length
  );
  let results = (outs
    VMLA_BufferRef:$result
  );
}

def VMLA_BufferCopyOp : VMLA_Op<"buffer.copy"> {
  let arguments = (ins
    VMLA_BufferRef:$src,
    VMLA_DeviceSize:$src_byte_offset,
    VMLA_BufferRef:$dst,
    VMLA_DeviceSize:$dst_byte_offset,
    VMLA_DeviceSize:$byte_length
  );
}

def VMLA_BufferFillOp : VMLA_Op<"buffer.fill"> {
  let arguments = (ins
    VMLA_BufferRef:$value,
    VMLA_BufferRef:$dst
  );
}

//===----------------------------------------------------------------------===//
// VMLA Ops: comparison
//===----------------------------------------------------------------------===//

def VMLA_CmpOp : VMLA_Op<"cmp"> {
  let arguments = (ins
    VMLA_CmpPredicateAttr:$predicate,
    VMLA_BufferRef:$lhs,
    VMLA_BufferRef:$rhs,
    VMLA_BufferRef:$dst,
    VMLA_AnyTypeAttr:$element_type
  );
}

def VMLA_SelectOp : VMLA_Op<"select"> {
  let arguments = (ins
    VMLA_BufferRef:$cond,
    VMLA_BufferRef:$lhs,
    VMLA_BufferRef:$rhs,
    VMLA_BufferRef:$dst,
    VMLA_AnyTypeAttr:$element_type
  );
}

//===----------------------------------------------------------------------===//
// VMLA Ops: shape/structure
//===----------------------------------------------------------------------===//

def VMLA_TransposeOp : VMLA_Op<"transpose"> {
  let arguments = (ins
    VMLA_BufferRef:$src,
    VMLA_Shape:$src_shape,
    I32ElementsAttr:$dims,
    VMLA_BufferRef:$dst,
    VMLA_AnyTypeAttr:$element_type
  );
}

def VMLA_ReverseOp : VMLA_Op<"reverse"> {
  let arguments = (ins
    VMLA_BufferRef:$src,
    VMLA_Shape:$src_shape,
    I32ElementsAttr:$dims,
    VMLA_BufferRef:$dst,
    VMLA_AnyTypeAttr:$element_type
  );
}

def VMLA_PadOp : VMLA_Op<"pad"> {
  let arguments = (ins
    VMLA_BufferRef:$src,
    VMLA_Shape:$src_shape,
    VMLA_BufferRef:$value,
    VMLA_BufferRef:$dst,
    VMLA_Shape:$dst_shape,
    I32ElementsAttr:$edge_padding_low,
    I32ElementsAttr:$edge_padding_high,
    I32ElementsAttr:$interior_padding,
    VMLA_AnyTypeAttr:$element_type
  );
}

def VMLA_TileOp : VMLA_Op<"tile"> {
  let arguments = (ins
    VMLA_BufferRef:$src,
    VMLA_Shape:$src_shape,
    VMLA_BufferRef:$dst,
    VMLA_Shape:$dst_shape,
    VMLA_AnyTypeAttr:$element_type
  );
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
def VMLA_DivOp : VMLA_BinaryOp<"div", VMLA_AnyTypeAttr>;
def VMLA_RemOp : VMLA_BinaryOp<"rem", VMLA_AnyTypeAttr>;
def VMLA_ExpOp : VMLA_BinaryOp<"exp", VMLA_FloatTypeAttr>;
def VMLA_LogOp : VMLA_BinaryOp<"log", VMLA_FloatTypeAttr>;
def VMLA_RsqrtOp : VMLA_BinaryOp<"rsqrt", VMLA_FloatTypeAttr>;
def VMLA_SqrtOp : VMLA_BinaryOp<"sqrt", VMLA_FloatTypeAttr>;
def VMLA_CosOp : VMLA_BinaryOp<"cos", VMLA_FloatTypeAttr>;
def VMLA_SinOp : VMLA_BinaryOp<"sin", VMLA_FloatTypeAttr>;
def VMLA_TanhOp : VMLA_BinaryOp<"tanh", VMLA_FloatTypeAttr>;
def VMLA_Atan2Op : VMLA_BinaryOp<"atan2", VMLA_FloatTypeAttr>;

def VMLA_MinOp : VMLA_BinaryOp<"min", VMLA_AnyTypeAttr>;
def VMLA_MaxOp : VMLA_BinaryOp<"max", VMLA_AnyTypeAttr>;
def VMLA_FloorOp : VMLA_UnaryOp<"floor", VMLA_FloatTypeAttr>;
def VMLA_CeilOp : VMLA_UnaryOp<"ceil", VMLA_FloatTypeAttr>;

//===----------------------------------------------------------------------===//
// VMLA Ops: conversion
//===----------------------------------------------------------------------===//

def VMLA_ConvertOp : VMLA_Op<"convert"> {
  let arguments = (ins
    VMLA_BufferRef:$src,
    VMLA_BufferRef:$dst,
    VMLA_AnyTypeAttr:$src_type,
    VMLA_AnyTypeAttr:$dst_type
  );
}

//===----------------------------------------------------------------------===//
// VMLA Ops: GEMM/GEMV
//===----------------------------------------------------------------------===//

def VMLA_MatMulOp : VMLA_Op<"matmul"> {
  let arguments = (ins
    VMLA_BufferRef:$lhs,
    VMLA_Shape:$lhs_shape,
    VMLA_BufferRef:$rhs,
    VMLA_Shape:$rhs_shape,
    VMLA_BufferRef:$dst,
    VMLA_Shape:$dst_shape,
    VMLA_FloatTypeAttr:$lhs_type,
    VMLA_FloatTypeAttr:$rhs_type,
    VMLA_FloatTypeAttr:$dst_type
  );
}

//===----------------------------------------------------------------------===//
// VMLA Ops: reduction
//===----------------------------------------------------------------------===//

class VMLA_ReduceOp<string mnemonic, list<OpTrait> traits = []> :
    VMLA_Op<mnemonic, traits> {
  let arguments = (ins
    VMLA_BufferRef:$src,
    VMLA_Shape:$src_shape,
    VMLA_BufferRef:$init,
    I32Attr:$dimension,
    VMLA_BufferRef:$dst,
    VMLA_Shape:$dst_shape,
    VMLA_AnyTypeAttr:$element_type
  );
}

def VMLA_ReduceSumOp : VMLA_ReduceOp<"reduce.sum">;
def VMLA_ReduceMinOp : VMLA_ReduceOp<"reduce.min">;
def VMLA_ReduceMaxOp : VMLA_ReduceOp<"reduce.max">;

#endif  // IREE_DIALECT_VMLA_OPS
