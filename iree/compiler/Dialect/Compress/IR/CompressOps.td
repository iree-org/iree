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

#ifndef IREE_DIALECT_COMPRESS_OPS
#define IREE_DIALECT_COMPRESS_OPS

include "iree/compiler/Dialect/Compress/IR/CompressBase.td"
include "mlir/Dialect/QuantOps/QuantPredicates.td"

//===----------------------------------------------------------------------===//
// Type definitions.
//===----------------------------------------------------------------------===//

def Compress_AnyTensor : TensorOf<
    [quant_RealPrimitiveType, AnyFloat, AnyInteger, AnyComplex]>;

//===----------------------------------------------------------------------===//
// Quantized Logical Kernel enum
// Quantization policies and backend code lowering matches generic QuantizedOps
// against a fixed set of "quantized logical kernels" defined by this enum.
// If such a kernel was created by outlining/fusing an original source-dialect
// sequence of ops, then the original outlined ops will be in the QuantizedOp
// body, allowing trivial expansion to the original form should the decision
// to quantize be reversed further down the line.
//
// While this set can grow over time, it is heavily biased towards optimization
// oriented dialects such as XLA (as opposed to frontend dialects, which
// often attach different, unrelated semantics to various ops): These should
// largely just be about the math.
//===----------------------------------------------------------------------===//

// A generic op that has no defined semantics. Transformation mechanics must
// be defined by accompanying operand groups.
def Compress_QLK_Generic : StrEnumAttrCase<"GENERIC">;

// Non-broadcasting binary arithmetic ops.
// Such ops will typically have a reasonable default lowering but may be
// implemented in terms of higher priority conversion patterns by backends that
// provide an optimized implementation.
def Compress_QLK_Binary_Add : StrEnumAttrCase<"BINARY_ADD">;
def Compress_QLK_Binary_Div : StrEnumAttrCase<"BINARY_DIV">;
def Compress_QLK_Binary_Mul : StrEnumAttrCase<"BINARY_MUL">;
def Compress_QLK_Binary_Sub : StrEnumAttrCase<"BINARY_SUB">;

def Compress_QLKAttr : StrEnumAttr<"QuantLogicalKernal", "", [
  Compress_QLK_Generic,
  Compress_QLK_Binary_Add,
  Compress_QLK_Binary_Div,
  Compress_QLK_Binary_Mul,
  Compress_QLK_Binary_Sub,
  ]>;


//===----------------------------------------------------------------------===//
// Qunatitzed transform constraint attribute.
// This attribute controls whether it is valid to rewrite an operand group
// to the expressed or storage type (or either if NONE). Typically, ops that
// do not reason about their contents will be set to STORAGE (i.e. "Reverse"),
// whereas arithmetic operations must be set to EXPRESSED (i.e. "Add")
// indicating that it is not valid to simply substitute some expressed type
// (i.e. f32) for a storage type (i.e. i8).
//
// Values:
//   EXPRESSED_OR_STORAGE : The operand group can directly accept either
//       expressed or storage types without constraint.
//   EXPRESSED : The operand group can only directly accept the expressed
//       type that it was originally defined in terms of.
//   PREDICATE : The operand group expects a predicate (I1) type and
//       transformations are free to perform any storage type appropriate
//       cast to generate one.
//===----------------------------------------------------------------------===//

def Compress_QTC_ExpressedOrStorage : StrEnumAttrCase<"EXPRESSED_OR_STORAGE">;
def Compress_QTC_Expressed : StrEnumAttrCase<"EXPRESSED">;
def Compress_QTC_Predicate : StrEnumAttrCase<"PREDICATE">;
def Compress_QTCAttr : StrEnumAttr<"QuantTransformConstraint", "", [
  Compress_QTC_ExpressedOrStorage,
  Compress_QTC_Expressed,
  Compress_QTC_Predicate,
  ]>;

//===----------------------------------------------------------------------===//
// OperandGroup
// Each operand group defines, by index, the constraints placed on sets of
// operand and results for generic logical kernels.
// Each operand and result must ultimately have the same type. Legal mechanisms
// for deriving the type are defined by the transform_constraint.
//===----------------------------------------------------------------------===//

def Compress_QuantOperandGroupAttr :
    StructAttr<"QuantOperandGroup", Compress_Dialect, [
  // Constraint on the type transformation which can be performed on the
  // operands in the quantized region.
  StructFieldAttr<"transform_constraint", Compress_QTCAttr>,

  // Indices of the operands and results that form this group.
  StructFieldAttr<"operand_indices", I32ArrayAttr>,
  StructFieldAttr<"result_indices", I32ArrayAttr>,
]>;

def Compress_QuantOperandGroupArrayAttr :
    TypedArrayAttrBase<Compress_QuantOperandGroupAttr,
                       "Array of operand groups">;

//===----------------------------------------------------------------------===//
// Quantization configuration struct.
//===----------------------------------------------------------------------===//

def Compress_QuantConfigAttr :
    StructAttr<"QuantConfig", Compress_Dialect, [
  StructFieldAttr<"logical_kernel", Compress_QLKAttr>,
  // For GENERIC logical kernels, the operand groups define how the quantizer
  // is permitted to transform the operation.
  StructFieldAttr<"operand_groups", Compress_QuantOperandGroupArrayAttr>,
]>;

//===----------------------------------------------------------------------===//
// QuantRegion op.
//===----------------------------------------------------------------------===//

def Compress_QuantRegionOp : Compress_Op<"quant_region", [
      NoSideEffect,
      IsolatedFromAbove,
      SingleBlockImplicitTerminator<"ReturnOp">]> {
  // TODO(laurenzo): These should probably just be any types with custom
  // validation.
  let arguments = (ins
                   Variadic<AnyType>:$operands,
                   Compress_QuantConfigAttr:$config);
  let results = (outs Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$body);
  let verifier = [{ return verifyQuantRegionOp(*this); }];
}

//===----------------------------------------------------------------------===//
// Structural op definitions.
//===----------------------------------------------------------------------===//

def Compress_ReturnOp : Compress_Op<"return", [Terminator]> {
  let summary = [{
    The `return` operation terminates a region and returns values.
  }];

  let arguments = (ins Variadic<Compress_AnyTensor>:$results);
}

#endif  // IREE_DIALECT_COMPRESS_OPS
