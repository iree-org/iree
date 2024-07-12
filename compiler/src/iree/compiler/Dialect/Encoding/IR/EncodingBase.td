// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_ENCODING_BASE
#define IREE_DIALECT_ENCODING_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def IREEEncoding_Dialect : Dialect {
  let name = "iree_encoding";
  let cppNamespace = "::mlir::iree_compiler::IREE::Encoding";
  let summary = [{
    Tensor encoding attributes and ops.
  }];
  let description = [{
    A dialect defining IREE tensor encoding attributes and related ops, used to
    implement data-tiling.
  }];
  let useDefaultAttributePrinterParser = 1;
}

//===---------------------------------------------------------------------===//
// Data layout encoding attributes
//===---------------------------------------------------------------------===//

class IREEEncoding_Attr<string name, list<Trait> traits = []>
  : AttrDef<IREEEncoding_Dialect, name, traits>;

class IREEEncoding_I32EnumAttr<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Encoding";
  let genSpecializedAttr = 0;
}

class IREEEncoding_EnumAttr<EnumAttrInfo enumInfo, string name = "">
  : EnumAttr<IREEEncoding_Dialect, enumInfo, name>;

// Enums for tagging operand operation in an EncodingAttr
def MATMUL : I32EnumAttrCase<"matmul", 0>;
def CONV   : I32EnumAttrCase<"conv", 1>;

def EncodingOpType : IREEEncoding_I32EnumAttr<"EncodingOpType",
    "Tracks the type of operation of the operand.", [
      MATMUL,
      CONV,
    ]>;

def EncodingOpTypeAttr:
  IREEEncoding_EnumAttr<EncodingOpType, "optype">;

def EncodingAttr :
    IREEEncoding_Attr<"Encoding"> {
  let mnemonic = "encoding";
  let summary = [{information to decide how to data-tile a tensor}];
   let description = [{
    This attribute describes the change in the layout for
    a given tensor to execute subsequent operations on
    the tiled layout. The encoding serves as a way to
    represent the change in the way the data is laid out in
    memory without changing the logical rank/extent of
    the tensor itself. When required, the encoding
    can be used to explicitly manifest the layout change
    through operations like pack/unpack.
  }];

  let assemblyFormat = "`<` struct(params) `>`";

  let parameters = (ins
    AttrParameter<"IntegerAttr", "this tensor operand's index in the parameter list">:$operand_index,
    AttrParameter<"EncodingOpTypeAttr", "operand type">:$op_type,
    AttrParameter<"ArrayAttr", "element types of the user's operands">:$element_types,
    OptionalParameter<"TypeAttr", "type of the original tensor type before padding">:$original_type,
    // TODO(#15466): generalize matmul_narrow_{M,N} into a list?
    OptionalParameter<"ArrayAttr", "Indexing maps of the operation using this tensor">:$user_indexing_maps,
    OptionalParameter<"DenseArrayAttr", "Values for padding M,N,K dimensions">:$max_paddings
  );

  let builders = [
    AttrBuilder<(ins "int64_t":$operandIndex,
        "EncodingOpType":$opType,
        "ArrayRef<Type>":$elemTypes, "Type":$origType,
        CArg<"ArrayRef<AffineMap>", "{}">:$maps,
        CArg<"ArrayRef<int64_t>", "{}">:$maxPaddings)>
  ];

  let extraClassDeclaration = [{
    /// Returns the indexing map used by the operand index in the encoding.
    AffineMap getMapForOperandIndex();

    /// Given the dim position of the encoding `user_indexing_maps`, returns the
    /// matching index of the given encoding's tensor.
    unsigned mapDimToOperandIndex(int64_t dimPos);

    /// Returns an integer array with values in `max_paddings`.
    ArrayRef<int64_t> getMaxPaddingsArray();
  
    /// optionally return narrow M if possible
    std::optional<int64_t> getNarrowM();

    // optionally return narrow N if possible
    std::optional<int64_t> getNarrowN();
  }];

  let genVerifyDecl = 0;
}

#endif // IREE_DIALECT_ENCODING_BASE
