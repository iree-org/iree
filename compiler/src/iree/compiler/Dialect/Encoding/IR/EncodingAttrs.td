// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_ENCODING_ATTRS
#define IREE_DIALECT_ENCODING_ATTRS

include "iree/compiler/Dialect/Encoding/IR/EncodingBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

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
    OptionalParameter<"ArrayAttr", "Indexing maps of the operation using this tensor">:$user_indexing_maps,
    OptionalParameter<"AffineMapAttr", "Indexing map that represents the broadcasting dims in the producer">:$bcast_map,
    // TODO(hanchung): The round_dims_to parameter can be revisited. We explicitly map them to M,N,K dimension for now.
    OptionalParameter<"DenseArrayAttr", "Values for padding M,N,K dimensions">:$round_dims_to
  );

  let builders = [
    AttrBuilder<(ins "int64_t":$operandIndex,
        "EncodingOpType":$opType,
        "ArrayRef<Type>":$elemTypes,
        CArg<"ArrayRef<AffineMap>", "{}">:$maps,
        CArg<"std::optional<AffineMap>", "{}">:$bcastMap,
        CArg<"ArrayRef<int64_t>", "{}">:$roundDimsTo)>
  ];

  let extraClassDeclaration = [{
    /// Returns the bcast_map composed with the user_indexing_map for the
    /// operand_index. The dimensions of the returned map are those of the
    /// data-tiled op's iteration space, and the results of the map are in
    /// the domain of the encoded tensor type.
    AffineMap getMapForOperandIndex();

    /// Given the dim position of the encoding `user_indexing_maps`, returns the
    /// matching index of the given encoding's tensor, using getMapForOperandIndex
    /// bcast_map and user_indexing_map.
    std::optional<unsigned> mapDimToOperandIndex(int64_t dimPos);

    /// Returns an integer array with values in `round_dims_to`.
    ArrayRef<int64_t> getRoundDimsToArray();

    /// Returns a vector with values in `element_types`.
    SmallVector<Type> getElementTypesArray();

    /// Clones an encoding with a new bcast_map
    EncodingAttr clone(AffineMap bcastMap);
  }];

  let genVerifyDecl = 0;
}

#endif // IREE_DIALECT_ENCODING_ATTRS
