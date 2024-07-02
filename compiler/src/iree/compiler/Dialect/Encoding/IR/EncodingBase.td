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
    AttrParameter<"ArrayAttr", "element types of the user's operands">:$element_types,
    OptionalParameter<"TypeAttr", "type of the original tensor type before padding">:$original_type,
    // TODO(#15466): generalize matmul_narrow_{M,N} into a list?
    OptionalParameter<"IntegerAttr", "optional M narrow dimension size (only for contraction op user_indexing_maps)">:$matmul_narrow_M,
    OptionalParameter<"IntegerAttr", "optional N narrow dimension size (only for contraction op user_indexing_maps)">:$matmul_narrow_N,
    OptionalParameter<"ArrayAttr", "Indexing maps of the operation using this tensor">:$user_indexing_maps,
    OptionalParameter<"AffineMapAttr", "Indexing map that represents the broadcasting dims in the producer">:$bcast_map,
    // TODO(hanchung): The round_dims_to parameter can be revisited. We explicitly map them to M,N,K dimension for now.
    OptionalParameter<"DenseArrayAttr", "Values for padding M,N,K dimensions">:$round_dims_to
  );

  let builders = [
    AttrBuilder<(ins "int64_t":$operandIndex,
        "ArrayRef<Type>":$elemTypes, "Type":$origType,
        CArg<"std::optional<int64_t>", "{}">:$matmulNarrowM,
        CArg<"std::optional<int64_t>", "{}">:$matmulNarrowN,
        CArg<"ArrayRef<AffineMap>", "{}">:$maps,
        CArg<"AffineMap", "{}">:$bcastMap,
        CArg<"ArrayRef<int64_t>", "{}">:$roundDimsTo)>
  ];

  let extraClassDeclaration = [{
    /// Returns the indexing map used by the operand index in the encoding.
    AffineMap getMapForOperandIndex();

    /// Given the dim position of the encoding `user_indexing_maps`, returns the
    /// matching index of the given encoding's tensor.
    unsigned mapDimToOperandIndex(int64_t dimPos);

    /// Returns an integer array with values in `round_dims_to`.
    ArrayRef<int64_t> getRoundDimsToArray();

    /// Clones an encoding with a new bcast_map
    EncodingAttr clone(AffineMap bcastMap);
  }];

  let genVerifyDecl = 0;
}

#endif // IREE_DIALECT_ENCODING_BASE
