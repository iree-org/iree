// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_ENCODING_ATTRS
#define IREE_DIALECT_ENCODING_ATTRS

include "iree/compiler/Dialect/Encoding/IR/EncodingBase.td"
include "iree/compiler/Dialect/Encoding/IR/EncodingInterfaces.td"
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


def PackedStorageAttr : IREEEncoding_Attr<"PackedStorage"> {
  let mnemonic = "packed_storage";
  let summary = [{Indicates packed storage data type.}];
  let description = [{
    This attribute indicates this is a back-to-back packed storage in memory.
    This attribute takes no arguments.
  }];
  let genVerifyDecl = 0;
}

def EncodingAttr :
    IREEEncoding_Attr<"Encoding", [
      DeclareAttrInterfaceMethods<IREEEncoding_SerializableEncodingAttrInterface, [
        "isSerialized",
        "cloneWithLayouts",
        "calculateStorageSizeInBytes",
      ]>
    ]> {
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
    // TODO(hanchung): Deprecate the round_dims_to field when we plumb the layouts
    // field through the whole stack. See https://github.com/iree-org/iree/issues/17924
    // for details. Note that today we abuse the attribute to carry narrow
    // matrix information. The end goal is deprecating the field and add a
    // "iteration_space_size" field to describe the shape. It is useful to
    // handle narrow matrix cases.
    OptionalParameter<"DenseArrayAttr", "Values for padding M,N,K dimensions">:$round_dims_to,
    OptionalParameter<"ArrayAttr", "An array of attributes that describes the "
    "potential layouts on the device. It is an array because a device could "
    "have several executable targets. Note that it can be any attribute that "
    "implements EncodingLayoutResolverAttrInterface. The expectation of the field "
    "is to bridge the logics between host codes and device codes. If an "
    "attribute does not implement the interface, it could be discarded anytime.">:$layouts
  );

  let builders = [
    AttrBuilder<(ins "int64_t":$operandIndex,
        "EncodingOpType":$opType,
        "ArrayRef<Type>":$elemTypes,
        CArg<"ArrayRef<AffineMap>", "{}">:$maps,
        CArg<"std::optional<AffineMap>", "{}">:$bcastMap,
        CArg<"ArrayRef<int64_t>", "{}">:$roundDimsTo,
        CArg<"ArrayRef<Attribute>", "{}">:$layouts)>
  ];

  let extraClassDeclaration = [{
    /// Returns the bcast_map composed with the user_indexing_map for the
    /// operand_index. The dimensions of the returned map are those of the
    /// data-tiled op's iteration space, and the results of the map are in
    /// the domain of the encoded tensor type.
    AffineMap getMapForOperandIndex() const;

    /// Given the dim position of the encoding `user_indexing_maps`, returns the
    /// matching index of the given encoding's tensor, using getMapForOperandIndex
    /// bcast_map and user_indexing_map.
    std::optional<unsigned> mapDimToOperandIndex(int64_t dimPos) const;

    /// Returns an integer array with values in `round_dims_to`.
    ArrayRef<int64_t> getRoundDimsToArray() const;

    /// Returns a vector with values in `element_types`.
    SmallVector<Type> getElementTypesArray();

    /// Clones an encoding with a new bcast_map
    EncodingAttr clone(AffineMap bcastMap);
  }];

  let genVerifyDecl = 0;
}

//===---------------------------------------------------------------------===//
// encoding.pad_encoding_layout
//===---------------------------------------------------------------------===//

def PadEncodingLayoutAttr : IREEEncoding_Attr<"PadEncodingLayout", [
      DeclareAttrInterfaceMethods<IREEEncoding_SerializableEncodingAttrInterface,
        ["calculateStorageSizeInBytes"]>
    ]> {
  let mnemonic = "pad_encoding_layout";
  let assemblyFormat = "`<` $padding `>`";

  let summary = "An attribute that encodes padding values of tensor dimensions";
  let description = [{
    Associates tensor dimensions with pad values (numbers of appended elements).
    The logical dimensions of the tensors do not change, and the elements in the
    padded regions are left uninitialized.

    This attribute implements `Encoding::SerializedEncodingLayoutResolverAttrInterface`,
    to provide a hook for tensor sizeof lowering. The implementation of this
    interface is backend-agnostic, but the emission of the pad encoding attribute
    itself can be target- or domain-specific.
  }];

  let parameters = (ins
    // How many padding elements to add along each tensor dimension.
    "DenseI32ArrayAttr":$padding
  );
}

//===---------------------------------------------------------------------===//
// encoding.unsupported_encoding
//===---------------------------------------------------------------------===//

def UnsupportedEncodingAttr :
    IREEEncoding_Attr<"UnsupportedEncoding", [
      DeclareAttrInterfaceMethods<IREEEncoding_EncodingLayoutResolverAttrInterface, [
        "cloneWithSimplifiedConfig",
        "getLayout",
      ]>
    ]> {
  let mnemonic = "unsupported_encoding";
  let summary = "A layout resolver attribute that always fails to get layout";
  let description = [{
    An layout resolver that never resolves the layout. I.e., it always fails on
    getting a layout. It can be used in testing or a default unknown attribute
    for encoding.
  }];

  let genVerifyDecl = 0;
}

//===---------------------------------------------------------------------===//
// Encoding specialization attributes, which are mainly for testing purpose.
//===---------------------------------------------------------------------===//

def TestingEncodingAttr :
    IREEEncoding_Attr<"TestingEncoding", [
      DeclareAttrInterfaceMethods<IREEEncoding_SerializableEncodingAttrInterface, [
        "isSerialized",
        "cloneWithLayouts",
      ]>
    ]> {
  let mnemonic = "testing_encoding";
  let summary = "An encoding attribute for testing purpose";

  let description = [{
    An attribute for testing purpose. It is intended to be attached on
    RankedTensorType as an encoding.
  }];

  let parameters = (ins
    OptionalParameter<"ArrayAttr", "An array of attributes that describes the "
    "layouts.">:$layouts
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 0;
}

def UnknownEncodingAttr : IREEEncoding_Attr<"UnknownEncoding"> {
  let mnemonic = "unknown_encoding";
  let summary = "A pure encoding attribute for testing purpose";

  let description = [{
    An attribute for testing purpose. It is intended to be attached on
    RankedTensorType as an encoding.
  }];
  let genVerifyDecl = 0;
}

def UnspecializedEncodingAttr :
    IREEEncoding_Attr<"UnspecializedEncoding", [
      DeclareAttrInterfaceMethods<IREEEncoding_EncodingLayoutResolverAttrInterface, [
        "cloneWithSimplifiedConfig",
      ]>
    ]> {
  let mnemonic = "unspecialized_encoding";

  let summary = "An attribute that indicates the encoding is not yet specialized";

  let description = [{
    This attribute indicates this is an unspecialized encoding. It implements
    very basic interface methods of EncodingLayoutResolverAttrInterface that converts
    it to the SpecializeEncodingAttr during encoding specialization. It is
    mainly for testing purpose as some transformations do not depend on actual
    dialects that implement the attribute interface.
  }];

  let parameters = (ins
    AttrParameter<"IntegerAttr", "The seed that attached on the attribute. "
    "Different seed values indicate different layouts.">:$seed
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 0;
}

def SpecializedEncodingAttr :
    IREEEncoding_Attr<"SpecializedEncoding", [
      DeclareAttrInterfaceMethods<IREEEncoding_EncodingLayoutResolverAttrInterface, [
        "getLayout",
      ]>
    ]> {
  let mnemonic = "specialized_encoding";

  let summary = "An attribute that indicates the encoding is specialized";

  let description = [{
    This attribute is similar to UnspecializedEncodingAttr, but with an optional
    type. The attribute denotes the layout of the type. Different seed values
    indicate different layouts, which can be used to emulate different encoding
    attributes.
  }];

  let parameters = (ins
    AttrParameter<"IntegerAttr", "The seed that attached on the attribute. "
    "Different seed values indicate different layouts.">:$seed,
    AttrParameter<"TypeAttr", "">:$type
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 0;
}

#endif // IREE_DIALECT_ENCODING_ATTRS
