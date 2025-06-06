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

//===---------------------------------------------------------------------===//
// iree_encoding.packed_storage
//===---------------------------------------------------------------------===//

def LayoutAttr :
    IREEEncoding_Attr<"Layout", [
      DeclareAttrInterfaceMethods<IREEEncoding_SerializableAttr, [
        "isSerialized",
        "isIdentityLayout",
        "calculateStorageSizeInBytes",
      ]>
    ]> {
  let mnemonic = "layout";
  let assemblyFormat = "`<` $layouts `>`";

  let summary = [{Indicates potential serialized layouts.}];
  let description = [{
    This attribute describes the potential layouts of the encoding. It is an
    array because a device could have multiple target layouts.

    It has a verifier that will ensure that all the attributes in layouts
    implement SerializableAttr.
  }];

  let parameters = (ins
    AttrParameter<"ArrayAttr", "Serialized layouts">:$layouts
  );

  // Implement custom builders to bypass the verification assertion in Base::get
  // method, i.e., `assert(succeeded(ConcreteT::verifyInvariants(...)))`.
  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins "ArrayAttr":$layouts)>,
  ];

  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// iree_encoding.packed_storage
//===---------------------------------------------------------------------===//

def PackedStorageAttr : IREEEncoding_Attr<"PackedStorage"> {
  let mnemonic = "packed_storage";
  let summary = [{Indicates packed storage data type.}];
  let description = [{
    This attribute indicates this is a back-to-back packed storage in memory.
    This attribute takes no arguments.
  }];
  let genVerifyDecl = 0;
}

//===---------------------------------------------------------------------===//
// iree_encoding.encoding
//===---------------------------------------------------------------------===//

def EncodingAttr :
    IREEEncoding_Attr<"Encoding", [
      DeclareAttrInterfaceMethods<IREEEncoding_SerializableAttr, [
        "isSerialized",
        "cloneWithLayouts",
      ]>,
      DeclareAttrInterfaceMethods<IREEEncoding_ContractionEncodingAttrInterface, [
        "getReductionDims",
      ]>
    ]> {
  let mnemonic = "encoding";
  let summary = [{Information to decide how to data-tile a tensor.}];
  let description = [{
    This attribute describes the change in the layout for
    a given tensor to execute subsequent operations on
    the tiled layout. The encoding serves as a way to
    represent the change in the way the data is laid out in
    memory without changing the logical rank/extent of
    the tensor itself. When required, the encoding
    can be used to explicitly manifest the layout change
    through operations like pack/unpack.

    This attribute has a verifier that will ensure that:
    - The `operand_index` shouldn't exceed the size of
      `user_indexing_maps`, if the latter is provided.
    - The maps in `user_indexing_maps` should be composable.
  }];

  let assemblyFormat = [{
    `<`
    struct($operand_index, $op_type, $element_types, $user_indexing_maps,
           custom<DynamicI64ArrayAttr>($iteration_sizes))
    `>`
  }];

  let parameters = (ins
    AttrParameter<"IntegerAttr", "this tensor operand's index in the parameter list">:$operand_index,
    AttrParameter<"EncodingOpTypeAttr", "operand type">:$op_type,
    AttrParameter<"ArrayAttr", "element types of the user's operands">:$element_types,
    OptionalParameter<"ArrayAttr", "Indexing maps of the operation using this tensor">:$user_indexing_maps,
    OptionalParameter<"ArrayAttr", "The original operation's iteration sizes. "
    "This is useful for handling skinny dimension sizes">:$iteration_sizes
  );

  let builders = [
    AttrBuilder<(ins "int64_t":$operandIndex,
        "EncodingOpType":$opType,
        "ArrayRef<Type>":$elemTypes,
        CArg<"ArrayRef<AffineMap>", "{}">:$maps,
        CArg<"ArrayRef<int64_t>", "{}">:$iterationSizes)>
  ];

  let extraClassDeclaration = [{
    /// Returns the composed `user_indexing_map` for the operand_index. The
    /// dimensions of the returned map are those of the data-tiled op's
    /// iteration space, and the results of the map are in the domain of
    /// the encoded tensor type.
    AffineMap getMapForOperandIndex() const;

    /// Returns the root/first map for every entry in `user_indexing_maps`. This
    /// is useful for deriving layout information of the originally encoded
    /// operation.
    SmallVector<AffineMap> getRootMaps() const;

    /// Returns the last map in the `user_indexing_maps` for the operand_index.
    /// This is useful for checking compatibility of new maps being added with
    /// the existing `user_indexing_maps`.
    AffineMap getLastMapForOperandIndex() const;

    /// Given the dim position of the encoding `user_indexing_maps`, returns the
    /// matching index of the given encoding's tensor, using getMapForOperandIndex.
    std::optional<unsigned> mapDimToOperandIndex(int64_t dimPos) const;

    /// Returns an integer array with values in `iteration_sizes`.
    SmallVector<int64_t> getIterationSizesArray() const;

    /// Returns a vector with values in `element_types`.
    SmallVector<Type> getElementTypesArray();

    /// Clones an encoding with a new indexing map being appended to the
    /// `operand_index` entry of `user_indexing_map`.
    EncodingAttr cloneWithNewOperandIndexingMap(AffineMap newIndexingMap);
  }];

  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// iree_encoding.matmul_k
//===---------------------------------------------------------------------===//

def MatmulKAttr : IREEEncoding_Attr<"MatmulK", [
      DeclareAttrInterfaceMethods<IREEEncoding_SerializableAttr, [
        "isSerialized",
        "cloneWithLayouts",
      ]>,
      DeclareAttrInterfaceMethods<IREEEncoding_ContractionEncodingAttrInterface, [
        "getReductionDims",
      ]>,
    ]> {
  let mnemonic = "matmul_k";
  let summary = [{An attribute that tracks reduction dimensions for matmul.}];
  let description = [{
    The attribute is specialized for tracking the matmul reduction dimensions
    only. The encoded dimensions are the reduction dimensions that will be
    further used in a matmul.

    Example: linalg.matmul ins(%lhs, %rhs) outs(%acc), where the indexing maps
             are:

      lhs: (M, N, K) -> (M, K)
      rhs: (M, N, K) -> (K, N)
      acc: (M, N, K) -> (M, N)

    The encoding for the lhs is iree_encoding.matmul_k<k_dims = [1]>.
    The encoding for the rhs is iree_encoding.matmul_k<k_dims = [0]>.
    The encoding for the acc is iree_encoding.matmul_k<k_dims = []>.

    Example: linalg.matmul_transpose_b ins(%lhs, %rhs) outs(%acc), where the
             indexing maps are:

      lhs: (M, N, K) -> (M, K)
      rhs: (M, N, K) -> (N, K)
      acc: (M, N, K) -> (M, N)

    The encoding for the lhs is iree_encoding.matmul_k<k_dims = [1]>.
    The encoding for the rhs is iree_encoding.matmul_k<k_dims = [1]>.
    The encoding for the acc is iree_encoding.matmul_k<k_dims = []>.

    There is no difference between static cases and dynamic cases because the
    dimension sizes are not encoded.

    Any encoding propagation that transforms reduction dimensions could result
    in undefined behavior, because it does not encode the transformations. The
    information is missing in this context.
  }];

  let assemblyFormat = "`<` struct(params) `>`";

  let parameters = (ins
    "DenseI32ArrayAttr":$k_dims
  );

  let builders = [
    AttrBuilder<(ins "ArrayRef<int32_t>":$k_dims)>
  ];
}

//===---------------------------------------------------------------------===//
// iree_encoding.pad_encoding_layout
//===---------------------------------------------------------------------===//

def PadEncodingLayoutAttr : IREEEncoding_Attr<"PadEncodingLayout", [
      DeclareAttrInterfaceMethods<IREEEncoding_SerializableAttr, [
        "calculateStorageSizeInBytes",
        "cloneWithLayouts",
        "isIdentityLayout",
        "isSerialized"
        ]>
    ]> {
  let mnemonic = "pad_encoding_layout";
  let assemblyFormat = "`<` custom<Padding>($padding) `>`";

  let summary = [{An attribute that encodes padding values of tensor dimensions.}];
  let description = [{
    Associates tensor dimensions with pad values (numbers of appended elements).
    The logical dimensions of the tensors do not change, and the elements in the
    padded regions are left uninitialized.

    It is expected that the number of pad values specified is same as the rank of
    the associated tensor.
    - If the value is `0` no padding is done along that dimension
    - If the value is positive static value, the tensor dimension is padded to that value
    - If the value is `?`, the tensor dimension is padded to a value determined by the
      backend, usually for better cache behavior.
    Negative values for padding is illegal.

    This attribute implements `Encoding::SerializedLayoutResolverAttr`,
    to provide a hook for tensor sizeof lowering. The implementation of this
    interface is backend-agnostic, but the emission of the pad encoding attribute
    itself can be target- or domain-specific.
  }];

  let parameters = (ins
    // How many padding elements to add along each tensor dimension.
    "DenseI64ArrayAttr":$padding
  );

  let builders = [
    AttrBuilder<(ins "ArrayRef<int64_t>":$padding)>
  ];

  let extraClassDeclaration = [{
    /// Returns a PadEncodingLayoutAttr attribute that pads zero elements.
    static PadEncodingLayoutAttr getIdentityAttr(MLIRContext *ctx, int rank);
  }];

  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// iree_encoding.identity_encoding
//===---------------------------------------------------------------------===//

def IdentityEncodingAttr :
    IREEEncoding_Attr<"IdentityEncoding", [
      DeclareAttrInterfaceMethods<IREEEncoding_LayoutResolverAttr, [
        "cloneWithSimplifiedConfig",
        "getLayout",
      ]>
    ]> {
  let mnemonic = "identity_encoding";
  let summary = [{A layout resolver that behaves like an identity function.}];
  let description = [{
    A layout resolver that behaves like an identity function. I.e., it resolves
    the layout to the same layout without encodings.
  }];

  let genVerifyDecl = 0;
}

//===---------------------------------------------------------------------===//
// iree_encoding.unsupported_encoding
//===---------------------------------------------------------------------===//

def UnsupportedEncodingAttr :
    IREEEncoding_Attr<"UnsupportedEncoding", [
      DeclareAttrInterfaceMethods<IREEEncoding_LayoutResolverAttr, [
        "cloneWithSimplifiedConfig",
        "getLayout",
      ]>
    ]> {
  let mnemonic = "unsupported_encoding";
  let summary = [{A layout resolver attribute that always fails to get layout.}];
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
      DeclareAttrInterfaceMethods<IREEEncoding_SerializableAttr, [
        "isSerialized",
        "cloneWithLayouts",
      ]>
    ]> {
  let mnemonic = "testing_encoding";
  let summary = [{An encoding attribute for testing purposes.}];

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
  let summary = [{A pure encoding attribute for testing purpose.}];

  let description = [{
    An attribute for testing purpose. It is intended to be attached on
    RankedTensorType as an encoding.
  }];
  let genVerifyDecl = 0;
}

def UnspecializedEncodingAttr :
    IREEEncoding_Attr<"UnspecializedEncoding", [
      DeclareAttrInterfaceMethods<IREEEncoding_LayoutResolverAttr, [
        "cloneWithSimplifiedConfig",
        "getLayout",
      ]>
    ]> {
  let mnemonic = "unspecialized_encoding";
  let assemblyFormat = "`<` $seed `>`";

  let summary = [{An attribute that indicates the encoding is not yet specialized.}];

  let description = [{
    This attribute indicates this is an unspecialized encoding. It implements
    very basic interface methods of LayoutResolverAttr that
    returns the SpecializedEncodingAttr with the same seed as serialized layout
    in encoding specialization. Different seed values indicate different layouts.

    It is mainly for testing purpose as some transformations do not depend on
    actual dialects that implement the attribute interface.
  }];

  let parameters = (ins "int32_t":$seed);
}

def SpecializedEncodingAttr :
    IREEEncoding_Attr<"SpecializedEncoding", [
      IREEEncoding_SerializableAttr
    ]> {
  let mnemonic = "specialized_encoding";
  let assemblyFormat = "`<` $seed (`,` $type^)? `>`";

  let summary = [{An attribute that indicates the encoding is specialized.}];

  let description = [{
    This attribute is similar to UnspecializedEncodingAttr, but with an optional
    type. The attribute denotes the layout of the type.

    Different seed values indicate different layouts, which can be used to
    emulate different encoding attributes.
  }];

  let parameters = (ins
    "int32_t":$seed,
    OptionalParameter<"TypeAttr", "">:$type
  );
}

#endif // IREE_DIALECT_ENCODING_ATTRS
