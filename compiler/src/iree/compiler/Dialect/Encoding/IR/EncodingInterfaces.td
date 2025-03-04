// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_ENCODING_INTERFACES
#define IREE_DIALECT_ENCODING_INTERFACES

include "iree/compiler/Dialect/Encoding/IR/EncodingBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// Attribute Interfaces
//===----------------------------------------------------------------------===//

def IREEEncoding_EncodingLayoutResolverAttrInterface :
  AttrInterface<"EncodingLayoutResolverAttrInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Encoding";
  let description = [{
    Interface used to query layout information needed to materialize encoding
    attributes.

    Any backend can implement the interface to interpret an encoding layout
    based on their needs.

    TBD. The current expectation of the interface is to propagate layout
    information from backends to the host compilation or other targets.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the attribute with simplified configuration/layouts. Attribute
        is immutable in MLIR concept. Different attributes can implement
        attribute interface methods differently, and they can carry target
        configuration (e.g., cpu features) for further lowering. However, some
        configuration/parameters can be dropped as long as they are no longer
        needed in the progressively lowering. This method provides a mechanism
        for such attribute to drop the outdated parameters and makes IR dump less
        verbose.
      }],
      /*retTy=*/"::mlir::Attribute",
      /*methodName=*/"cloneWithSimplifiedConfig",
      /*args=*/(ins
        "::mlir::DictionaryAttr":$config
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(false && "unimplemented interface method");
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the an attribute implementing the  which is either common format
        or wrapped by an attribute that implements the
        `SerializableEncodingAttrInterface` interface.
        If it is in common format (e.g., a regular tensor type), we can easily
        calculate the storage size. Otherwise, we will need a hook from
        external, and the hook can come from an attribute that implements the
        interface.
        Returns nullptr if it fails to get the layout.
      }],
      /*retTy=*/"::mlir::Attribute",
      /*methodName=*/"getLayout",
      /*args=*/(ins
        "::mlir::RankedTensorType":$type
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(false && "unimplemented interface method");
        return {};
      }]
    >
  ];
}

def IREEEncoding_SerializableEncodingAttrInterface :
  AttrInterface<"SerializableEncodingAttrInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Encoding";
  let description = [{
    Interface for encoding attributes that can be attached on a tensor type. An
    encoding has two states, serialized and unserialized. An encoding can be set
    with high level encoded information, e.g., operation type, element types of
    operands, etc. During the lowering, the virtual encodings are progressively
    lowered to physical encodings/operations. At some point, an encoding could
    in the middle state, where the final serialized data is encoded in the
    attribute. I.e., the high level information is replaced with serialized
    information, and they are no longer needed. The interface methods are
    intended to help the lowering.

    There are few core methods:

    - `isSerialized`: checks if the encoding is serialized or not.
    - `cloneWithLayouts`: creates a serializable encoding with the layouts
      information.
    - The rest of methods that interpret the encodings. E.g.,
      `calculateStorageSizeInBytes` is the method to parse layouts and produce
      the storage size in bytes.

    The attributes implementing this interface may be target-specific or general
    enough to be shared across backends, depending on the layouts used.

    TBD: The serialization does not need to tie to layouts. It is the only use
         case today, and could be relaxed in the future.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns true iff the attribute already has the serialized information.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isSerialized",
      /*args=*/(ins
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(false && "unimplemented interface method");
        return true;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Creates an encoding with a new layout list. It is valid to drop any
        other optional parameters used in layout resolving, because they are
        already resolved and being attached to the encoding attribute.
      }],
      /*retTy=*/"::mlir::Attribute",
      /*methodName=*/"cloneWithLayouts",
      /*args=*/(ins
        "ArrayRef<::mlir::Attribute>":$layoutAttr
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(false && "unimplemented interface method");
        return {};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the encoding is compatible with `other`. By default,
        they are always compatible if any attribute is not serialized. If they
        are both serialized, the method checks if they are identical or not.
        We could do a less fuzzier match by overriding the implementation.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isCompatibleWith",
      /*args=*/(ins
        "IREE::Encoding::SerializableEncodingAttrInterface":$other
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto attr =
          llvm::dyn_cast_or_null<SerializableEncodingAttrInterface>($_attr);
        if (!attr) {
          return false;
        }
        if (attr.isSerialized() && other.isSerialized()) {
          return $_attr == other;
        }
        return true;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the storage size (in bytes) for the tensor types with an
        optional encoding. Returns Value() if the size is unknown, i.e., it can
        not be inferred with existing information.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"calculateStorageSizeInBytes",
      /*args=*/(ins
        "::mlir::Location":$loc,
        "::mlir::OpBuilder &":$builder,
        "RankedTensorType":$type,
        "ValueRange":$dynamicDims
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(false && "unimplemented interface method");
        return {};
      }]
    >
  ];

  let extraClassDeclaration = [{
    /// Returns true if they are the same attribute. Otherwise, returns true if
    /// both attributes implement SerializableEncodingAttrInterface interface
    /// and they are compatible with each other.
    static bool areCompatible(Attribute lhs, Attribute rhs);
  }];
}

//===----------------------------------------------------------------------===//
// Type Interfaces
//===----------------------------------------------------------------------===//

def IREEEncoding_EncodingTypeInterface :
  TypeInterface<"EncodingTypeInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Encoding";

  let description = [{
    Interface used to access/update tensor types with encodings.
  }];

  let methods = [
    InterfaceMethod<
      [{
        Returns the tensor type with the encoding.
      }],
      /*retTy=*/"::mlir::Type",
      /*methodName=*/"getEncodingType",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      [{
        Returns the same type but with the updated encoding.
      }],
      /*retTy=*/"::mlir::Type",
      /*methodName=*/"updateEncoding",
      /*args=*/(ins
        "::mlir::Attribute":$encoding)
    >,
  ];
}

#endif // IREE_DIALECT_ENCODING_INTERFACES
