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

def IREEEncoding_EncodingLayoutAttrInterface :
  AttrInterface<"EncodingLayoutAttrInterface"> {
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
        `SerializedEncodingLayoutAttrInterface` interface.
        If it is in common format (e.g., a regular tensor type), we can easily
        calculate the storage size. Otherwise, we will need a hook from
        external, and the hook can come from an attribute that implements the
        interface.
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
    >,
    InterfaceMethod<
      /*desc=*/[{
        Clones an encoding with a new layout list. It is valid to drop any other
        optional parameters used in layout resolving, because they are already
        resolved and being attached to the encoding attribute.
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
    >
  ];
}

def IREEEncoding_SerializedEncodingAttrInterface :
  AttrInterface<"SerializedEncodingLayoutAttrInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Encoding";
  let description = [{
    Interface used to query serialized layout information needed to materialize
    encoding attributes.

    The attributes implementing this interface may be target-specific or general
    enough to be shared across backends, depending on the layouts used.
  }];

  let methods = [
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
