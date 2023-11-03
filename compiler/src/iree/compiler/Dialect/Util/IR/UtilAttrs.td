// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_UTIL_IR_UTIL_ATTRS
#define IREE_DIALECT_UTIL_IR_UTIL_ATTRS

include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// #util.byte_pattern
//===----------------------------------------------------------------------===//

def Util_BytePatternAttr : AttrDef<Util_Dialect, "BytePattern", [
  TypedAttrInterface,
  DeclareAttrInterfaceMethods<Util_SizedStorageAttr, [
    "getStorageSize",
  ]>,
  DeclareAttrInterfaceMethods<Util_SerializableAttrInterface, [
    "serializeToBuffer",
    "serializeToStream",
  ]>,
]> {
  let mnemonic = "byte_pattern";
  let summary = [{an attribute containing a filled byte pattern}];
  let description = [{
    A dense serializable attribute with the given byte pattern.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"">:$type,
    AttrParameter<"int64_t", "">:$pattern
  );

  let assemblyFormat = [{
    `<` $pattern `>`
  }];
}

//===----------------------------------------------------------------------===//
// #util.byte_range
//===----------------------------------------------------------------------===//

def Util_ByteRangeAttr : AttrDef<Util_Dialect, "ByteRange", []> {
  let mnemonic = "byte_range";
  let summary = [{defines a range of bytes}];
  let description = [{
    Specifies a starting offset and total length in bytes.
  }];

  let parameters = (ins
    AttrParameter<"int64_t", "">:$offset,
    AttrParameter<"int64_t", "">:$length
  );

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// #util.composite
//===----------------------------------------------------------------------===//

def Util_CompositeAttr : AttrDef<Util_Dialect, "Composite", [
  DeclareAttrInterfaceMethods<Util_SizedStorageAttr, [
    "getStorageSize",
  ]>,
  DeclareAttrInterfaceMethods<Util_SerializableAttrInterface, [
    "serializeToBuffer",
    "serializeToStream",
  ]>,
]> {
  let mnemonic = "composite";
  let summary = [{an attribute composed of a sequence of attributes}];
  let description = [{
    Models a concatenated set of serializable attributes that when combined
    form a single sequence of i8 elements. As each value references the uniqued
    storage of the composite element this attribute is cheap to construct. When
    the full flattened range is required it can be efficiently streamed via the
    SerializableAttrInterface. All values must also be serializable.

    All values are tightly packed to byte boundaries. If padding is required it
    can be inserted as splat elements attributes with the padding value
    (usually 0). Sub-byte aligned element types will have their individual
    components padded to byte alignment.
  }];

  let parameters = (ins
    AttrParameter<"int64_t", "">:$totalLength,
    AttrParameter<"ArrayAttr", "">:$values
  );

  let builders = [
    AttrBuilder<(ins
      "ArrayRef<Attribute>":$values
    )>,
  ];

  let extraClassDeclaration = [{
    int64_t getNumElements() const { return getTotalLength(); }
  }];

  let genVerifyDecl = 1;

  let hasCustomAssemblyFormat = 1;
}

#endif  // IREE_DIALECT_UTIL_IR_UTIL_ATTRS
