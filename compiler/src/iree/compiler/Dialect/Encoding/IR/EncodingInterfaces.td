// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_ENCODING_INTERFACES
#define IREE_DIALECT_ENCODING_INTERFACES

include "iree/compiler/Dialect/Encoding/IR/EncodingBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def IREEEncoding_EncodingLayoutAttrInterface :
  AttrInterface<"EncodingLayoutAttrInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Encoding";
  let description = [{
    Interface used to query layout information needed to materialize encoding
    attributes.

    Any backend can implement the interface to interpret an encoding layout
    based on their needs.

    TBD. The current expectation of the interface is to propagate layout
    information from backends to the host compliation or other targets.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the storage size (in bytes) for the tensor types with an
        optional encoding.
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

#endif // IREE_DIALECT_ENCODING_INTERFACES
