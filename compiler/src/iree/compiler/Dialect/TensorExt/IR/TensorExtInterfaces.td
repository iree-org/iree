// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_TENSOR_EXT_INTERFACES
#define IREE_DIALECT_TENSOR_EXT_INTERFACES

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

def IREETensorExt_SparseShapeAttrInterface :
  AttrInterface<"SparseShapeAttrInterface"> {

  let cppNamespace = "::mlir::iree_compiler::IREE::TensorExt";

  let description = [{
    Attribute interface to annotate sparse layouts.

    For operations that cast a source tensor/memref type to a sparse layout, these
    attributes are used to represent that the created shaped type has a logical shape
    + sparse layout. The interface here is to allow for different sparse
    representations that are opaque to the transformations.
  }];

  let methods = [
    InterfaceMethod<
      /*description =*/"Return the dimensions that are sparse",
      /*retTy=*/"::llvm::SmallVector<int64_t>",
      /*methodName=*/"getSparseDimensions"
    >
  ];
}

def IREETensorExt_SparseCastOpInterface :
  OpInterface<"SparseCastOpInterface"> {

  let cppNamespace = "::mlir::iree_compiler::IREE::TensorExt";

  let description = [{
    Interface for operations that create shaped types with sparse layout.

    Operations that implement this interface cast a source operand of tensor/memref type
    to a shaped type with a logical shape with an sparse layout. The sparsity information
    is represented using a `SparseShapeAttrInterface` on the type. For `tensor` type this
    is recorded as an `EncodingAttr`. For `memref` type this is represented using the
    `MemrefLayoutAttrInterface`.
  }];

  let verify = [{
   return ::mlir::iree_compiler::IREE::TensorExt::verifySparseCastOpInterface(
     ::mlir::cast<::mlir::iree_compiler::IREE::TensorExt::SparseCastOpInterface>($_op));
  }];
}

#endif // IREE_DIALECT_TENSOR_EXT_INTERFACES
