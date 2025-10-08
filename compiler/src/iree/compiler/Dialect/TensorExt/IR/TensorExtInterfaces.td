// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_TENSOR_EXT_INTERFACES
#define IREE_DIALECT_TENSOR_EXT_INTERFACES

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

def IREETensorExt_SparseTensorAttrInterface :
  AttrInterface<"SparseTensorAttrInterface"> {

  let cppNamespace = "::mlir::iree_compiler::IREE::TensorExt";

  let description = [{
    Attribute interface to annotate sparse layouts.

    Operations that create tensors with sparse layouts, tensor encodings
    are used to represent that the created tensor has a logical shape + sparse
    layout. The interface here is to allow for different sparse
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

def IREETensorExt_SparseOpInterface :
  OpInterface<"SparseOpInterface"> {

  let cppNamespace = "::mlir::iree_compiler::IREE::TensorExt";

  let description = [{
    Interface for operations that create tensors with sparse layout.

    Operations that implement this interface create a tensor type with a
    given logical shape with tensor encoding which implements the
    `SparseTensorAttrInterface`.
  }];
}

#endif // IREE_DIALECT_TENSOR_EXT_INTERFACES
