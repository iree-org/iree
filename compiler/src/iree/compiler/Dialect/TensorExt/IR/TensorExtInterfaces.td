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

  let methods = [
    InterfaceMethod<
      /*description =*/[{Generate to a loop structure that iterates over the exact sparse
      iteration space.

      Given a range, expected to be "bigger" (or a hull) of the actual sparse iteration,
      generate a loop structure that iterates over the exact sparse iteration space. The
      methods returns the induction variables of the generated loops.
      }],
      /*retTy =*/"::llvm::FailureOr<::llvm::SmallVector<::mlir::Value>>",
      /*methodName =*/"lowerLoopRange",
      /*arguments =*/(ins
          "::mlir::RewriterBase &":$rewriter,
          "::llvm::ArrayRef<int64_t>":$sparseDims,
          "::llvm::ArrayRef<::mlir::Range>":$givenRanges)
    >,
    InterfaceMethod<
      /*description =*/[{Return an estimate of the range account for the sparsity.

      Typically, the actual sparse iteration space is such that the bounds of one dimension
      depends on the position of the other dimensions. This methods is meant to get an
      estimate of such that the bounds dont depend on position of the other dimensions.
      }],
      /*retTy =*/"::llvm::FailureOr<::llvm::SmallVector<::mlir::Range>>",
      /*methodName =*/"getEstimatedLoopRange",
      /*arguments =*/(ins
          "::mlir::RewriterBase &":$rewriter,
          "::llvm::ArrayRef<int64_t>":$sparseDims,
          "::llvm::ArrayRef<::mlir::Range>":$givenRanges)
    >
  ];
}

#endif // IREE_DIALECT_TENSOR_EXT_INTERFACES
