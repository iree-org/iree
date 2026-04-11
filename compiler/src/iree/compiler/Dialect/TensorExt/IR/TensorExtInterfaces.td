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
      /*description =*/"Returns the dimensions that are sparse",
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

  let methods = [
    InterfaceMethod<
      /*description =*/[{
        Generate to a loop structure that iterates over the exact sparse data given a
        range that represents orthonormal hull of the sparse range.

        Given a range, expected to be the orthonormal hull of the actual sparse iteration,
        generate a loop structure that iterates over the exact sparse iteration space. The
        method returns the induction variables of the generated loops. On return the
        insertion point is set to start of the innermost loop upon success.

        - `sparseDims` is the list of sparse dimensions of the result that are expected
          by the caller to lower to loops.
        - `givenRanges` for the `sparseDims` specifies the range of the orthonormal hull
          of the sparse range.
      }],
      /*retTy =*/"::llvm::FailureOr<::llvm::SmallVector<::mlir::Value>>",
      /*methodName =*/"lowerLoopRange",
      /*arguments =*/(ins
          "::mlir::RewriterBase &":$rewriter,
          "::llvm::ArrayRef<int64_t>":$sparseDims,
          "::llvm::ArrayRef<::mlir::Range>":$givenRanges)
    >,
    InterfaceMethod<
      /*description =*/[{
        Return a range that represents an tighter estimate of the number
        of sparse elements in the source, but still using the same rank as the given range.

        Typically, the actual sparse iteration space is such that the bounds of one dimension
        depends on the position of the other dimensions. This methods is meant to get an
        estimate such that the bounds dont depend on position of the other dimensions.

        - `sparseDims` is the list of sparse dimensions of the result that are expected
          by the caller to lower to loops.
        - `givenRanges` for the `sparseDims` specifies the range of the orthonormal hull
          of the sparse range.
      }],
      /*retTy =*/"::llvm::FailureOr<::llvm::SmallVector<::mlir::Range>>",
      /*methodName =*/"getEstimatedLoopRange",
      /*arguments =*/(ins
          "::mlir::RewriterBase &":$rewriter,
          "::llvm::ArrayRef<int64_t>":$sparseDims,
          "::llvm::ArrayRef<::mlir::Range>":$givenRanges)
    >,
    InterfaceMethod<
      /*description =*/[{
      Resolve a range in terms of the source tensor.

      Given a range that represents a slice of a sparse tensor (all dimensions
      of the result), return an equivalent range in terms of the source tensor.

      The inBounds bit vector (one bit per range entry) specifies whether each dimension's
      access is guaranteed to be in-bounds. If false for any dimension, the implementation
      should fail, as out-of-bounds handling is not yet supported.

      The optional paddingValue parameter specifies the value to use for out-of-bounds
      accesses. This is reserved for future use when guard generation is implemented.
      }],
      /*retTy =*/"::llvm::FailureOr<::llvm::SmallVector<::mlir::Range>>",
      /*methodName =*/"resolveRange",
      /*arguments =*/(ins
          "::mlir::RewriterBase &":$rewriter,
          "::llvm::ArrayRef<::mlir::Range>":$givenRanges,
          "const ::llvm::BitVector &":$inBounds,
          "::std::optional<::mlir::Value>":$paddingValue)
    >,
    InterfaceMethod<
      /*description =*/[{
        Returns which sparse dimensions can be distributed across workgroups.

        For sparse dimensions where bounds depend on outer dimensions, those
        dimensions cannot be distributed across workgroups. This method returns
        a BitVector where bit i is set if sparse dimension i (from
        SparseShapeAttrInterface::getSparseDimensions()) can be distributed.
      }],
      /*retTy =*/"::llvm::BitVector",
      /*methodName =*/"getDistributionInfoForSparseDimensions"
    >
  ];
}

#endif // IREE_DIALECT_TENSOR_EXT_INTERFACES
