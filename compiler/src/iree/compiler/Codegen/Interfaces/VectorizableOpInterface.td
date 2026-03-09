// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_INTERFACES_VECTORIZABLE_OP_INTERFACE
#define IREE_CODEGEN_INTERFACES_VECTORIZABLE_OP_INTERFACE

include "mlir/IR/OpBase.td"

def VectorizableOpInterface : OpInterface<"VectorizableOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler";

  let description = [{
    Interface for ops that can be vectorized, i.e., converted from
    tensor/memref operations to vector operations.

    The driver is responsible for:
    - Computing vector sizes (e.g., from lowering_config attributes).
    - Calling isVectorizable() before vectorize().
    - Replacing the original op with the returned values.

    Implementation-specific options are passed via DictionaryAttr.
    Each op type documents which options it supports.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Checks if the op can be vectorized with the given parameters.
        Must not create any operations or modify IR.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isVectorizable",
      /*args=*/(ins
        "::llvm::ArrayRef<int64_t>":$vectorSizes,
        "::llvm::ArrayRef<bool>":$scalableDims,
        "::mlir::DictionaryAttr":$options)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Vectorizes the op and returns replacement values.
        The caller is responsible for replacing/erasing the original op
        using the returned values.

        This should only be called after isVectorizable() returns true.
        isVectorizable() is the complete precondition check — if it
        returns true, vectorize() is expected to succeed. The return
        type is FailureOr because implementations may wrap upstream
        functions (e.g., linalg::vectorize()) that return FailureOr,
        and we cannot guarantee their preconditions are exhaustive.
        A failure after isVectorizable() returns true indicates a bug
        in the precondition check.
      }],
      /*retTy=*/"::mlir::FailureOr<::llvm::SmallVector<::mlir::Value>>",
      /*methodName=*/"vectorize",
      /*args=*/(ins
        "::mlir::RewriterBase &":$rewriter,
        "::llvm::ArrayRef<int64_t>":$vectorSizes,
        "::llvm::ArrayRef<bool>":$scalableDims,
        "::mlir::DictionaryAttr":$options)
    >,
  ];

  let extraClassDeclaration = [{
    /// Convenience overloads that default options to nullptr.
    bool isVectorizable(::llvm::ArrayRef<int64_t> vectorSizes,
                        ::llvm::ArrayRef<bool> scalableDims) {
      return isVectorizable(vectorSizes, scalableDims, {});
    }
    ::mlir::FailureOr<::llvm::SmallVector<::mlir::Value>>
    vectorize(::mlir::RewriterBase &rewriter,
              ::llvm::ArrayRef<int64_t> vectorSizes,
              ::llvm::ArrayRef<bool> scalableDims) {
      return vectorize(rewriter, vectorSizes, scalableDims, {});
    }
  }];
}

#endif // IREE_CODEGEN_INTERFACES_VECTORIZABLE_OP_INTERFACE
