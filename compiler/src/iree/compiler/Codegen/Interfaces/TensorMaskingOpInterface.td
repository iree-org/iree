// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_INTERFACES_TENSOR_MASKING_OP_INTERFACE
#define IREE_CODEGEN_INTERFACES_TENSOR_MASKING_OP_INTERFACE

include "mlir/IR/OpBase.td"

def TensorMaskingOpInterface : OpInterface<"TensorMaskingOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler";

  let description = [{}];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{}],
      /*retTy=*/"FailureOr<SmallVector<Value>>",
      /*methodName=*/"getMaskedImplementation",
      /*args=*/(ins
        "OpBuilder &":$builder,
        "ArrayRef<OpFoldResult>":$padMultiples)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Mask this operation as a producer of an input to another masked
        operation. Returns the replacement for the result with resultNumber
        padded to the given padSizes.
      }],
      /*retTy=*/"FailureOr<Value>",
      /*methodName=*/"maskAsProducer",
      /*args=*/(ins
        "OpBuilder &":$builder,
        "unsigned":$resultNumber,
        "ArrayRef<OpFoldResult>":$padSizes)
    >,
  ];
}

#endif // IREE_CODEGEN_INTERFACES_TENSOR_MASKING_OP_INTERFACE
