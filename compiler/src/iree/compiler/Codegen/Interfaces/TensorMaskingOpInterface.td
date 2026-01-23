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
  ];
}

#endif // IREE_CODEGEN_INTERFACES_TENSOR_MASKING_OP_INTERFACE
