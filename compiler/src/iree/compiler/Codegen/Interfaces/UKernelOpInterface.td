// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_INTERFACES_UKERNEL_OP_INTERFACE
#define IREE_CODEGEN_INTERFACES_UKERNEL_OP_INTERFACE

include "mlir/IR/OpBase.td"

def UKernelOpInterface : OpInterface<"UKernelOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";

  let description = [{
    An interface for ops that wrap a call to microkernels.
  }];
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Method to lower the Micro Kernel operation into a function call.
      }],
      /*retType=*/"FailureOr<mlir::CallOpInterface>",
      /*methodName=*/"lowerToFunctionCall",
      /*args=*/(ins "RewriterBase &":$rewriter),
      /*methodBody=*/"",
      /*defautImplementation=*/"return failure();"
    >,
  ];
}

#endif // IREE_CODEGEN_INTERFACES_UKERNEL_OP_INTERFACE
