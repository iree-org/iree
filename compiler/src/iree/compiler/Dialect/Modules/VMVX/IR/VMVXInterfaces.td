// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_MODULES_VMVX_INTERFACES
#define IREE_DIALECT_MODULES_VMVX_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREE::VMVX::VMVXOpInterface
//===----------------------------------------------------------------------===//

def VMVX_OpInterface : OpInterface<"VMVXOp"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::VMVX";

  let description = [{
    Interface for VMVX ops that can be used to customize the lowering.
    This is required as there is not a way to get reflection information about
    ops.
  }];

  let methods = [
    StaticInterfaceMethod<
      "Extracts type information attributes that may be required by the op.",
      "void", "extractTypeAttributes", (ins "OperationState &":$state, "ArrayRef<Type>":$operandTypes, "ArrayRef<Type>":$resultTypes), [{
        ConcreteOp::extractTypeAttributes(state, operandTypes, resultTypes);
      }], [{
        return;  // default to no extraction
      }]
    >,
  ];
}

#endif  // IREE_DIALECT_MODULES_VMVX_INTERFACES
