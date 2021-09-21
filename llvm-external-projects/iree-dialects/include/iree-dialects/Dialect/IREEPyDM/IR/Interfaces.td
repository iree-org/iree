// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_INTERFACES_TD
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_INTERFACES_TD

include "mlir/IR/OpBase.td"

def PythonTypeInterface : TypeInterface<"PythonTypeInterface"> {
  let description = [{
    Interface implemented by all Python types which represent a value in the
    data model.
  }];
  let cppNamespace = "::mlir::iree_pydm";

  let methods = [
    InterfaceMethod<[{
      Gets the type code.
    }], "BuiltinTypeCode", "getTypeCode", (ins)>,

    InterfaceMethod<[{
      Gets a Python-relevant type name for this type.

      This is used for both diagnostic messages and sorting order. No two
      MLIR types can have the same name.
    }], "llvm::StringRef", "getPythonTypeName", (ins)>,

    InterfaceMethod<[{
      For numeric types, returns the promotion order.
      Types with a lower promotion order will be promoted to the higher order
      for most binary functions.
    }], "llvm::Optional<int>", "getNumericPromotionOrder", (ins),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{
      return {};
    }]>
  ];
}

#endif // IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_INTERFACES_TD
