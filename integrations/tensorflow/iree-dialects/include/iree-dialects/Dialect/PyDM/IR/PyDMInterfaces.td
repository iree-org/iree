// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_PYDM_IR_PYDM_INTERFACES_TD
#define IREE_DIALECTS_DIALECT_PYDM_IR_PYDM_INTERFACES_TD

include "mlir/IR/OpBase.td"

def PythonTypeInterface : TypeInterface<"PythonTypeInterface"> {
  let description = [{
    Interface implemented by all Python types which represent a value in the
    data model.
  }];
  let cppNamespace = "::mlir::iree_compiler::IREE::PYDM";

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
      Returns whether the type can be refined any further or whether it is
      already in the most refined form. Most types default to false.
    }], "bool", "isRefinable", (ins),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{
      return false;
    }]>,

    InterfaceMethod<[{
      For numeric types, returns the NumericCategory.
    }], "llvm::Optional<NumericCategory>", "getNumericCategory", (ins),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{
      return {};
    }]>,

    InterfaceMethod<[{
      For numeric types, returns an appropriate subtype code, which is an
      integer from 0-3 representing the specific type with the NumericCategory.
      Weak types return None if though getNumericCategory() returns a value.
    }], "llvm::Optional<int>", "getNumericSubTypeCode", (ins),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{
      return {};
    }]>,

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

def TypeRefinableOpInterface : OpInterface<"TypeRefinableOpInterface"> {
  let description = [{
    Implemented by ops which can cooperate with type inference/refinement.
  }];
  let cppNamespace = "::mlir::iree_compiler::IREE::PYDM";

  let methods = [
    InterfaceMethod<[{
      Performs an in-place refinement of the result types, returning whether
      any changes were made. As a general rule, it is only legal to *refine*
      types into a more specialized form which is suitable for a
      static_info_cast.
    }], "bool", "refineResultTypes", (ins),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{
      return false;
    }]>
  ];
}

#endif // IREE_DIALECTS_DIALECT_PYDM_IR_PYDM_INTERFACES_TD
