// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VECTOREXT_INTERFACES
#define IREE_DIALECT_VECTOREXT_INTERFACES

include "mlir/IR/OpBase.td"

def VectorLayoutInterface : AttrInterface<"VectorLayoutInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::VectorExt";

  let description = [{
    Vector Layout Attribute Interface.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Check if this layout is valid for the given shape.",
      /*retTy=*/"LogicalResult",
      /*methodName=*/"isValidLayout",
      /*args=*/(ins "::mlir::ShapedType":$shape, "::mlir::Location":$loc)
    >,
    InterfaceMethod<
      /*description=*/"Permutes the given layout.",
      /*retTy=*/"VectorLayoutInterface",
      /*methodName=*/"permute",
      /*args=*/(ins "::llvm::ArrayRef<int64_t>":$permutation)
    >,
    InterfaceMethod<
      /*description=*/"Projects the given layout.",
      /*retTy=*/"VectorLayoutInterface",
      /*methodName=*/"project",
      /*args=*/(ins "::llvm::ArrayRef<bool>":$droppedDims)
    >,
    InterfaceMethod<
      /*description=*/"Get the distributed shape for the given vector type.",
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getDistributedShape",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*description=*/"Get the rank of the undistributed vector for this layout.",
      /*retTy=*/"int64_t",
      /*methodName=*/"getRank",
      /*args=*/(ins)
    >
  ];
}

#endif  // IREE_DIALECT_VECTOREXT_INTERFACES
