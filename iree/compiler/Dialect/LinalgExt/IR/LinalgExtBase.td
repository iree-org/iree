// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_BASE
#define IREE_DIALECT_LINALGEXT_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def LinalgExt_Dialect : Dialect {
  let name = "linalg_ext";
  let cppNamespace = "::mlir::iree_compiler::linalg_ext";
  let description = [{
    The `linalg_ext` dialect is intended to experiment more support for
    non-structured operations, ie, can not be represented in Linalg operations.
  }];
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

class RankedTensorOrMemRefOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes,
      Or<[IsMemRefTypePred, And<[IsTensorTypePred, HasRankPred]>]>,
  "ranked tensor or memref", "::mlir::ShapedType">;

def AnyRankedTensorOrMemRefType : RankedTensorOrMemRefOf<[AnyType]>;

#endif // IREE_DIALECT_LINALGEXT_BASE
