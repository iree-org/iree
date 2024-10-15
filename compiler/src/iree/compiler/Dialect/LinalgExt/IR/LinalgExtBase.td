// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_BASE
#define IREE_DIALECT_LINALGEXT_BASE

include "mlir/Dialect/Utils/StructuredOpsUtils.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def IREELinalgExt_Dialect : Dialect {
  let name = "iree_linalg_ext";
  let cppNamespace = "::mlir::iree_compiler::IREE::LinalgExt";
  let summary = [{
    IREE Linalg Extensions.
  }];
  let description = [{
    A dialect designed for experimenting with non-structured operations that
    cannot be represented efficiently/directly by the Linalg dialect.
  }];
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

class RankedTensorOrMemRefOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes,
      Or<[IsMemRefTypePred, And<[IsTensorTypePred, HasRankPred]>]>,
  "ranked tensor or memref", "::mlir::ShapedType">;

def AnyRankedTensorOrMemRefType : RankedTensorOrMemRefOf<[AnyType]>;

class RankedTensorOrScalarType<list<Type> allowedTypes> :
  AnyTypeOf<[RankedTensorOf<allowedTypes>, AnyTypeOf<allowedTypes>]>;

def AnyRankedTensorOrScalarType :
  RankedTensorOrScalarType<[AnySignlessIntegerOrIndex, AnyFloat]>;

//===----------------------------------------------------------------------===//
// Enum definitions
//===----------------------------------------------------------------------===//

def IREELinalgExt_IteratorTypeEnum : EnumAttr<IREELinalgExt_Dialect,
    IteratorType, "iterator_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def IREELinalgExt_IteratorTypeArrayAttr :
    TypedArrayAttrBase<IREELinalgExt_IteratorTypeEnum,
    "LinalgExt iterator type">;

#endif // IREE_DIALECT_LINALGEXT_BASE
