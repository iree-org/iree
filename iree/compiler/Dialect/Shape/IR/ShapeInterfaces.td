// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_SHAPE_INTERFACES
#define IREE_DIALECT_SHAPE_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Op interfaces
//===----------------------------------------------------------------------===//

def Shape_ShapeCarryingOpInterface : OpInterface<"ShapeCarryingInterface"> {
  let description = [{
    Interface for ops that interact with dynamically shaped inputs and outputs.
    Such ops are able to materialize RankedShapes on demand for any operand or
    result that derives from ShapedType.
  }];

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/[{Returns a RankedShape for the given shaped result value.}],
      /*retTy=*/"Value",
      /*methodName=*/"buildResultValueRankedShape",
      /*args=*/(ins "Value":$result, "OpBuilder &":$builder),
      /*methodBody=*/[{
        auto carryingOp = dyn_cast<ShapeCarryingInterface>(result.getDefiningOp());
        return carryingOp.buildResultRankedShape(
            result.cast<mlir::OpResult>().getResultNumber(), builder);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{Returns a RankedShape for the given shaped operand index.}],
      /*retTy=*/"Value",
      /*methodName=*/"buildOperandRankedShape",
      /*args=*/(ins "unsigned":$idx, "OpBuilder &":$builder)
    >,
    InterfaceMethod<
      /*desc=*/[{Returns a RankedShape for the given shaped result index.}],
      /*retTy=*/"Value",
      /*methodName=*/"buildResultRankedShape",
      /*args=*/(ins "unsigned":$idx, "OpBuilder &":$builder)
    >,
  ];
}

#endif  // IREE_DIALECT_SHAPE_INTERFACES
