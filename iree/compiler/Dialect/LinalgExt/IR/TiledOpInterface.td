// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_TILEDOPINTERFACE
#define IREE_DIALECT_LINALGEXT_TILEDOPINTERFACE

include "mlir/IR/OpBase.td"

def TiledOpInterface : OpInterface<"TiledOpInterface"> {
  let description = [{
    Interface for allowing operations to expose information needed to
    tile it (similar to LinalgOp, but without having access to
    indexing maps)
  }];
  let cppNamespace = "::mlir::iree_compiler::linalg_ext";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of `StringRef`s that describe the number of
          loops and the iterator types of the operation. The list is
          expected to use
          `getParallelIteratorTypeName()`/`getReductionIteratorTypeName()`
          from MLIR Structured Op Utils.
        }],
        /*retType=*/"SmallVector<StringRef>",
        /*methodName=*/"getLoopIteratorTypes"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of ranges that describe the loop bounds and
          step for the loops of the operation.
        }],
        /*retTy=*/"SmallVector<Range>",
        /*methodName=*/"getLoopBounds",
        /*args=*/(ins "OpBuilder &":$b)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Generates a tiled version of the operation given the tile
          size for the loops.
        }],
        /*retType=*/"Operation *",
        /*methodName=*/"getTiledImplementation",
        /*args=*/(ins
            "OpBuilder &":$b,
            "ValueRange ":$outputs,
            "ArrayRef<OpFoldResult> ":$offsets,
            "ArrayRef<OpFoldResult> ":$sizes,
            "SmallVectorImpl<SmallVector<OpFoldResult, 4>> &":$resultOffsets,
            "SmallVectorImpl<SmallVector<OpFoldResult, 4>> &":$resultSizes),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return nullptr;
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Generates the loop body implementation. Assume that all the parallel
          loops and reduction loops are created and the insertion point of the
          build is set to the innermost of the loop. This method implements the
          loop body IRs.
        }],
        /*retType=*/"LogicalResult",
        /*methodName=*/"generateScalarImplementation",
        /*args=*/(ins
            "OpBuilder &":$b,
            "Location ":$loc,
            "ValueRange ":$ivs),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return failure();
        }]
      >
  ];
}

#endif  // IREE_DIALECT_LINALGEXT_TILEDOPINTERFACES
