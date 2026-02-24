// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_INTERFACES_HOISTABLE_REGION_OP_INTERFACE
#define IREE_CODEGEN_INTERFACES_HOISTABLE_REGION_OP_INTERFACE

include "mlir/IR/OpBase.td"

def HoistableRegionOpInterface
    : OpInterface<"HoistableRegionOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";

  let description = [{
    Interface for operations with regions from which loop-invariant code
    can be hoisted. This covers non-loop region ops (like barrier regions)
    and loop-like ops (like linalg.generic) that don't implement
    LoopLikeOpInterface.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the regions to inspect for hoistable operations.
      }],
      /*retType=*/"::llvm::SmallVector<::mlir::Region *>",
      /*methodName=*/"getHoistableRegions"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the value is defined outside of this op's regions.
      }],
      /*retType=*/"bool",
      /*methodName=*/"isDefinedOutsideOfRegions",
      /*args=*/(ins "::mlir::Value":$value),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return !$_op->isAncestor(value.getParentRegion()->getParentOp());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the given op inside the region is safe to hoist out.

        Implementations should account for whether the region is guaranteed
        to execute. For regions that may not execute (e.g., linalg.generic
        with potentially zero trip count), this should require
        isSpeculatable in addition to isMemoryEffectFree. For
        always-executing regions (e.g., barrier_region), only
        isMemoryEffectFree is needed.
      }],
      /*retType=*/"bool",
      /*methodName=*/"isHoistable",
      /*args=*/(ins "::mlir::Operation *":$op)
    >,
  ];
}

#endif // IREE_CODEGEN_INTERFACES_HOISTABLE_REGION_OP_INTERFACE
