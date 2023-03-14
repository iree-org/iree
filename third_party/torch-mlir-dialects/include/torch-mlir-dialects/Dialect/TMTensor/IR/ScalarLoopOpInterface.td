//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECT_TMTENSOR_SCALARLOOPOPINTERFACE
#define TORCH_MLIR_DIALECT_TMTENSOR_SCALARLOOPOPINTERFACE

include "mlir/IR/OpBase.td"

def ScalarLoopOpInterface : OpInterface<"ScalarLoopOpInterface"> {
  let description = [{
    Interface for allowing operations to expose information needed to
    lower it to for loops
  }];
  let cppNamespace = "::mlir::torch::TMTensor";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of `IteratorTYpe`s that describe the number of
          loops and the iterator types of the operation. The list is
          expected to use
          `getParallelIteratorTypeName()`/`getReductionIteratorTypeName()`
          from MLIR Structured Op Utils.
        }],
        /*retType=*/"SmallVector<utils::IteratorType>",
        /*methodName=*/"getLoopIteratorTypes"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of ranges that describe the loop bounds and
          step for the loops of the operation.
        }],
        /*retTy=*/"SmallVector<Range>",
        /*methodName=*/"getIterationDomain",
        /*args=*/(ins "OpBuilder &":$b)
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

#endif  // TORCH_MLIR_DIALECT_TMTENSOR_SCALARLOOPOPINTERFACES
