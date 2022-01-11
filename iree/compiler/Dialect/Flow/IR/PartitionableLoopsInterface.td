// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_FLOW_IR_PARTITIONABLE_LOOPS_INTERFACE
#define IREE_DIALECT_FLOW_IR_PARTITIONABLE_LOOPS_INTERFACE

include "mlir/IR/OpBase.td"

def FLOW_PartitionableLoopsInterface : OpInterface<"PartitionableLoopsInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Flow";

  let description = [{
    An interface for ops to indicate which implicit loops that can be partitioned.

    Using this interface allows for consistent specification of which loops of
    an operation are partitioned.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the total number of loops that the operation represents.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumLoops"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the loop IDs (0 being outermost) that are partitionable.

        The size of the vector returned is always lesser than
        `maxNumPartitionedLoops`.
      }],
      /*retTy=*/"llvm::SmallVector<unsigned>",
      /*methodName=*/"getPartitionableLoops",
      /*args=*/(ins "unsigned":$maxNumPartitionedLoops)
    >
  ];
}

#endif // IREE_DIALECT_FLOW_IR_PARTITIONABLE_LOOPS_INTERFACE
