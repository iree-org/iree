// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_INTERFACES_PARTITIONABLE_LOOPS_INTERFACE
#define IREE_CODEGEN_INTERFACES_PARTITIONABLE_LOOPS_INTERFACE

include "mlir/IR/OpBase.td"

def PartitionableLoopsInterface : OpInterface<"PartitionableLoopsInterface"> {
  let cppNamespace = "::mlir::iree_compiler";

  let description = [{
    An interface for ops to indicate which implicit loops that can be partitioned.

    Using this interface allows for consistent specification of which loops of
    an operation are partitioned.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the loop IDs (0 being outermost) that are partitionable.

        If `maxNumPartitionedLoops` is passed the size of the vector returned
        is always lesser than the value.
      }],
      /*retTy=*/"::llvm::SmallVector<unsigned>",
      /*methodName=*/"getPartitionableLoops",
      /*args=*/(ins "std::optional<unsigned>":$maxNumPartitionedLoops),
      /*methodBody=*/"",
      /*defaultImplementation*/"return {};"
    >,
  ];
}

#endif // IREE_CODEGEN_INTERFACES_PARTITIONABLE_LOOPS_INTERFACE
