// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_FLOW_INTERFACES
#define IREE_DIALECT_FLOW_INTERFACES

include "iree/compiler/Dialect/Util/IR/UtilBase.td"

//===----------------------------------------------------------------------===//
// IREE::Flow::StreamableOpInterface
//===----------------------------------------------------------------------===//

def FLOW_StreamableOp : OpInterface<"StreamableOpInterface"> {
  let description = [{
    Interface for ops that can be used within a stream.

    Some ops can exist both within a stream and outside of a stream. This allows
    optimizations to place certain ops such that they are performed in a
    synchronous (outside of a stream) or asynchronous (inside of a stream)
    fashion.

    The goal of the stream forming process is to move as many operations that
    can be used within a stream into one and only using non-streamed ops as a
    last resort.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the op is transfer operation (as defined by the HAL).
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isTransfer",
      /*args=*/(ins)
    >,
  ];
}

#endif  // IREE_DIALECT_FLOW_INTERFACES
