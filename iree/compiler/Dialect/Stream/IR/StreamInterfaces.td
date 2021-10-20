// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_STREAM_INTERFACES
#define IREE_DIALECT_STREAM_INTERFACES

include "iree/compiler/Dialect/Util/IR/UtilBase.td"

//===----------------------------------------------------------------------===//
// IREE::Stream::AffinityOpInterface
//===----------------------------------------------------------------------===//

def Stream_AffinityOp : OpInterface<"AffinityOpInterface"> {
  let description = [{
    TBD. Used to denote a stream affinity for ops and specify the kind of
    environment the ops are expected run in.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the stream affinity for the op, indicating where it should run.
      }],
      /*retTy=*/"IREE::Stream::AffinityAttr",
      /*methodName=*/"getAffinity",
      /*args=*/(ins),
      /*methodBody=*/[{
        return $_self->getAttr("affinity").template dyn_cast_or_null<IREE::Stream::AffinityAttr>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Sets the stream affinity for the op, indicating where it should run.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setAffinity",
      /*args=*/(ins "IREE::Stream::AffinityAttr":$value),
      /*methodBody=*/[{
        $_self->setAttr("affinity", value);
      }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::Stream::StreamableOpInterface
//===----------------------------------------------------------------------===//

def Stream_StreamableOp : OpInterface<"StreamableOpInterface"> {
  let description = [{
    Interface for ops that can be asynchronous executed in a streaming context.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the op is slicing out/in memory instead of real work.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isMetadata",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the op should be cloned into consumer streams.
        These ops should be cheaper to recompute than to transfer their contents
        across streams (such as splats).
      }],
      /*retTy=*/"bool",
      /*methodName=*/"preferCloneToConsumers",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
  ];
}

//===----------------------------------------------------------------------===//
// IREE::Stream::SubviewEffectOpInterface
//===----------------------------------------------------------------------===//

def Stream_SubviewEffectOp : OpInterface<"SubviewEffectOpInterface"> {
  let description = [{
    Interface for ops that operate on subviews of resources used to query the
    memory effects for subviews on operands.
  }];

  let methods = [
    // TODO(benvanik): get memory effect + range of an operand
  ];
}

//===----------------------------------------------------------------------===//
// IREE::Stream::AsyncOpInterface
//===----------------------------------------------------------------------===//

// TODO(benvanik): add interface for async ops:
//   getWaitTimepoint
//   setWaitTimepoint
//   getWaitResources
//   getSignalTimepoint
//   setSignalTimepoint
//   getSignalResources
//   + maybe mutable resource accessors? (MutableOperandRange)
// This would let us rework code relying on AsyncExecuteOp/CmdExecuteOp to work
// with both, and wait elision canonicalization patterns to be shared across
// the async resource ops and execution ops.

#endif  // IREE_DIALECT_STREAM_INTERFACES
