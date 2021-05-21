// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_DIALECT_FLOW_INTERFACES
#define IREE_DIALECT_FLOW_INTERFACES

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

//===----------------------------------------------------------------------===//
// IREE::Flow::ClosureOpInterface
//===----------------------------------------------------------------------===//

def FLOW_ClosureOpInterface : OpInterface<"ClosureOpInterface"> {
  let description = [{
    Interface for ops that follow the Flow dialect closure semantics (explicit
    captures, dynamic-shape awareness, and normal operand/result SSA behavior).

    Implementing this interface enables optimizations that perform manipulation
    across the closure capture boundary (outside of the op <-> regions within
    the op).
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the body region of the closure (may have multiple blocks).
      }],
      /*retTy=*/"Region &",
      /*methodName=*/"getClosureBodyRegion",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return this->getOperation()->getRegion(0);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{Returns all closure operand values.}],
      /*retTy=*/"Operation::operand_range",
      /*methodName=*/"getClosureOperands",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{Returns all closure result values.}],
      /*retTy=*/"Operation::result_range",
      /*methodName=*/"getClosureResults",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the given operation can exist in the closure.
        Not all operations that a closure can contain are guaranteed to be folded
        into the closure, such as when the operation may have side-effects.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"canClosureContainOp",
      /*args=*/(ins "Operation *":$op)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Clones the op while removing specified operands and results.
        The body of the op will be transferred to the new op and the entry block
        will have its arguments removed.

        The returned op will be free standing. Callers must insert it into a block
        where desired (most often just replacing the current op).
      }],
      /*retTy=*/"ClosureOpInterface",
      /*methodName=*/"cloneReplacementExcludingOperandsAndResults",
      /*args=*/(ins "ArrayRef<unsigned>":$excludedOperandIndices,
                    "ArrayRef<unsigned>":$excludedResultIndices)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the output is also read within the region.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isOutputReadWithinRegion",
      /*args=*/(ins "unsigned":$resultIndex)
    >
  ];
}

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
