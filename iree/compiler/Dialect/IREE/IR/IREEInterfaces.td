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

#ifndef IREE_DIALECT_IREE_INTERFACES
#define IREE_DIALECT_IREE_INTERFACES

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

//===----------------------------------------------------------------------===//
// IREE::TiedOpInterface
//===----------------------------------------------------------------------===//

def IREE_TiedOpInterface : OpInterface<"TiedOpInterface"> {
  let description = [{
    An operation that "ties" one or more results to its operands indicating
    that the result is directly related to the operand in an operation-defined
    way. Results are still SSA values distinct from the operands and the tie is
    strictly a relationship relevant to transformations and not something that
    modifies IR definitions.

    Example:
      An operation on tensors that wants to indicate that the storage for a
      result should alias the storage for an operand, performing an "in-place"
      operation. Since tensors are still used there is no hard requirement that
      uses of the result SSA value alias the operand; a copy may still be
      introduced.

      See: flow.dispatch.workgroups

    Example:
      An operation on buffers that wants to encode activity on the buffer in IR
      (such as a barrier, a transfer operation, etc) such that the SSA use-def
      chain is representing the state of the buffer at various points in time
      but that the underlying buffers are all tied together.

      See: hal.stream.barrier

    The default implementations use an attribute on the op to store the
    relationship:
      `OptionalAttr<IREE_TiedOpStorageAttr>:$tied_operands`

    Note that `$tied_operands` are indices inside the operand range returned
    by `getTiedOperandsIndexAndLength`, which may *not* be the full operand
    range of the op.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the set of operands that results may be tied to as an
        (index, length) pair ala getODSOperandIndexAndLength.

        By default assumes all operands may be tied. If an op treats some
        operands as special then the op can override this and specify only the
        ones it will tie. For example, a cond_branch that has a condition
        operand as well as the successor operands would return only the range
        of successor operands.
      }],
      /*retTy=*/"std::pair<unsigned, unsigned>",
      /*methodName=*/"getTiedOperandsIndexAndLength", (ins),
      /*args=*/[{}],
      /*defaultImplementation=*/[{
        return {0, $_op.getNumOperands()};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Walks up the SSA use-def chain to find the first defined value reachable
        from the given value by traversing tied ops. The returned value may be
        in another block if that block dominates the one the result is defined
        in.

        Note that the returned value may be a block argument and have no
        defining op, and the search will not continue past branches.
        If the result is untied then the result itself is returned.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"getTiedResult",
      /*args=*/(ins "unsigned":$resultIndex),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::TiedOpInterface::findTiedBaseValue($_op.getResult(resultIndex));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the operand tied to the given result of the op or nullptr if
        none.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"getTiedResultOperand",
      /*args=*/(ins "Value":$result),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        auto resultIndex = result.cast<mlir::OpResult>().getResultNumber();
        auto operandIndex = IREE::detail::getTiedResultOperandIndex($_op, resultIndex);
        return operandIndex.hasValue() ?
            $_op.getOperand(operandIndex.getValue()) :
            nullptr;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the operand index tied to the given result index, if any.

        Note that the index returned is into the full range of all operands of
        the current op.
      }],
      /*retTy=*/"::llvm::Optional<unsigned>",
      /*methodName=*/"getTiedResultOperandIndex",
      /*args=*/(ins "unsigned":$resultIndex),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::detail::getTiedResultOperandIndex($_op, resultIndex);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Sets the operand index tied to the given result index, if any.

        Note that the index should be into the operand range returned by
        `getTiedOperandsIndexAndLength`.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setTiedResultOperandIndex",
      /*args=*/(ins "unsigned":$resultIndex,
                    "::llvm::Optional<unsigned>":$operandIndex),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::detail::setTiedResultOperandIndex($_op, resultIndex, operandIndex);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns an array containing the tied result operand indices with -1
        indicating that a result is not tied.

        Note that the index returned is into the full range of all operands of
        the current op.
      }],
      /*retTy=*/"SmallVector<int64_t, 4>",
      /*methodName=*/"getTiedResultOperandIndices",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::detail::getTiedResultOperandIndices($_op);
      }]
    >,
  ];

  let extraClassDeclaration = [{
    static StringRef getStorageAttrName() { return "tied_operands"; }

    // Indicates that a result is not tied to any operand.
    static constexpr int64_t kUntiedIndex = -1;

    // Walks the SSA use-def chain to find the first defined value reachable
    // from the given value by traversing tied ops. Note that the returned
    // value may be a block argument and have no defining op.
    static Value findTiedBaseValue(Value derivedValue);
  }];

  let verify = [{
    return IREE::detail::verifyTiedOp($_op);
  }];
}

#endif  // IREE_DIALECT_IREE_INTERFACES
