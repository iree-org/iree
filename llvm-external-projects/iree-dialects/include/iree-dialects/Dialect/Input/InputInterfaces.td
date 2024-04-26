// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_INPUT_INTERFACES_TD
#define IREE_DIALECTS_DIALECT_INPUT_INTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREE::Input::TiedOpInterface
//===----------------------------------------------------------------------===//

def IREEInput_TiedOpInterface : OpInterface<"TiedOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Input";

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

    See Util dialect for more documentation and examples.

    The default implementations use an attribute on the op to store the
    relationship:
      `OptionalAttr<IREEInput_TiedOpStorageAttr>:$tied_operands`

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
        return {0, $_op->getNumOperands()};
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the set of results that may be tied to operands as an
        (index, length) pair ala getODSResultIndexAndLength.

        By default assumes all results may be tied. If an op treats some
        results as special then the op can override this and specify only the
        ones it will tie.
      }],
      /*retTy=*/"std::pair<unsigned, unsigned>",
      /*methodName=*/"getTiedResultsIndexAndLength", (ins),
      /*args=*/[{}],
      /*defaultImplementation=*/[{
        return {0, $_op->getNumResults()};
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
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getTiedResult",
      /*args=*/(ins "unsigned":$resultIndex),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::Input::TiedOpInterface::findTiedBaseValue(
          $_op.getResult(resultIndex));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the operand tied to the given result of the op or nullptr if
        none.
      }],
      /*retTy=*/"mlir::Value",
      /*methodName=*/"getTiedResultOperand",
      /*args=*/(ins "Value":$result),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        auto resultIndex = cast<mlir::OpResult>(result).getResultNumber();
        auto operandIndex = cast<TiedOpInterface>($_op.getOperation())
            .getTiedResultOperandIndex(resultIndex);
        return operandIndex.has_value() ?
            $_op.getOperand(operandIndex.value()) :
            nullptr;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the operand index tied to the given result index, if any.

        Note that the index returned is into the full range of all operands of
        the current op.
      }],
      /*retTy=*/"::std::optional<unsigned>",
      /*methodName=*/"getTiedResultOperandIndex",
      /*args=*/(ins "unsigned":$resultIndex),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::Input::detail::getTiedResultOperandIndex($_op,
                                                              resultIndex);
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
                    "::std::optional<unsigned>":$operandIndex),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::Input::detail::setTiedResultOperandIndex($_op, resultIndex,
                                                              operandIndex);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns an array containing the tied result operand indices with -1
        indicating that a result is not tied.

        Note that the index returned is into the full range of all operands of
        the current op.
      }],
      /*retTy=*/"SmallVector<int64_t>",
      /*methodName=*/"getTiedResultOperandIndices",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::Input::detail::getTiedResultOperandIndices($_op);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the given flattened operand index is tied to one or more
        results.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isOperandTied",
      /*args=*/(ins "unsigned":$operandIndex),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::Input::detail::isOperandTied($_op, operandIndex);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns a list of result values that are tied to the given operand.
      }],
      /*retTy=*/"SmallVector<Value>",
      /*methodName=*/"getOperandTiedResults",
      /*args=*/(ins "unsigned":$operandIndex),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return IREE::Input::detail::getOperandTiedResults($_op, operandIndex);
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

    // Returns true if any of |value|'s uses have tied it to a result.
    static bool hasAnyTiedUses(Value value);
  }];

  let verify = [{
    return IREE::Input::detail::verifyTiedOp(cast<TiedOpInterface>($_op));
  }];
}

#endif // IREE_DIALECTS_DIALECT_INPUT_INTERFACES_TD
