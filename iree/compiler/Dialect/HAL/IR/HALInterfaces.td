// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_HAL_INTERFACES
#define IREE_DIALECT_HAL_INTERFACES

include "iree/compiler/Dialect/Util/IR/UtilBase.td"

//===----------------------------------------------------------------------===//
// IREE::HAL::SizeAwareOpInterface
//===----------------------------------------------------------------------===//

def HAL_InferTypeSize : TypeInterface<"InferTypeSizeInterface"> {
  let description = [{
    Allows types to be queried for their size by inserting the required logic
    when required.
  }];

  let methods = [
    InterfaceMethod<
      [{Builds an expression computing the size of the value.}],
      "Value", "inferSizeFromValue", (ins "Location":$loc,
                                          "Value":$value,
                                          "OpBuilder &":$builder)
    >,
  ];
}

def HAL_SizeAwareType : TypeInterface<"SizeAwareTypeInterface"> {
  let description = [{
    Denotes that a type is size-aware and must always have a size value
    associated with it in the IR. See `SizeAwareOp` for more information.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns a size for the given sized value.}],
      "Value", "getSize", (ins "Value":$value)
    >,
  ];
}

def HAL_SizeAwareOp : OpInterface<"SizeAwareOpInterface"> {
  let description = [{
    An operation that is able to provide size values for all size-aware operands
    and results.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns a size for the given sized operand index.}],
      "Value", "getOperandSize", (ins "unsigned":$idx)
    >,
    InterfaceMethod<
      [{Returns a size for the given sized result index.}],
      "Value", "getResultSize", (ins "unsigned":$idx)
    >,
    InterfaceMethod<
      [{Returns a size for the given sized result value.}],
      "Value", "getResultSizeFromValue", (ins "Value":$value),
      /*defaultImplementation=*/[{
        for (unsigned i = 0; i < $_self->getNumResults(); ++i) {
          if ($_self->getResult(i) == value) return $_self.getResultSize(i);
        }
        return {};
      }]
    >,
  ];
}

def HAL_MatchAttrInterface :
    AttrInterface<"MatchAttrInterface"> {
  let description = [{
    An attribute that can be used in `hal.*.match.*` expressions.
    Each attribute defines some subexpression that can be expanded to one or
    more operations that performs the actual query and matching logic.
  }];

  let methods = [
    InterfaceMethod<
      [{
        Builds a set of operations that evaluate to a boolean (i1) value
        indicating whether the expression tree represented by the match
        attribute is true for the given value.
      }],
      "Value", "buildConditionExpression",
      (ins "Location":$loc, "Value":$value, "OpBuilder":$builder)
    >,
  ];
}

#endif  // IREE_DIALECT_HAL_INTERFACES
