// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_HAL_INTERFACES
#define IREE_DIALECT_HAL_INTERFACES

include "iree/compiler/Dialect/Util/IR/UtilBase.td"

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
      (ins "Location":$loc, "Value":$device, "OpBuilder":$builder)
    >,
  ];
}

#endif  // IREE_DIALECT_HAL_INTERFACES
