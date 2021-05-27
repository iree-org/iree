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

#ifndef IREE_DIALECT_HAL_INTERFACES
#define IREE_DIALECT_HAL_INTERFACES

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

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

#endif  // IREE_DIALECT_HAL_INTERFACES
