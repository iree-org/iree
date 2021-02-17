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

#ifndef IREE_DIALECT_SHAPE_INTERFACES
#define IREE_DIALECT_SHAPE_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Op interfaces
//===----------------------------------------------------------------------===//

def Shape_ShapeCarryingOpInterface : OpInterface<"ShapeCarryingInterface"> {
  let description = [{
    Interface for ops that interact with dynamically shaped inputs and outputs.
    Such ops are able to materialize RankedShapes on demand for any operand or
    result that derives from ShapedType.
  }];

  let methods = [
    StaticInterfaceMethod<
      [{Returns a RankedShape for the given shaped result value.}],
      "Value", "buildResultValueRankedShape",
      (ins "Value":$result, "OpBuilder &":$builder), [{
        auto carryingOp = dyn_cast<ShapeCarryingInterface>(result.getDefiningOp());
        for (unsigned idx = 0; idx < carryingOp->getNumResults(); ++idx) {
          if (carryingOp->getResult(idx) == result) {
            return carryingOp.buildResultRankedShape(idx, builder);
          }
        }
        llvm_unreachable("called on result not from a shape-aware-op");
      }]
    >,
    InterfaceMethod<
      [{Returns a RankedShape for the given shaped operand index.}],
      "Value", "buildOperandRankedShape", (ins "unsigned":$idx, "OpBuilder &":$builder)
    >,
    InterfaceMethod<
      [{Returns a RankedShape for the given shaped result index.}],
      "Value", "buildResultRankedShape", (ins "unsigned":$idx, "OpBuilder &":$builder)
    >,
  ];
}

#endif  // IREE_DIALECT_SHAPE_INTERFACES
