// Copyright 2019 Google LLC
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

#ifndef IREE_DIALECT_SHAPE_OPS
#define IREE_DIALECT_SHAPE_OPS

include "iree/compiler/Dialect/Shape/IR/ShapeBase.td"

//===----------------------------------------------------------------------===//
// Op types
//===----------------------------------------------------------------------===//

class Shape_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Shape_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ print$cppClass(p, *this); }];
}

class Shape_PureOp<string mnemonic, list<OpTrait> traits = []> :
    Shape_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

//===----------------------------------------------------------------------===//
// Dynamic shape support
//===----------------------------------------------------------------------===//

def Shape_TieShapeOp : Shape_PureOp<"tie_shape"> {
  let summary = "Ties a tensor and a shape together.";
  let description = [{
    Ties a specific tensor and its shape together in the IR, allowing further
    conversions to re-associate the two. This has no runtime implication and
    will be removed late in conversion.

    Usage:
      %0 = shape.tie_shape %1, %2 : tensor<...>, shape.ranked_shape<...>
  }];

  let arguments = (ins AnyTensor:$operand, Shape_RankedShape:$shape);
  let results = (outs AnyTensor:$result);

  let verifier = [{ return verify$cppClass(*this); }];
}

def Shape_GetRankedShapeOp : Shape_PureOp<"get_ranked_shape"> {
  let summary = "Gets the RankedShape associated with the given Tensor.";
  let description = [{
    Early in compilation, this op is used to resolve the RankedShape from an
    arbitrary tensor value. It will typically be converted later to a
    RankedShape loaded from an appropriate computation.

    Getting the RankedShape of a statically shaped tensor will canonicalize
    to a static_ranked_shape op and will never cause a further SSA dependency.

    Usage:
      %0 = shape.get_ranked_shape %arg0 : tensor<2x?xf32> ->
          !shape.ranked_shape<2x?xf32>

    Canonicalization: This op includes a canonicalization pattern such that
    if its operand is supplied by a tie_shape op, then it will replace itself
    with the tie_shape's shape() operand. In this way, a function with all
    shapes materialized and tied to intermediate tensors should canonicalize
    to contain no get_ranked_shape ops.
  }];

  let arguments = (ins AnyTensor:$operand);
  let results = (outs Shape_RankedShape:$shape);

  let verifier = [{ return verify$cppClass(*this); }];
  let hasCanonicalizer = 1;
}

#endif  // IREE_DIALECT_SHAPE_OPS
