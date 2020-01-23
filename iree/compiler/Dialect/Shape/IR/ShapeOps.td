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

def Shape_GetRankedShapeOp : Shape_PureOp<"get_ranked_shape"> {
  let summary = "Gets the RankedShape associated with the given Tensor.";
  let description = [{
    Early in compilation, this op is used to resolve the RankedShape from an
    arbitrary tensor value. It will typically be converted later to a
    RankedShape loaded from an appropriate computation.

    Getting the RankedShape of a statically shaped tensor will canonicalize
    to a static_ranked_shape op and will never cause a further SSA dependency.
  }];

  let arguments = (ins AnyTensor:$operand);
  let results = (outs Shape_RankedShape:$shape);

  let verifier = [{ return verify$cppClass(*this); }];
}

#endif  // IREE_DIALECT_SHAPE_OPS
