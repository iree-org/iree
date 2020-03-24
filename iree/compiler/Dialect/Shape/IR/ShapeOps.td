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
include "mlir/Interfaces/SideEffects.td"
include "mlir/IR/OpAsmInterface.td"

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
// RankedShapeType manipulation
//===----------------------------------------------------------------------===//

def Shape_TieShapeOp : Shape_PureOp<"tie_shape",
    [AllTypesMatch<["operand", "result"]>]> {
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

  let assemblyFormat = "operands attr-dict `:` type($operand) `,` type($shape)";

  let verifier = [{ return verify$cppClass(*this); }];

  let builders = [
    // Short-hand for building with a shape equivalent to the tensor operand
    // and 'index' dim type.
    OpBuilder<[{
      Builder *builder, OperationState &result, Value operand, Value shape
    }]>,
  ];
  let hasCanonicalizer = 1;
}

def Shape_CastCompatibleShapeOp : Shape_PureOp<"cast_compatible_shape"> {
  let summary = "Casts and asserts that one or more shapes are compatible.";
  let description = [{
    In the most general case, each operand and the result can have different
    shapes, so long as:
      a) The operand shapes are co-compatible (known dimensions are equal
         or unknown).
      b) The result shape is at least as general as the operand shapes.

    TODO: This op is currently very conservative, statically verifying that
    all of the shapes are strictly the same.

    Usage:
      %0 = shape.cast_compatible_shape %1, ... %3 :
          !shape.ranked_shape<...>...
  }];

  let arguments = (ins Variadic<Shape_RankedShape>:$operands);
  let results = (outs Shape_RankedShape:$result);

  let assemblyFormat = [{
    operands `:` type(operands) `->` type($result) attr-dict
  }];

  let verifier = [{ return verify$cppClass(*this); }];
  let hasCanonicalizer = 1;
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
          !shape.ranked_shape<[2,?]>  // based on index type
      %0 = shape.get_ranked_shape %arg0 : tensor<2x?xf32> ->
          !shape.ranked_shape<[2,?],i32>  // explicit dim type

    Canonicalization: This op includes a canonicalization pattern such that
    if its operand is supplied by a tie_shape op, then it will replace itself
    with the tie_shape's shape() operand. In this way, a function with all
    shapes materialized and tied to intermediate tensors should canonicalize
    to contain no get_ranked_shape ops.

    Any get_ranked_shape on a fully static shape will canonicalize to a const
    with unit value:
      %0 = constant_ranked_shape : !shape.ranked_shape<[1,2],i32>
  }];

  let arguments = (ins AnyTensor:$operand);
  let results = (outs Shape_RankedShape:$shape);

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($shape)
  }];

  let verifier = [{ return verify$cppClass(*this); }];
  let hasCanonicalizer = 1;

  let builders = [
    // Short-hand for building with a shape equivalent to the tensor operand
    // and 'index' dim type.
    OpBuilder<[{
      Builder *builder, OperationState &result, Value operand
    }]>,
  ];

  let extraClassDeclaration = [{
    RankedShapeType getRankedShape() {
      return shape().getType().cast<RankedShapeType>();
    }
  }];
}

def Shape_ConstRankedShapeOp : Shape_PureOp<"const_ranked_shape",
    [ConstantLike, DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
  let summary = "A constant ranked_shape.";
  let description = [{
    Holds a RankedShape value. Note that it is only legal to store a constant
    RankedShape that is fully static, as anything more specific should be
    in the type, not have dims represented as const SSA values.

    Usage:
      %0 = shape.const_ranked_shape : !shape.ranked_shape<[1,2]>
  }];

  let arguments = (ins);
  let results = (outs Shape_RankedShape:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result, Type type
      }]>,
  ];
  let verifier = [{ return verify$cppClass(*this); }];
}

def Shape_MakeRankedShapeOp : Shape_PureOp<"make_ranked_shape"> {
  let summary = "Makes a ranked_shape from individual dims.";
  let description = [{
    Given a list of SSA values holding compatible dims, makes a corresponding
    ranked_shape.

    Usage:
      %0 = shape.make_ranked_shape %dim0, %dim1 : (i32, i32) ->
          !shape.ranked_shape<[?,?,128]>

    Note that the type of the dims is is implied by the dim type of the result.
  }];

  let arguments = (ins Variadic<Shape_DimType>:$dynamic_dimensions);
  let results = (outs Shape_RankedShape:$shape);

  let assemblyFormat = "$dynamic_dimensions `:` functional-type($dynamic_dimensions, $shape) attr-dict";

  let extraClassDeclaration = [{
    RankedShapeType getRankedShapeType() {
      return shape().getType().cast<RankedShapeType>();
    }
  }];
  let verifier = [{ return verify$cppClass(*this); }];
  let hasCanonicalizer = 1;
}

def Shape_RankedDimOp : Shape_PureOp<"ranked_dim"> {
  let summary = "Gets a dimension value from a ranked_shape.";
  let description = [{
    Static dimensions will fold to constants.

    Usage:
      %0 = shape.const ranked_shape : !shape.ranked_shape<[1,2]>
      %1 = shape.ranked_dim %0[0] : !shape.ranked_shape<[1,2]> -> i32
  }];

  let arguments = (ins Shape_RankedShape:$shape,
                   APIntAttr:$index);
  let results = (outs Shape_DimType:$result);
  let verifier = [{ return verify$cppClass(*this); }];

  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result,
      Type dimType, Value shape, int index
    }]>,
    // dimType is defaulted to IndexType.
    OpBuilder<[{
      Builder *builder, OperationState &result,
      Value shape, int index
    }]>,
  ];

  let extraClassDeclaration = [{
    RankedShapeType getRankedShapeType() {
      return shape().getType().cast<RankedShapeType>();
    }
    unsigned getIndex() {
      return getAttrOfType<IntegerAttr>("index").getValue().getZExtValue();
    }
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Shape_RankedDimsOp : Shape_PureOp<"ranked_dims"> {
  let summary = "Gets all dimension values from a ranked_shape.";
  let description = [{
    Static dimensions will fold to constants.

    Usage:
      %0 = shape.const ranked_shape : !shape.ranked_shape<[1,2]>
      %1, %2 = shape.ranked_dims %0 : !shape.ranked_shape<[1,2]> -> (i32, i32)
  }];

  let arguments = (ins Shape_RankedShape:$shape);
  let results = (outs Variadic<Shape_DimType>:$result);

  let assemblyFormat = "$shape `:` type($shape) `->` type($result) attr-dict";

  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result, Type dimType, Value shape
    }]>,
    OpBuilder<[{
      Builder *builder, OperationState &result, Value shape
    }]>,
  ];

  let extraClassDeclaration = [{
    RankedShapeType getRankedShapeType() {
      return shape().getType().cast<RankedShapeType>();
    }
  }];
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Broadcasting
//===----------------------------------------------------------------------===//

def Shape_RankedBroadcastShapeOp : Shape_PureOp<"ranked_broadcast_shape"> {
  let summary = "Broadcasts operands to a result shape.";
  let description = [{
    Applies numpy broadcasting semantics to shape operands.

    Usage:
      %0 = shape.ranked_broadcast_shape %shp0, %shp1 :
          !shape.ranked_shape<...>, !shape.ranked_shape<...>
  }];

  let arguments = (ins Shape_RankedShape:$lhs,
                       Shape_RankedShape:$rhs,
                       I64ElementsAttr:$lhs_broadcast_dimensions,
                       I64ElementsAttr:$rhs_broadcast_dimensions);
  let results = (outs Shape_RankedShape:$result);

  // TODO: Custom parser/printer
  let parser = ?;
  let printer = ?;
}

def Shape_RankedBroadcastInDimOp : Shape_PureOp<"ranked_broadcast_in_dim"> {
  let summary = "Broadcasts dimensions from the input into the result.";
  let description = [{
    Usage:
      %0 = shape.ranked_broadcast_in_dim [...] (%operand, %result_shp) :
          tensor<...xf32>, !shape.ranked_shape<...xi32>

      Note that the result type will be a RankedTensorType with dims from
      %result_shp and the element type from %operand.
  }];

  let arguments = (ins AnyRankedTensor:$operand,
                       Shape_RankedShape:$result_shape,
                       OptionalAttr<I64ElementsAttr>:$broadcast_dimensions);
  let results = (outs AnyRankedTensor:$result);

  // TODO: Custom parser/printer
  let parser = ?;
  let printer = ?;
}

#endif  // IREE_DIALECT_SHAPE_OPS
