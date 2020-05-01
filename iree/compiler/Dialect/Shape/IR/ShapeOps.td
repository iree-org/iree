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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffects.td"
include "mlir/Interfaces/ViewLikeInterface.td"
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

def Shape_TieShapeOp : Shape_PureOp<"tie_shape", [
    AllTypesMatch<["operand", "result"]>,
    DeclareOpInterfaceMethods<ViewLikeOpInterface>
  ]> {
  let summary = "Ties a tensor and a shape together.";
  let description = [{
    Ties a specific tensor and its shape together in the IR, allowing further
    conversions to re-associate the two. This has no runtime implication and
    will be removed late in conversion.

    Usage:
      %0 = shape.tie_shape %1, %2 : tensor<...>, shape.ranked_shape<...>
  }];

  let arguments = (ins AnyType:$operand, Shape_RankedShape:$shape);
  let results = (outs AnyType:$result);

  let assemblyFormat = "operands attr-dict `:` type($operand) `,` type($shape)";

  let verifier = [{ return verify$cppClass(*this); }];

  let builders = [
    // Short-hand for building with a shape equivalent to the tensor operand
    // and 'index' dim type.
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, Value operand, Value shape
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
      OpBuilder &builder, OperationState &result, Value operand
    }]>,
  ];

  let extraClassDeclaration = [{
    RankedShapeType getRankedShape() {
      return shape().getType().cast<RankedShapeType>();
    }
  }];
}

// TODO(silvasean): What if the shape is an error shape?
def Shape_ToExtentTensorOp : Shape_PureOp<"to_extent_tensor",
     [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Convert a ranked shape to a tensor of extents.";
  let description = [{
    Convert a !shapex.ranked_shape to a rank-1 tensor of integers.

    Examples:
    %t0 = "shapex.to_extent_tensor"(%rs0)
      : (!shapex.ranked_shape<[3,?,5]>)
      -> tensor<3xi32>
    The resulting tensor will, for example, have elements [3,4,5] if the
    dynamic dimension is 4 at runtime.
  }];
  let arguments = (ins Shape_RankedShape:$shape);
  let results = (outs Shape_ExtentTensor:$extent_tensor);

  // TODO: Custom parser/printer
  let parser = ?;
  let printer = ?;
}

def Shape_FromExtentTensorOp : Shape_PureOp<"from_extent_tensor",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Convert a tensor of extents to a ranked shape.";
  let description = [{
    Convert a rank-1 tensor of integers to a !shapex.ranked_shape.

    Examples:
    %t0 = "shapex.from_dimension_tensor"(%rs0)
      : (tensor<3xi32>)
      -> !shapex.ranked_shape<[?,?,?],i32>
  }];
  let arguments = (ins Shape_ExtentTensor:$extent_tensor);
  let results = (outs Shape_RankedShape:$shape);

  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    // Declaration for overridden method from InferTypeOpInterface.
    static bool isCompatibleReturnTypes(ArrayRef<Type> lhs, ArrayRef<Type> rhs);
  }];

  // TODO: Custom parser/printer
  let parser = ?;
  let printer = ?;
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
      OpBuilder &builder, OperationState &result, Type type
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
      OpBuilder &builder, OperationState &result,
      Type dimType, Value shape, int index
    }]>,
    // dimType is defaulted to IndexType.
    OpBuilder<[{
      OpBuilder &builder, OperationState &result,
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
      OpBuilder &builder, OperationState &result, Type dimType, Value shape
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, Value shape
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
                       I64ElementsAttr:$broadcast_dimensions);
  let results = (outs AnyRankedTensor:$result);

  // TODO: Custom parser/printer
  let parser = ?;
  let printer = ?;
}

//===----------------------------------------------------------------------===//
// Shape manipulations.
//===----------------------------------------------------------------------===//

def Shape_GatherExtentsOp : Shape_PureOp<"gather_extents",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Gather extents across shapes.";
  let description = [{
    Gathers extents across the !shapex.ranked_shape's in `shapes`.

    This op conceptually performs the following operation:
    1. The extents of all shapes in `shapes` are concatenated together into
       a single list.
    2. The resulting shape is constructed by extracting extents from the
       combined list according to `indices`.
    In pseudocode:
    ```
    shapes = ... # a list of lists of extents
    # Example: shapes = [[3,-1],[2,7]]
    extents = [extent for extent in shape for shape in shapes]
    # Example: extents = [3,-1,2,7]
    # or to use another terminology: `extents = flatmap(shapes)`
    results = [extents[index] for index in indices]
    ```

    A large class of shape manipulations can be canonicalized into this op,
    including:
    - taking slices of shapes
    - concatenating shapes
    - permuting shapes
    The intuition behind this op is that eventually each extent will be
    exploded into its own SSA value. At which point, this op merely becomes
    and identification of each SSA value of the output extents with an
    SSA value of the input extents.
    This op has the useful property that is closed under composition with
    itself, thus allowing an arbitrarily complex subgraph consisting of just
    this op to be folded together.

    Some examples of shape transfer functions captured with this op:

    - Taking the last two extents of a shape:
      - [d0,d1,d2,d3] indices=[2,3] -> [d2,d3]
    - Concatenating three shapes:
      - [d0,d1] [d2,d3] [d4,d5] indices=[0,1,2,3,4,5] -> [d0,d1,d2,d3,d4,d5]
    - Shape transfer function for transpose with permutation [0,2,1]:
      - [d0,d1,d2] indices=[0,2,1] -> [d0,d2,d1]
    - Shape transfer function for outer product of a vector with itself:
      - Initial state: [d0] [d1] indices=[0,1] -> [d0,d1]
      - Canonicalized to a single-operand op after observing that both inputs
        are the same !shapex.ranked_shape value: [d0] indices=[0,0] -> [d0,d0]
    - Shape transfer function for matmul with a batch dimension on the LHS:
      - [d0,d1,d2] [d4,d5] indices=[0,1,2,5] -> [d0,d1,d2,d5]

    This op is somewhat inspired by the LLVM `shufflevector` instruction.

    Possible future pretty syntax for single-arg case:
    %rs = shapex.gather_extents %0[0,2,1] : !shapex.ranked_shape<[5,6,7]>
    Consider a pretty syntax for "concat":
    %rs = shapex.gather_extents concat(%0, %1) : !shapex.ranked_shape<[5,6,7]>, !shapex.ranked_shape<[8,9]>

  }];
  let arguments = (ins
    Variadic<Shape_RankedShape>:$shapes,
    I64ElementsAttr:$indices
  );
  let results = (outs
    Shape_RankedShape:$result
  );

  let verifier = [{ return verify$cppClass(*this); }];

  // TODO: Custom parser/printer
  let parser = ?;
  let printer = ?;

  let extraClassDeclaration = [{
    static SmallVector<int64_t, 6> getConcatenatedExtents(ValueRange values);
  }];
}


#endif  // IREE_DIALECT_SHAPE_OPS
