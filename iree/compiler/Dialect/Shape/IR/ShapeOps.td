// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_SHAPE_OPS
#define IREE_DIALECT_SHAPE_OPS

include "iree/compiler/Dialect/Shape/IR/ShapeBase.td"
include "iree/compiler/Dialect/Shape/IR/ShapeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
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
    Shape_ShapeCarryingOpInterface,
    AllTypesMatch<["operand", "result"]>,
    DeclareOpInterfaceMethods<ViewLikeOpInterface>,
  ]> {
  let summary = "Ties a tensor and a shape together.";
  let description = [{
    Ties a specific tensor and its shape together in the IR, allowing further
    conversions to re-associate the two. This has no runtime implication and
    will be removed late in conversion.

    Usage:
      %0 = shapex.tie_shape %1, %2 : tensor<...>, shapex.ranked_shape<...>
  }];

  let arguments = (ins AnyType:$operand, Shape_RankedShape:$shape);
  let results = (outs AnyType:$result);

  let assemblyFormat = "operands attr-dict `:` type($operand) `,` type($shape)";

  let extraClassDeclaration = [{
    // ShapeCarryingInterface:
    Value buildOperandRankedShape(unsigned, OpBuilder) { return shape(); }
    Value buildResultRankedShape(unsigned, OpBuilder) { return shape(); }
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
      %0 = shapex.get_ranked_shape %arg0 : tensor<2x?xf32> ->
          !shapex.ranked_shape<[2,?]>  // based on index type
      %0 = shapex.get_ranked_shape %arg0 : tensor<2x?xf32> ->
          !shapex.ranked_shape<[2,?],i32>  // explicit dim type

    Canonicalization: This op includes a canonicalization pattern such that
    if its operand is supplied by a tie_shape op, then it will replace itself
    with the tie_shape's shape() operand. In this way, a function with all
    shapes materialized and tied to intermediate tensors should canonicalize
    to contain no get_ranked_shape ops.

    Any get_ranked_shape on a fully static shape will canonicalize to a const
    with unit value:
      %0 = constant_ranked_shape : !shapex.ranked_shape<[1,2],i32>
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
    OpBuilder<(ins "Value":$operand)>,
  ];

  let extraClassDeclaration = [{
    RankedShapeType getRankedShape() {
      return shape().getType().cast<RankedShapeType>();
    }
  }];
}

def Shape_ConstRankedShapeOp : Shape_PureOp<"const_ranked_shape",
    [ConstantLike, DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "A constant ranked_shape.";
  let description = [{
    Holds a RankedShape value. Note that it is only legal to store a constant
    RankedShape that is fully static, as anything more specific should be
    in the type, not have dims represented as const SSA values.

    Usage:
      %0 = shapex.const_ranked_shape : !shapex.ranked_shape<[1,2]>
  }];

  let arguments = (ins);
  let results = (outs Shape_RankedShape:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$type)>,
  ];
  let verifier = [{ return verify$cppClass(*this); }];
}

def Shape_MakeRankedShapeOp : Shape_PureOp<"make_ranked_shape"> {
  let summary = "Makes a ranked_shape from individual dims.";
  let description = [{
    Given a list of SSA values holding compatible dims, makes a corresponding
    ranked_shape.

    Usage:
      %0 = shapex.make_ranked_shape %dim0, %dim1 : (i32, i32) ->
          !shapex.ranked_shape<[?,?,128]>

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
      %0 = shapex.const ranked_shape : !shapex.ranked_shape<[1,2]>
      %1 = shapex.ranked_dim %0[0] : !shapex.ranked_shape<[1,2]> -> i32
  }];

  let arguments = (ins Shape_RankedShape:$shape,
                   APIntAttr:$index);
  let results = (outs Shape_DimType:$result);
  let verifier = [{ return verify$cppClass(*this); }];

  let builders = [
    OpBuilder<(ins "Type":$dimType, "Value":$shape, "int":$index)>,
    // dimType is defaulted to IndexType.
    OpBuilder<(ins "Value":$shape, "int":$index)>,
  ];

  let extraClassDeclaration = [{
    RankedShapeType getRankedShapeType() {
      return shape().getType().cast<RankedShapeType>();
    }
    unsigned getIndex() {
      auto index = getOperation()->getAttrOfType<IntegerAttr>("index");
      return index.getValue().getZExtValue();
    }
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

#endif  // IREE_DIALECT_SHAPE_OPS
