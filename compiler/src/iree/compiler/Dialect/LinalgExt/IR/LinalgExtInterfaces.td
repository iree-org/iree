// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_INTERFACES
#define IREE_DIALECT_LINALGEXT_INTERFACES

include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"

// The interface is a subset of LinalgStructuredInterface.
def LinalgExtInterface : OpInterface<"LinalgExtOp"> {
  let methods = [];

  let extraClassDeclaration = [{
    /// Returns the value that expresses the shape of the output in terms of
    /// shape of the input operands where possible.
    LogicalResult reifyResultShapes(OpBuilder &b,
        mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);

    //========================================================================//
    // Helper functions to mutate the `operand_segment_sizes` attribute.
    // These are useful when cloning and changing operand types.
    //========================================================================//
    void setNumInputs(unsigned num) { setOperandSegmentAt(0, num); }
    void setNumOutputBuffers(unsigned num) { setOperandSegmentAt(1, num); }

    private:
    void setOperandSegmentAt(unsigned idx, unsigned val) {
      auto attr = cast<DenseIntElementsAttr>(
        (*this)->getAttr("operand_segment_sizes"));
      unsigned i = 0;
      auto newAttr = attr.mapValues(IntegerType::get(getContext(), 32),
        [&](const APInt &v) { return (i++ == idx) ? APInt(32, val) : v; });
      getOperation()->setAttr("operand_segment_sizes", newAttr);
    }
  }];

  let verify = [{ return detail::verifyLinalgExtOpInterface($_op); }];
}

// Interface that allows for fusion of both LinalgExt and Linalg ops.
def LinalgFusionInterface : OpInterface<"LinalgFusionOpInterface"> {
  let methods = [
    //===------------------------------------------------------------------===//
    // Interface methods for fusion.
    //===------------------------------------------------------------------===//
    InterfaceMethod<
      /*desc=*/[{
        Return an AffineMap for each operand or nullptr if the operand
        does not have an indexing map representation.
      }],
      /*retTy=*/"SmallVector<AffineMap>",
      /*methodName=*/"getIndexingMapsForOperands",
      /*args=*/(ins),
      /*methodBody=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return an AffineMap for each result or nullptr if the result
        does not have an indexing map representation.
      }],
      /*retTy=*/"SmallVector<AffineMap>",
      /*methodName=*/"getIndexingMapsForResults",
      /*args=*/(ins),
      /*methodBody=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{
        For each operand/result return indexing map or nullptr if an
        operand or result does not have an indexing map representation.
      }],
      /*retTy=*/"SmallVector<AffineMap>",
      /*methodName=*/"getIndexingMapsArray",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto inputMaps = $_op.getIndexingMapsForOperands();
        llvm::append_range(inputMaps, $_op.getIndexingMapsForResults());
        return inputMaps;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the number of parallel loops.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumParallelLoops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return llvm::count($_op.getLoopIteratorTypes(), utils::IteratorType::parallel);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the total number of loops.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumLoops",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getLoopIteratorTypes().size();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the static loop ranges.
      }],
      /*retTy=*/"SmallVector<int64_t, 4>",
      /*methodName=*/"getStaticLoopRanges",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
          SmallVector<int64_t, 4> loopRanges;
          llvm::for_each($_op.getOperands(), [&](Value operand) {
          if (auto shapedType = dyn_cast<ShapedType>(operand.getType())) {
            llvm::append_range(loopRanges, shapedType.getShape());
            }
          });
          return loopRanges;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing map for an op's `result` or nullptr if
        the indexing map is not representable.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getIndexingMapMatchingResult",
      /*args=*/(ins "OpResult":$result),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(result.getOwner() == $_op);
        return $_op.getIndexingMapsForResults()[result.getResultNumber()];
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the indexing map for `opOperand` or nullptr if
        the indexing map is not representable.
      }],
      /*retTy=*/"AffineMap",
      /*methodName=*/"getMatchingIndexingMap",
      /*args=*/(ins "OpOperand*":$opOperand),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(opOperand->getOwner() == $_op);
        return getIndexingMapsArray()[opOperand->getOperandNumber()];
      }]
    >,


  ];
}

#endif  // IREE_DIALECT_LINALGEXT_INTERFACES
