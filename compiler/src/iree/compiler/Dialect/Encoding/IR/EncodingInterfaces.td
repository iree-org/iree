// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_ENCODING_INTERFACES
#define IREE_DIALECT_ENCODING_INTERFACES

include "iree/compiler/Dialect/Encoding/IR/EncodingBase.td"

// The interface is a subset of LinalgStructuredInterface.
def EncodingInterface : OpInterface<"EncodingOp"> {
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

  let verify = [{ return detail::verifyEncodingOpInterface($_op); }];
}

#endif  // IREE_DIALECT_ENCODING_INTERFACES
