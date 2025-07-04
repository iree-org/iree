// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/ShapeUtils.h"

namespace mlir::iree_compiler {

bool compareShapesEqual(ShapedType lhsType, ValueRange lhsDynamicDims,
                        ShapedType rhsType, ValueRange rhsDynamicDims) {
  if (lhsType.hasStaticShape() && rhsType.hasStaticShape()) {
    // Static shape equivalence means we can fast-path the check.
    return lhsType == rhsType;
  }
  if (lhsType.getRank() != rhsType.getRank()) {
    return false;
  }
  unsigned dynamicDimIndex = 0;
  unsigned numNonmatchingSSADims = 0;
  for (unsigned i = 0; i < lhsType.getRank(); ++i) {
    if (lhsType.isDynamicDim(i) != rhsType.isDynamicDim(i)) {
      // Static/dynamic dimension mismatch - definitely differ.
      return false;
    } else if (lhsType.isDynamicDim(i)) {
      unsigned j = dynamicDimIndex++;
      if (lhsDynamicDims[j] != rhsDynamicDims[j]) {
        numNonmatchingSSADims++;
      }
    } else {
      if (lhsType.getDimSize(i) != rhsType.getDimSize(i)) {
        // Static dimensions differ.
        return false;
      }
    }
  }
  return numNonmatchingSSADims <= 1;
}

} // namespace mlir::iree_compiler
