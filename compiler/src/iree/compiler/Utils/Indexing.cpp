// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/Indexing.h"

using namespace mlir;

namespace mlir::iree_compiler {
LogicalResult basisFromSizesStrides(ArrayRef<int64_t> sizes,
                                    ArrayRef<int64_t> strides,
                                    SmallVectorImpl<int64_t> &basis,
                                    SmallVectorImpl<size_t> &dimToResult) {
  assert(sizes.size() == strides.size());
  size_t numDims = sizes.size();
  basis.reserve(numDims);

  SmallVector<std::tuple<int64_t, int64_t, size_t>> terms =
      llvm::map_to_vector(llvm::enumerate(strides, sizes), [&](auto tuple) {
        auto [dim, stride, size] = tuple;
        return std::make_tuple(stride, size, dim);
      });
  llvm::sort(terms);

  int64_t previousSizes = 1;
  SmallVector<std::optional<size_t>> basisEntryToDim;
  basisEntryToDim.reserve(numDims);
  for (auto [stride, size, dim] : terms) {
    if (stride == 0) {
      stride = 1;
      size = 1;
    }
    if (stride % previousSizes != 0)
      return failure();

    // Handle casis like threads = {4, 8}, strides = {1, 16}, which need an
    // extra basis element.
    if (stride != previousSizes) {
      int64_t jumpSize = stride / previousSizes;
      basisEntryToDim.push_back(std::nullopt);
      basis.push_back(jumpSize);
      previousSizes *= jumpSize;
    }

    basisEntryToDim.push_back(dim);
    basis.push_back(size);
    previousSizes *= size;
  }

  // Post-process. The basis is backwards and the permutation
  // we've constructed is the inverse of what we need.
  std::reverse(basis.begin(), basis.end());
  size_t basisLength = basis.size();
  dimToResult.assign(numDims, ~0);
  for (auto [reverseBasisPos, dimPos] : llvm::enumerate(basisEntryToDim)) {
    if (!dimPos)
      continue;
    // There's an extra overflow term at the front of the delineraize results,
    // so this subtraction lands in the [1, basisLength] range we need it
    // to be in.
    dimToResult[*dimPos] = basisLength - reverseBasisPos;
  }
  return success();
}
} // namespace mlir::iree_compiler
