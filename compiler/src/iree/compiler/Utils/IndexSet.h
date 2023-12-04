// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_INDEXSET_H_
#define IREE_COMPILER_UTILS_INDEXSET_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"

namespace mlir {
namespace iree_compiler {

// Simple cache for generated index values.
// Always inserts at the location specified by the builder when constructed.
class IndexSet {
public:
  explicit IndexSet(Location loc, OpBuilder builder)
      : loc(loc), builder(builder) {}

  Value get(int64_t value) {
    auto it = memoizedIndices.find(value);
    if (it != memoizedIndices.end())
      return it->second;
    auto memoizedValue =
        builder.create<arith::ConstantIndexOp>(loc, value).getResult();
    memoizedIndices[value] = memoizedValue;
    return memoizedValue;
  }
  Value get(APInt value) { return get(value.getSExtValue()); }

  void populate(ValueRange values) {
    for (auto value : values) {
      APInt intValue;
      if (matchPattern(value, m_ConstantInt(&intValue))) {
        memoizedIndices.insert(std::make_pair(intValue.getSExtValue(), value));
      }
    }
  }

private:
  Location loc;
  OpBuilder builder;
  DenseMap<int64_t, Value> memoizedIndices;
};

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_UTILS_INDEXSET_H_
