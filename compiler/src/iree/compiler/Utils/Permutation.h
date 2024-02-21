// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_PERMUTATION_H_
#define IREE_COMPILER_UTILS_PERMUTATION_H_

#include <iterator>
#include <type_traits>
#include <utility>

#include "llvm/ADT/ADL.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler {

// Example: values = (1, 2, 3), permutation = (2, 1, 0)
// output = (3, 2, 1).
// TODO: make applyPermutation at mlir/Dialect/Utils/IndexingUtils.h in MLIR
// generic and use it instead.
template <typename ValuesIt, typename PermutationRange, typename OutIt>
void permute(ValuesIt valuesBegin, ValuesIt valuesEnd,
             PermutationRange &&permutation, OutIt outBegin) {
  assert(std::distance(valuesBegin, valuesEnd) >= llvm::adl_size(permutation));
  llvm::transform(permutation, outBegin,
                  [valuesBegin](auto i) { return valuesBegin[i]; });
}

template <typename ValuesRange, typename PermutationRange, typename OutIt>
void permute(ValuesRange &&values, PermutationRange &&permutation,
             OutIt outBegin) {
  permute(llvm::adl_begin(std::forward<ValuesRange>(values)),
          llvm::adl_end(std::forward<ValuesRange>(values)), permutation,
          outBegin);
}

template <typename T, typename Index>
SmallVector<T> permute(ArrayRef<T> values, ArrayRef<Index> permutation) {
  SmallVector<T> res;
  permute(values, permutation, std::back_inserter(res));
  return res;
}

// Check if the range is a sequence of numbers starting from 0.
// Example: (0, 1, 2, 3).
// TODO: Make the isIdentityPermutation in MLIR more generic to not only
// accept int64_t and delete this.
template <typename Range>
bool isIdentityPermutation(Range &&range) {
  using ValueType = std::decay_t<decltype(*std::begin(range))>;
  ValueType i = static_cast<ValueType>(0);
  return llvm::all_of(std::forward<Range>(range), [&i](ValueType v) {
    bool res = (v == i);
    ++i;
    return res;
  });
}

// Make a permutation that moves src to dst.
// Example with size = 5, src = 1, dst = 3.
// output = (0, 2, 3, 1, 4).
// Example with size = 2, src = 0, dst = 1.
// output = (1, 0).
template <typename T, typename OutIt>
void makeMovePermutation(T size, T src, T dst, OutIt outBegin) {
  assert(src < size && dst < size && size > static_cast<T>(0));
  T outSize = 0;
  for (T i = 0; i < size; ++i) {
    if (outSize == dst) {
      *outBegin = src;
      ++outBegin;
      ++outSize;
    }
    if (i == src) {
      ++i;
      if (i >= size) {
        break;
      }
    }

    *outBegin = i;
    ++outBegin;
    ++outSize;
  }

  if (size != outSize) {
    *outBegin = src;
    ++outBegin;
  }
}

template <typename T>
SmallVector<T> makeMovePermutation(T size, T src, T dst) {
  SmallVector<T> res;
  res.reserve(size);
  makeMovePermutation(size, src, dst, std::back_inserter(res));
  return res;
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_PERMUTATION_H_
