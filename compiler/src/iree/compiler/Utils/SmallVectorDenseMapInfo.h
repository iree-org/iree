// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_SMALLVECTORDENSEMAPINFO_H_
#define IREE_COMPILER_UTILS_SMALLVECTORDENSEMAPINFO_H_

#include <numeric>
#include <utility>

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
template <typename T, unsigned N>
struct DenseMapInfo<SmallVector<T, N>> {
  static SmallVector<T, N> getEmptyKey() {
    return SmallVector<T, N>(1, llvm::DenseMapInfo<T>::getEmptyKey());
  }

  static SmallVector<T, N> getTombstoneKey() {
    return SmallVector<T, N>(1, llvm::DenseMapInfo<T>::getTombstoneKey());
  }

  static unsigned getHashValue(const SmallVector<T, N> &v) {
    hash_code hash = llvm::DenseMapInfo<T>::getHashValue(
        llvm::DenseMapInfo<T>::getEmptyKey());
    return std::accumulate(v.begin(), v.end(), hash,
                           [](hash_code hash, const T &element) {
                             return hash_combine(hash, element);
                           });
  }

  static bool isEqual(const SmallVector<T, N> &lhs,
                      const SmallVector<T, N> &rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }

    return llvm::all_of_zip(lhs, rhs, [](const T &lhs, const T &rhs) {
      return DenseMapInfo<T>::isEqual(lhs, rhs);
    });
  }
};
} // namespace llvm

#endif // IREE_COMPILER_UTILS_SMALLVECTORDENSEMAPINFO_H_
