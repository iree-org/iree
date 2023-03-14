// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_ADTEXTRAS_H_
#define IREE_COMPILER_UTILS_ADTEXTRAS_H_

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

namespace mlir::iree_compiler {

template <typename FirstIteratee, typename... RestIteratees>
auto enumerate_zip_equal(FirstIteratee&& first, RestIteratees&&... rest) {
  size_t numElements =
      std::distance(llvm::adl_begin(first), llvm::adl_end(first));
  return llvm::zip_equal(llvm::seq(static_cast<size_t>(0), numElements),
                         std::forward<FirstIteratee>(first),
                         std::forward<RestIteratees>(rest)...);
}

}  // namespace mlir::iree_compiler

#endif  // IREE_COMPILER_UTILS_ADT_EXTRAS_H_
