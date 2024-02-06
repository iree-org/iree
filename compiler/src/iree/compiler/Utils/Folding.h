// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_FOLDING_H_
#define IREE_COMPILER_UTILS_FOLDING_H_

#include <iterator>
#include <utility>
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/OpDefinition.h"
namespace mlir::iree_compiler {

// Convert a `Value` or an `Attribute` range to a range of `OpFoldResult`.
template <typename Range, typename OutIt>
void toOpFoldResults(Range &&range, OutIt outIt) {
  llvm::transform(std::forward<Range>(range), outIt,
                  [](auto v) { return OpFoldResult(v); });
}

template <typename Range>
SmallVector<OpFoldResult> toOpFoldResults(Range &&range) {
  SmallVector<OpFoldResult> res;
  toOpFoldResults(std::forward<Range>(range), std::back_inserter(res));
  return res;
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_FOLDING_H_
