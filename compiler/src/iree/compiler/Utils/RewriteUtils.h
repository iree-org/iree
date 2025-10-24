// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_REWRITEUTILS_H_
#define IREE_COMPILER_UTILS_REWRITEUTILS_H_

#include "iree/compiler/Utils/Permutation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir::iree_compiler {

template <typename PermutationRange>
void permuteValues(RewriterBase &rewriter, Location loc, ValueRange values,
                   PermutationRange perm) {
  // If use RAUW(ValueRange -> ValueRange), replacements are applied
  // sequentially meaning the last value in the range replaces everything
  // because it overrides earlier replacements. To avoid this create an
  // unrealized conversion cast that we drop after replacement.
  SmallVector<Value> replacements;
  permute(values, perm, std::back_inserter(replacements));
  SmallVector<Type> types =
      llvm::map_to_vector(replacements, [](Value v) { return v.getType(); });
  auto replacementCast =
      UnrealizedConversionCastOp::create(rewriter, loc, types, replacements);

  for (auto [iv, r] : llvm::zip_equal(values, replacementCast.getResults())) {
    rewriter.replaceAllUsesExcept(iv, r, replacementCast);
  }

  // Erase the temporary unrealized conversion cast.
  rewriter.replaceOp(replacementCast, replacementCast.getOperands());
}

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_REWRITEUTILS_H_
