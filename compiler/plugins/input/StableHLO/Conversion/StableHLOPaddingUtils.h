// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOPADDINGUTILS_H_
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOPADDINGUTILS_H_

#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler::stablehlo {

struct PaddedDimension {
  Value size;
  Value isNegative;
};

// Builds max(size + max(size - 1, 0) * interior + low + high, 0), preserving
// cancellation even when the dilated intermediate exceeds 64 bits. `size` is
// an index; padding operands are signless i64 storage with the given logical
// signedness. The input is trusted: the interior must be nonnegative and a
// nonnegative result must fit the index type, and no runtime check enforces
// either. `isNegative` lets a caller distinguish a fully cropped convolution
// from a valid pad.
PaddedDimension buildPaddedDimension(OpBuilder &builder, Location loc,
                                     Value size, Value low, Value high,
                                     Value interior, bool isUnsignedPadding);

} // namespace mlir::iree_compiler::stablehlo

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOPADDINGUTILS_H_
