// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_WIDENNARROWINTEGERS_H_
#define IREE_COMPILER_CODEGEN_COMMON_WIDENNARROWINTEGERS_H_

#include <functional>

#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

// Selectors are copied into the patterns; their captures must remain valid
// while the patterns run. Only scalar/vector integer types below the requested
// width are widened.
//
// Populates patterns that run ops reading the high bits of integers matching
// `isNarrow` at `wideWidthInBits` bits, on operands extended by their
// signedness.
void populateWidenNarrowIntegerHighBitOpPatterns(
    RewritePatternSet &patterns, unsigned wideWidthInBits,
    const std::function<bool(Type)> &isNarrow);

// Populates bit-count widening independently of arithmetic emulation. Some
// targets have native narrow integers but require wider bit-count operands.
void populateWidenNarrowIntegerBitCountPatterns(
    RewritePatternSet &patterns, unsigned wideWidthInBits,
    const std::function<bool(Type)> &isNarrow);

// Populates operations whose low bits give the narrow result. Widened integer
// arithmetic drops narrow overflow flags, which do not describe its operands.
void populateWidenNarrowIntegerLowBitOpPatterns(
    RewritePatternSet &patterns, unsigned wideWidthInBits,
    const std::function<bool(Type)> &isNarrow);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_WIDENNARROWINTEGERS_H_
