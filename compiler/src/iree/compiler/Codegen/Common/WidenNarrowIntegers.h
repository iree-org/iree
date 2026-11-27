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

// The populate functions below add patterns that compute operations on narrow
// integers in a wider integer type. A narrow integer is a scalar or vector
// integer type that matches `isNarrow` and is narrower than `wideWidthInBits`.
// Emulated storage may hold such a value in the low bits of a wider word, with
// undefined high bits. `isNarrow` is copied into the patterns; values it
// captures must remain valid while the patterns run.

// Populates patterns for operations whose result depends on the high bits of
// their operands, such as division, right shifts, minimum, maximum and signed
// comparisons. Each operand is first extended to `wideWidthInBits` bits by the
// operation's signedness; a narrow result is truncated back.
void populateWidenNarrowIntegerHighBitOpPatterns(
    RewritePatternSet &patterns, unsigned wideWidthInBits,
    const std::function<bool(Type)> &isNarrow);

// Populates bit-count widening independently of arithmetic emulation. Some
// targets have native narrow integers but require wider bit-count operands.
void populateWidenNarrowIntegerBitCountPatterns(
    RewritePatternSet &patterns, unsigned wideWidthInBits,
    const std::function<bool(Type)> &isNarrow);

// Populates patterns for operations whose low bits give the narrow result,
// such as addition and multiplication, computed at `wideWidthInBits` bits.
// Widened integer arithmetic drops narrow overflow flags, which do not
// describe its operands.
void populateWidenNarrowIntegerLowBitOpPatterns(
    RewritePatternSet &patterns, unsigned wideWidthInBits,
    const std::function<bool(Type)> &isNarrow);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_WIDENNARROWINTEGERS_H_
