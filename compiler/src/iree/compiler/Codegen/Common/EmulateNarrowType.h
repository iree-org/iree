// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_COMPILER_CODEGEN_COMMON_EMULATENARROWTYPE_H_
#define IREE_COMPILER_CODEGEN_COMMON_EMULATENARROWTYPE_H_

#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace mlir::iree_compiler {
using NarrowTypeConversionPopulationFn =
    std::function<void(arith::NarrowTypeEmulationConverter &,
                       RewritePatternSet &, ConversionTarget &)>;
LogicalResult emulateNarrowType(Operation *root,
                                std::optional<NarrowTypeConversionPopulationFn>
                                    populateCallback = std::nullopt);
} // namespace mlir::iree_compiler
#endif // IREE_COMPILER_CODEGEN_COMMON_EMULATENARROWTYPE_H_
