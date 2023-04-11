// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_INPUTCONVERSION_STABLEHLO_REWRITERS_H_
#define IREE_COMPILER_INPUTCONVERSION_STABLEHLO_REWRITERS_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::stablehlo {

/// Populates the patterns that convert from StableHLO to Linalg on tensors.
void populateStableHloToLinalgConversionPatterns(MLIRContext* context,
                                                 TypeConverter& typeConverter,
                                                 RewritePatternSet* patterns,
                                                 bool enablePrimitiveOps);

}  // namespace mlir::iree_compiler::stablehlo

#endif  // IREE_COMPILER_INPUTCONVERSION_STABLEHLO_REWRITERS_H_
