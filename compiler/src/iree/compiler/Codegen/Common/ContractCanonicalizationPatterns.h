// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_CONTRACTCANONICALIZATIONPATTERNS_H_
#define IREE_COMPILER_CODEGEN_COMMON_CONTRACTCANONICALIZATIONPATTERNS_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

void populateContractLayoutCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    PatternBenefit benefit);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_CONTRACTCANONICALIZATIONPATTERNS_H_
