// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SAMPLES_CUSTOM_MODULES_DIALECT_CONVERSION_PATTERNS_H_
#define IREE_SAMPLES_CUSTOM_MODULES_DIALECT_CONVERSION_PATTERNS_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Custom {

// Populates conversion patterns from the tensor-based custom dialect ops to the
// HAL buffer-based ones.
void populateCustomToHALPatterns(MLIRContext *context,
                                 OwningRewritePatternList &patterns,
                                 TypeConverter &typeConverter);

// Populates conversion patterns from the custom dialect to the VM dialect.
void populateCustomToVMPatterns(MLIRContext *context,
                                SymbolTable &importSymbols,
                                OwningRewritePatternList &patterns,
                                TypeConverter &typeConverter);

}  // namespace Custom
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_SAMPLES_CUSTOM_MODULES_DIALECT_CONVERSION_PATTERNS_H_
