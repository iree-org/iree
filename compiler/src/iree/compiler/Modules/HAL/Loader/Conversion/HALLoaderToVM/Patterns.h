// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_MODULES_HAL_LOADER_CONVERSION_HALLOADERTOVM_PATTERNS_H_
#define IREE_COMPILER_MODULES_HAL_LOADER_CONVERSION_HALLOADERTOVM_PATTERNS_H_

#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

// Populates conversion patterns from the hal_loader dialect to the VM dialect.
void populateHALLoaderToVMPatterns(MLIRContext *context,
                                   ConversionTarget &conversionTarget,
                                   TypeConverter &typeConverter,
                                   SymbolTable &importSymbols,
                                   RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_MODULES_HAL_LOADER_CONVERSION_HALLOADERTOVM_PATTERNS_H_
