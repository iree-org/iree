// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_MODULES_HAL_LOADER_CONVERSION_HALLOADER_CONVERTHALLOADERTOVM_H_
#define IREE_COMPILER_DIALECT_MODULES_HAL_LOADER_CONVERSION_HALLOADER_CONVERTHALLOADERTOVM_H_

#include "iree/compiler/Dialect/Modules/HAL/Loader/IR/HALLoaderOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Populates conversion patterns from the hal_loader dialect to the VM dialect.
void populateHALLoaderToVMPatterns(MLIRContext *context,
                                   ConversionTarget &conversionTarget,
                                   TypeConverter &typeConverter,
                                   SymbolTable &importSymbols,
                                   RewritePatternSet &patterns);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_MODULES_HAL_LOADER_CONVERSION_HALLOADER_CONVERTHALLOADERTOVM_H_
