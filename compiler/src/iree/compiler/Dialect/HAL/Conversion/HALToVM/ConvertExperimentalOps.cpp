// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

void populateHALExperimentalToVMPatterns(MLIRContext *context,
                                         SymbolTable &importSymbols,
                                         TypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::ExFileFromMemoryOp>>(
      context, importSymbols, typeConverter, "hal.ex.file.from_memory");
}

} // namespace mlir::iree_compiler
