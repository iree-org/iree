// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

void populateHALSemaphoreToVMPatterns(MLIRContext *context,
                                      SymbolTable &importSymbols,
                                      TypeConverter &typeConverter,
                                      RewritePatternSet &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::SemaphoreCreateOp>>(
      context, importSymbols, typeConverter, "hal.semaphore.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::SemaphoreQueryOp>>(
      context, importSymbols, typeConverter, "hal.semaphore.query");
  patterns.insert<VMImportOpConversion<IREE::HAL::SemaphoreSignalOp>>(
      context, importSymbols, typeConverter, "hal.semaphore.signal");
  patterns.insert<VMImportOpConversion<IREE::HAL::SemaphoreFailOp>>(
      context, importSymbols, typeConverter, "hal.semaphore.fail");
  patterns.insert<VMImportOpConversion<IREE::HAL::SemaphoreAwaitOp>>(
      context, importSymbols, typeConverter, "hal.semaphore.await");
}

}  // namespace iree_compiler
}  // namespace mlir
