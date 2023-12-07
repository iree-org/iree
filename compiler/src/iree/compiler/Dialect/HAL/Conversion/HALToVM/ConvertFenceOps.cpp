// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

void populateHALFenceToVMPatterns(MLIRContext *context,
                                  SymbolTable &importSymbols,
                                  TypeConverter &typeConverter,
                                  RewritePatternSet &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::FenceCreateOp>>(
      context, importSymbols, typeConverter, "hal.fence.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::FenceJoinOp>>(
      context, importSymbols, typeConverter, "hal.fence.join");
  patterns.insert<VMImportOpConversion<IREE::HAL::FenceQueryOp>>(
      context, importSymbols, typeConverter, "hal.fence.query");
  patterns.insert<VMImportOpConversion<IREE::HAL::FenceSignalOp>>(
      context, importSymbols, typeConverter, "hal.fence.signal");
  patterns.insert<VMImportOpConversion<IREE::HAL::FenceFailOp>>(
      context, importSymbols, typeConverter, "hal.fence.fail");
  patterns.insert<VMImportOpConversion<IREE::HAL::FenceAwaitOp>>(
      context, importSymbols, typeConverter, "hal.fence.await");
}

} // namespace mlir::iree_compiler
