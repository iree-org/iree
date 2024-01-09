// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

void populateHALDevicesToVMPatterns(MLIRContext *context,
                                    SymbolTable &importSymbols,
                                    TypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::DevicesCountOp>>(
      context, importSymbols, typeConverter, "hal.devices.count");
  patterns.insert<VMImportOpConversion<IREE::HAL::DevicesGetOp>>(
      context, importSymbols, typeConverter, "hal.devices.get");
}

} // namespace mlir::iree_compiler
