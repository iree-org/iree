// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

void populateHALExperimentalToVMPatterns(MLIRContext *context,
                                         SymbolTable &importSymbols,
                                         TypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::ExSharedDeviceOp>>(
      context, importSymbols, typeConverter, "hal.ex.shared_device");
}

}  // namespace iree_compiler
}  // namespace mlir
