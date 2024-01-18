// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

void populateHALBufferViewToVMPatterns(MLIRContext *context,
                                       SymbolTable &importSymbols,
                                       TypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewCreateOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewAssertOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.assert");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewBufferOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.buffer");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewElementTypeOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.element_type");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewEncodingTypeOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.encoding_type");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewRankOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.rank");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewDimOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.dim");
  patterns.insert<VMImportOpConversion<IREE::HAL::BufferViewTraceOp>>(
      context, importSymbols, typeConverter, "hal.buffer_view.trace");
}

} // namespace mlir::iree_compiler
