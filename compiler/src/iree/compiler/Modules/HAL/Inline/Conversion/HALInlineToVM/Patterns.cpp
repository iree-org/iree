// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/HAL/Inline/Conversion/HALInlineToVM/Patterns.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

void populateHALInlineToVMPatterns(MLIRContext *context,
                                   ConversionTarget &conversionTarget,
                                   TypeConverter &typeConverter,
                                   SymbolTable &importSymbols,
                                   RewritePatternSet &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::Inline::BufferAllocateOp>>(
      context, importSymbols, typeConverter, "hal_inline.buffer.allocate");
  patterns.insert<
      VMImportOpConversion<IREE::HAL::Inline::BufferAllocateInitializedOp>>(
      context, importSymbols, typeConverter,
      "hal_inline.buffer.allocate.initialized");
  patterns.insert<VMImportOpConversion<IREE::HAL::Inline::BufferWrapOp>>(
      context, importSymbols, typeConverter, "hal_inline.buffer.wrap");
  patterns.insert<VMImportOpConversion<IREE::HAL::Inline::BufferSubspanOp>>(
      context, importSymbols, typeConverter, "hal_inline.buffer.subspan");
  patterns.insert<VMImportOpConversion<IREE::HAL::Inline::BufferLengthOp>>(
      context, importSymbols, typeConverter, "hal_inline.buffer.length");
  patterns.insert<VMImportOpConversion<IREE::HAL::Inline::BufferStorageOp>>(
      context, importSymbols, typeConverter, "hal_inline.buffer.storage");

  patterns.insert<VMImportOpConversion<IREE::HAL::Inline::BufferViewCreateOp>>(
      context, importSymbols, typeConverter, "hal_inline.buffer_view.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::Inline::BufferViewAssertOp>>(
      context, importSymbols, typeConverter, "hal_inline.buffer_view.assert");
  patterns.insert<VMImportOpConversion<IREE::HAL::Inline::BufferViewBufferOp>>(
      context, importSymbols, typeConverter, "hal_inline.buffer_view.buffer");
  patterns
      .insert<VMImportOpConversion<IREE::HAL::Inline::BufferViewElementTypeOp>>(
          context, importSymbols, typeConverter,
          "hal_inline.buffer_view.element_type");
  patterns.insert<
      VMImportOpConversion<IREE::HAL::Inline::BufferViewEncodingTypeOp>>(
      context, importSymbols, typeConverter,
      "hal_inline.buffer_view.encoding_type");
  patterns.insert<VMImportOpConversion<IREE::HAL::Inline::BufferViewRankOp>>(
      context, importSymbols, typeConverter, "hal_inline.buffer_view.rank");
  patterns.insert<VMImportOpConversion<IREE::HAL::Inline::BufferViewDimOp>>(
      context, importSymbols, typeConverter, "hal_inline.buffer_view.dim");
  patterns.insert<VMImportOpConversion<IREE::HAL::Inline::BufferViewTraceOp>>(
      context, importSymbols, typeConverter, "hal_inline.buffer_view.trace");

  patterns.insert<VMImportOpConversion<IREE::HAL::Inline::DeviceQueryOp>>(
      context, importSymbols, typeConverter, "hal_inline.device.query.i64");
}

} // namespace mlir::iree_compiler
