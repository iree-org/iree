// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_custom_modules/dialect/conversion_patterns.h"

#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree_custom_modules/dialect/custom_dialect.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Custom {

void populateCustomToHALPatterns(MLIRContext *context,
                                 RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  // We can use the HAL conversion handler for this tensor->buffer conversion
  // as we just want the simple form. If we wanted to perform additional
  // verification or have a specific use case (such as a place where only the
  // buffer is required and the shape is not) we could add our own.
  patterns.insert<HALOpConversion<TensorToMessageOp, BufferToMessageOp>>(
      context, typeConverter);
  patterns.insert<HALOpConversion<MessageToTensorOp, MessageToBufferOp>>(
      context, typeConverter);
}

void populateCustomToVMPatterns(MLIRContext *context,
                                SymbolTable &importSymbols,
                                RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  // We can use the VM conversion handler for all of these as they are simple
  // 1:1 mappings. More complex mappings can provide their own conversions
  // (such as the HAL dialect does).
  patterns.insert<VMImportOpConversion<IREE::Custom::BufferToMessageOp>>(
      context, importSymbols, typeConverter, "custom.buffer_to_message");
  patterns.insert<VMImportOpConversion<IREE::Custom::MessageToBufferOp>>(
      context, importSymbols, typeConverter, "custom.message_to_buffer");
  patterns.insert<VMImportOpConversion<IREE::Custom::PrintOp>>(
      context, importSymbols, typeConverter, "custom.print");
  patterns.insert<VMImportOpConversion<IREE::Custom::ReverseOp>>(
      context, importSymbols, typeConverter, "custom.reverse");
  patterns.insert<VMImportOpConversion<IREE::Custom::GetUniqueMessageOp>>(
      context, importSymbols, typeConverter, "custom.get_unique_message");
}

}  // namespace Custom
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
