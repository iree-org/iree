// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Modules/Strings/Conversion/ConversionPatterns.h"

#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/Modules/Strings/IR/Ops.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Strings {

void populateStringsToHALPatterns(MLIRContext *context,
                                  OwningRewritePatternList &patterns,
                                  TypeConverter &typeConverter) {
  patterns.insert<HALOpConversion<IREE::Strings::ToStringTensorOp,
                                  IREE::Strings::ToStringTensorOp>>(
      context, typeConverter);
  patterns.insert<
      HALOpConversion<IREE::Strings::GatherOp, IREE::Strings::GatherOp>>(
      context, typeConverter);
}

void populateStringsToVMPatterns(MLIRContext *context,
                                 SymbolTable &importSymbols,
                                 OwningRewritePatternList &patterns,
                                 TypeConverter &typeConverter) {
  patterns.insert<VMImportOpConversion<IREE::Strings::I32ToStringOp>>(
      context, importSymbols, typeConverter, "strings.i32_to_string");
  patterns.insert<VMImportOpConversion<IREE::Strings::PrintOp>>(
      context, importSymbols, typeConverter, "strings.print");
  patterns.insert<VMImportOpConversion<IREE::Strings::ToStringTensorOp>>(
      context, importSymbols, typeConverter, "strings.to_string_tensor");
  patterns.insert<VMImportOpConversion<IREE::Strings::StringTensorToStringOp>>(
      context, importSymbols, typeConverter, "strings.string_tensor_to_string");
  patterns.insert<VMImportOpConversion<IREE::Strings::GatherOp>>(
      context, importSymbols, typeConverter, "strings.gather");
  patterns.insert<VMImportOpConversion<IREE::Strings::ConcatOp>>(
      context, importSymbols, typeConverter, "strings.concat");
}

}  // namespace Strings
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
