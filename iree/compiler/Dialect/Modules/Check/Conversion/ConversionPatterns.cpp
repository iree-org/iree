// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Modules/Check/Conversion/ConversionPatterns.h"

#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/Modules/Check/IR/CheckOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Check {

void populateCheckToVMPatterns(MLIRContext *context, SymbolTable &importSymbols,
                               RewritePatternSet &patterns,
                               TypeConverter &typeConverter) {
  patterns.insert<VMImportOpConversion<IREE::Check::ExpectTrueOp>>(
      context, importSymbols, typeConverter, "check.expect_true");
  patterns.insert<VMImportOpConversion<IREE::Check::ExpectFalseOp>>(
      context, importSymbols, typeConverter, "check.expect_false");
  patterns.insert<VMImportOpConversion<IREE::Check::ExpectAllTrueOp>>(
      context, importSymbols, typeConverter, "check.expect_all_true");
  patterns.insert<VMImportOpConversion<IREE::Check::ExpectEqOp>>(
      context, importSymbols, typeConverter, "check.expect_eq");
  patterns.insert<VMImportOpConversion<IREE::Check::ExpectAlmostEqOp>>(
      context, importSymbols, typeConverter, "check.expect_almost_eq");
}

void populateCheckToHALPatterns(MLIRContext *context,
                                RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  // The same op handles both tensors and buffer views.
  patterns
      .insert<HALOpConversion<IREE::Check::ExpectAllTrueOp,
                              IREE::Check::ExpectAllTrueOp>,
              HALOpConversion<IREE::Check::ExpectEqOp, IREE::Check::ExpectEqOp>,
              HALOpConversion<IREE::Check::ExpectAlmostEqOp,
                              IREE::Check::ExpectAlmostEqOp>>(context,
                                                              typeConverter);
}

}  // namespace Check
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
