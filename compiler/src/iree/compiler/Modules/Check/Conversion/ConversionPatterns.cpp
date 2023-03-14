// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/Check/Conversion/ConversionPatterns.h"

#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Modules/Check/IR/CheckOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Check {

// Converts check ops to vm.call ops with handling for when the check module is
// not compiled in (we just ignore them). This allows us to run benchmarks on
// modules using the check ops.
template <typename T, typename Adaptor = typename T::Adaptor>
struct OptionalCheckImportConversion : public VMImportOpConversion<T, Adaptor> {
  using VMImportOpConversion<T, Adaptor>::VMImportOpConversion;
  LogicalResult matchAndRewrite(
      T op, typename T::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto hasImport = rewriter.create<IREE::VM::ImportResolvedOp>(
        op.getLoc(), rewriter.getI32Type(), this->importOp.getName());
    auto *followingBlock = rewriter.splitBlock(rewriter.getInsertionBlock(),
                                               rewriter.getInsertionPoint());
    auto *callBlock = rewriter.createBlock(followingBlock);
    rewriter.setInsertionPointAfter(hasImport);
    rewriter.create<IREE::VM::CondBranchOp>(op.getLoc(), hasImport, callBlock,
                                            followingBlock);
    rewriter.setInsertionPointToStart(callBlock);
    auto results = rewriteToCall(op, adaptor, this->importOp,
                                 *this->getTypeConverter(), rewriter);
    if (!results.has_value()) return failure();
    rewriter.replaceOp(op, results.value());
    rewriter.create<IREE::VM::BranchOp>(op.getLoc(), followingBlock);
    return success();
  }
};

void populateCheckToVMPatterns(MLIRContext *context, SymbolTable &importSymbols,
                               RewritePatternSet &patterns,
                               TypeConverter &typeConverter) {
  patterns.insert<OptionalCheckImportConversion<IREE::Check::ExpectTrueOp>>(
      context, importSymbols, typeConverter, "check.expect_true");
  patterns.insert<OptionalCheckImportConversion<IREE::Check::ExpectFalseOp>>(
      context, importSymbols, typeConverter, "check.expect_false");
  patterns.insert<OptionalCheckImportConversion<IREE::Check::ExpectAllTrueOp>>(
      context, importSymbols, typeConverter, "check.expect_all_true");
  patterns.insert<OptionalCheckImportConversion<IREE::Check::ExpectEqOp>>(
      context, importSymbols, typeConverter, "check.expect_eq");
  patterns.insert<OptionalCheckImportConversion<IREE::Check::ExpectAlmostEqOp>>(
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
