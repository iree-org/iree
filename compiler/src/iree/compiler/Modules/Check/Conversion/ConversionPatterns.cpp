// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Modules/Check/Conversion/ConversionPatterns.h"

#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/HAL/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
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
  LogicalResult
  matchAndRewrite(T op, typename T::Adaptor adaptor,
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
    if (!results.has_value())
      return failure();
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

// Attempts to rewrite an op that may use tensor values into an op using HAL
// buffers.
static LogicalResult applyDefaultCheckBufferRewrite(
    Operation *srcOp, ValueRange operands, StringRef dstOpName,
    TypeConverter &typeConverter, ConversionPatternRewriter &rewriter) {
  OperationState state{srcOp->getLoc(), dstOpName};
  state.addAttributes(srcOp->getAttrs());

  // Add device argument.
  Value device = rewriter.create<IREE::HAL::ExSharedDeviceOp>(srcOp->getLoc());
  state.addOperands({device});

  for (auto [srcOperand, dstOperand] :
       llvm::zip_equal(srcOp->getOperands(), operands)) {
    // Check that any type that should have been mapped to buffer view was.
    // This is just to catch conflicts in type conversions that may sneak in
    // during development.
    assert(
        (!HALTypeConverter::shouldConvertToBufferView(srcOperand.getType()) ||
         dstOperand.getType().isa<IREE::HAL::BufferViewType>()) &&
        "expect that tensors have been mapped to buffer views");
    state.addOperands({dstOperand});
  }
  for (auto resultType : srcOp->getResultTypes()) {
    if (HALTypeConverter::shouldConvertToBufferView(resultType)) {
      state.addTypes(IREE::HAL::BufferViewType::get(rewriter.getContext()));
    } else {
      // Normal pass-through result.
      if (failed(typeConverter.convertType(resultType, state.types))) {
        return failure();
      }
    }
  }

  auto *dstOp = rewriter.create(state);
  rewriter.replaceOp(srcOp, dstOp->getResults());
  return success();
}

// HAL tensor-to-buffer conversion utility.
// This can be used by dialects to model custom op conversion from a dialect
// that uses the MLIR tensor type to the IREE HAL buffer type. At this point
// during conversion the source values will be TensorType and the target values
// will be IREE::HAL::BufferTypes. Any static information available about the
// tensor (such as static dimensions, element type, layout, etc) are extracted
// here and lowered as expanded values.
//
// The ABI is currently very basic and will change with the introduction of more
// dynamic shape logic.
//
// Source:
//   my.tensor_op(%arg0 : tensor<2x4xf32>)
// Target:
//   %arg0_view = hal.buffer_view.create %arg0, ...
//   my.buffer_op(%arg0_view : !hal.buffer_view)
template <typename SRC, typename DST>
class HALCheckOpConversion : public OpConversionPattern<SRC> {
public:
  HALCheckOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern<SRC>(context), typeConverter(typeConverter) {}

  LogicalResult
  matchAndRewrite(SRC srcOp, typename SRC::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return applyDefaultCheckBufferRewrite(srcOp, adaptor.getOperands(),
                                          DST::getOperationName(),
                                          typeConverter, rewriter);
  }

protected:
  TypeConverter &typeConverter;
};

void populateCheckToHALPatterns(MLIRContext *context,
                                RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  // The same op handles both tensors and buffer views.
  patterns.insert<
      HALCheckOpConversion<IREE::Check::ExpectAllTrueOp,
                           IREE::Check::ExpectAllTrueOp>,
      HALCheckOpConversion<IREE::Check::ExpectEqOp, IREE::Check::ExpectEqOp>,
      HALCheckOpConversion<IREE::Check::ExpectAlmostEqOp,
                           IREE::Check::ExpectAlmostEqOp>>(context,
                                                           typeConverter);
}

} // namespace Check
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
