// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/LLVMLowerings.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#define DEBUG_TYPE "iree-codegen-expand-strided-metadata"

namespace mlir::iree_compiler {

namespace {

struct AssumeAlignmentOpLowering
    : public ConvertOpToLLVMPattern<IREE::Codegen::AssumeAlignmentOp> {
  using ConvertOpToLLVMPattern<
      IREE::Codegen::AssumeAlignmentOp>::ConvertOpToLLVMPattern;
  explicit AssumeAlignmentOpLowering(const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<IREE::Codegen::AssumeAlignmentOp>(converter) {}

  LogicalResult
  matchAndRewrite(IREE::Codegen::AssumeAlignmentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value memref = adaptor.getSource();
    unsigned alignment = op.getAlignment();
    auto loc = op.getLoc();

    auto srcMemRefType = cast<MemRefType>(op.getSource().getType());
    Value ptr = getStridedElementPtr(loc, srcMemRefType, memref, /*indices=*/{},
                                     rewriter);

    // TODO: Handle non-power of 2 alignments.
    if (!llvm::isPowerOf2_32(alignment)) {
      rewriter.eraseOp(op);
      return success();
    }

    // Emit llvm.intr.ptrmask(memref, -alignment)) for power of 2
    // alignments.
    MemRefDescriptor memRefDescriptor(memref);
    auto intPtrType =
        getIntPtrType(memRefDescriptor.getElementPtrType().getAddressSpace());
    // If alignment is a power of 2, -alignment will be
    // 0b11...1[0 x log_2(alignment)] by 2's complement which is exactly the
    // mask we want.
    int64_t maskVal = -alignment;
    Value mask = createIndexAttrConstant(rewriter, loc, intPtrType, maskVal);
    StringAttr intrName = rewriter.getStringAttr("llvm.ptrmask");
    Value newPtrValue = rewriter
                            .create<LLVM::CallIntrinsicOp>(
                                loc, memRefDescriptor.getElementPtrType(),
                                intrName, ValueRange{ptr, mask})
                            .getResult(0);
    memRefDescriptor.setAllocatedPtr(rewriter, op.getLoc(), newPtrValue);
    rewriter.replaceOp(op, (Value)memRefDescriptor);

    return success();
  }
};

} // namespace

void populateConvertIREECodegenToLLVMPatterns(
    const LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<AssumeAlignmentOpLowering>(typeConverter);
}

} // namespace mlir::iree_compiler
