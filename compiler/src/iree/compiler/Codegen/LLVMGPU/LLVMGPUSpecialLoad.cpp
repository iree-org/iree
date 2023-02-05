// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/ConvertToLLVM.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-llvmgpu-special-load-store"

namespace mlir {
namespace iree_compiler {

namespace {

/// Insert nontemporal to memref.load if the subspanOp has
/// DescriptorFlags::Streaming flag.
struct MemrefLoadSetNonTemporal : public OpRewritePattern<memref::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::LoadOp op,
                                PatternRewriter &rewriter) const override {
    if (auto subspanOp =
            (op.getMemRef()
                 .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>())) {
      if (!op.getNontemporal() &&
          bitEnumContainsAny(subspanOp.getDescriptorFlags().value_or(
                                 IREE::HAL::DescriptorFlags::None),
                             IREE::HAL::DescriptorFlags::Streaming)) {
        rewriter.replaceOpWithNewOp<memref::LoadOp>(
            op, op->getResultTypes(), op.getMemRef(), op.getIndices(), true);
        return success();
      }
    }
    return failure();
  }
};

/// Insert nontemporal to memref.store if the subspanOp has
/// DescriptorFlags::WriteOnly flag
struct MemrefStoreSetNonTemporal : public OpRewritePattern<memref::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::StoreOp op,
                                PatternRewriter &rewriter) const override {
    if (auto subspanOp =
            (op.getMemRef()
                 .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>())) {
      if (!op.getNontemporal() &&
          bitEnumContainsAny(subspanOp.getDescriptorFlags().value_or(
                                 IREE::HAL::DescriptorFlags::None),
                             IREE::HAL::DescriptorFlags::WriteOnly)) {
        rewriter.replaceOpWithNewOp<memref::StoreOp>(
            op, op.getValue(), op.getMemRef(), op.getIndices(), true);
      }
    }
    return failure();
  }
};

/// Generate load PTX for llvm.load {nontemporal}. This is the only way since
/// cache eviction suffixes are not implemented in llvm core.
struct GenerateLoadPTX : public OpRewritePattern<LLVM::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LoadOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getNontemporal()) return failure();
    auto gepOp = op.getAddr().getDefiningOp<LLVM::GEPOp>();
    if (!gepOp) return failure();

    const char *asmStr = "ld.global.nc.L1::no_allocate.f32 $0, [$1];\n";
    const char *asmConstraints = "=f, l";
    SmallVector<Value> asmVals{gepOp->getResults()};
    auto asmDialectAttr = LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                                    LLVM::AsmDialect::AD_ATT);
    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op, op.getRes().getType(),
        /*operands=*/asmVals,
        /*asm_string=*/asmStr,
        /*constraints=*/asmConstraints, /*has_side_effects=*/true,
        /*is_align_stack=*/false, /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/ArrayAttr());

    return success();
  }
};

/// Generate load PTX for llvm.store {nontemporal}.
struct GenerateStorePTX : public OpRewritePattern<LLVM::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::StoreOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getNontemporal()) return failure();

    auto gepOp = op.getAddr().getDefiningOp<LLVM::GEPOp>();
    if (!gepOp) return failure();

    const char *asmStr = "st.global.L1::no_allocate.f32 [$0], $1;\n";
    const char *asmConstraints = "l, f";
    SmallVector<Value> asmVals{gepOp->getResults().front(), op.getValue()};
    auto asmDialectAttr = LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                                    LLVM::AsmDialect::AD_ATT);
    auto voidtype = LLVM::LLVMVoidType::get(getContext());

    rewriter.create<LLVM::InlineAsmOp>(
        op->getLoc(), voidtype,
        /*operands=*/asmVals,
        /*asm_string=*/asmStr,
        /*constraints=*/asmConstraints, /*has_side_effects=*/true,
        /*is_align_stack=*/false, /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/ArrayAttr());
    rewriter.eraseOp(op);
    return success();
  }
};
}  // namespace

void populateSpecialLoadStore(RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<MemrefLoadSetNonTemporal, MemrefStoreSetNonTemporal>(ctx);
}

void populateSpecialLoadStorePTX(RewritePatternSet &patterns,
                                 MLIRContext *ctx) {
  patterns.insert<GenerateLoadPTX, GenerateStorePTX>(ctx);
}

}  // namespace iree_compiler
}  // namespace mlir
