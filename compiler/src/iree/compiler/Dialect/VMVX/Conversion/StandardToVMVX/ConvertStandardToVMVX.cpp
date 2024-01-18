// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VMVX/Conversion/StandardToVMVX/ConvertStandardToVMVX.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXOps.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXTypes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

/// Pattern to lower operations that become a no-ops at this level.
template <typename OpTy>
struct FoldAsNoOp final : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

/// Removes unrealized_conversion_cast ops introduced during progressive
/// lowering when possible.
struct RemoveIdentityConversionCast final
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() == 1 && op->getNumResults() == 1 &&
        adaptor.getOperands().front().getType() ==
            op->getResultTypes().front()) {
      rewriter.replaceOp(op, adaptor.getOperands());
      return success();
    }

    return failure();
  }
};

} // namespace

void populateStandardToVMVXPatterns(MLIRContext *context,
                                    RewritePatternSet &patterns,
                                    TypeConverter &typeConverter) {
  // We type/shape erase memrefs as we lower so there is no need for reshapes.
  patterns.insert<FoldAsNoOp<memref::CollapseShapeOp>>(typeConverter, context);
  patterns.insert<FoldAsNoOp<memref::ExpandShapeOp>>(typeConverter, context);

  patterns.insert<RemoveIdentityConversionCast>(typeConverter, context);
}

} // namespace mlir::iree_compiler
