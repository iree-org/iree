// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir::iree_compiler::IREE::Util {

// Used for custom printing support.
struct UtilOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  /// Hooks for getting an alias identifier alias for a given symbol, that is
  /// not necessarily a part of this dialect. The identifier is used in place of
  /// the symbol when printing textual IR. These aliases must not contain `.` or
  /// end with a numeric digit([0-9]+). Returns success if an alias was
  /// provided, failure otherwise.
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (auto compositeAttr = llvm::dyn_cast<CompositeAttr>(attr)) {
      os << "composite_of_" << compositeAttr.getTotalLength() << "b";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

// Used to control inlining behavior.
struct UtilInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Check the inlining policy specified on the callable first.
    if (auto inliningPolicy =
            callable->getAttrOfType<IREE::Util::InliningPolicyAttrInterface>(
                "inlining_policy")) {
      if (!inliningPolicy.isLegalToInline(call, callable))
        return false;
    }

    // Check any extended inlining policies that may come from dialect
    // attributes on both the callee and caller.
    for (auto attr : callable->getDialectAttrs()) {
      if (auto inliningPolicy =
              dyn_cast<IREE::Util::InliningPolicyAttrInterface>(
                  attr.getValue())) {
        if (!inliningPolicy.isLegalToInline(call, callable))
          return false;
      }
    }

    // Sure!
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    // Sure!
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    // Sure!
    return true;
  }

  void handleTerminator(Operation *op, Block *newDest) const final {
    auto returnOp = dyn_cast<IREE::Util::ReturnOp>(op);
    if (!returnOp)
      return;
    OpBuilder builder(op);
    builder.create<mlir::cf::BranchOp>(op->getLoc(), newDest,
                                       returnOp.getOperands());
    op->erase();
  }

  void handleTerminator(Operation *op, ValueRange valuesToReplace) const final {
    auto returnOp = dyn_cast<IREE::Util::ReturnOp>(op);
    if (!returnOp)
      return;
    assert(returnOp.getNumOperands() == valuesToReplace.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands())) {
      valuesToReplace[it.index()].replaceAllUsesWith(it.value());
    }
  }

  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const override {
    return nullptr;
  }
};

UtilDialect::UtilDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<UtilDialect>()) {
  context->loadDialect<arith::ArithDialect>();

  addInterfaces<UtilOpAsmInterface, UtilInlinerInterface>();

  registerAttributes();
  registerTypes();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Util/IR/UtilOps.cpp.inc"
      >();
}

Operation *UtilDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  if (isa<IREE::Util::NullAttr>(value)) {
    return builder.create<IREE::Util::NullOp>(loc, type);
  } else if (arith::ConstantOp::isBuildableWith(value, type)) {
    return builder.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(value));
  }
  return nullptr;
}

template <typename DimOp>
struct FoldDimOp : public OpRewritePattern<DimOp> {
  using OpRewritePattern<DimOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(DimOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.getSource();
    while (auto assumeAlignmentOp =
               source.getDefiningOp<memref::AssumeAlignmentOp>()) {
      source = assumeAlignmentOp.getViewSource();
    }
    auto shapeAwareOp =
        dyn_cast_or_null<ShapeAwareOpInterface>(source.getDefiningOp());
    if (!shapeAwareOp)
      return failure();

    // We only support static dimension indices today (as in general we only
    // support ranked shapes). If we find dynamic indices sneaking in we will
    // need to do something much more complex - or prevent them from sneaking
    // in.
    APInt index;
    if (!matchPattern(op.getIndex(), m_ConstantInt(&index))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-constant dim index unsupported");
    }

    // If it's a static dim then just fold to that.
    auto type = llvm::cast<ShapedType>(source.getType());
    int64_t staticDim = type.getDimSize(index.getZExtValue());
    if (!ShapedType::isDynamic(staticDim)) {
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, staticDim);
      return success();
    }

    // Otherwise try to get the dynamic dimension cheaply without the need to
    // insert new IR.
    unsigned dynamicIdx = type.getDynamicDimIndex(index.getZExtValue());
    auto dynamicDims = shapeAwareOp.getResultDynamicDimsFromValue(source);
    rewriter.replaceOp(op, dynamicDims[dynamicIdx]);

    return success();
  }
};

void UtilDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.insert<FoldDimOp<memref::DimOp>>(getContext());
  results.insert<FoldDimOp<tensor::DimOp>>(getContext());
}

} // namespace mlir::iree_compiler::IREE::Util
