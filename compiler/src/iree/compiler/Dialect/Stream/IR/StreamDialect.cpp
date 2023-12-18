// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"

namespace mlir::iree_compiler::IREE::Stream {

namespace {

// Used to control inlining behavior.
struct StreamInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
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
};

struct StreamFolderInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  bool shouldMaterializeInto(Region *region) const override {
    // TODO(benvanik): redirect constants to the region scope when small.
    return false;
  }
};

// Tries to fold away unrealized_conversion_cast ops if the downstream consumers
// don't need the extra information. These are inserted during conversion or
// transforms that may interop with external dialects.
//
// Specifically matches:
//   %0 = builtin.unrealized_conversion_cast %arg0, %arg1 :
//        !stream.resource<transient>, index to !stream.resource<transient>
struct StripResourceConversionCastPattern
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(UnrealizedConversionCastOp castOp,
                                PatternRewriter &rewriter) const override {
    auto result = castOp.getResult(0);
    if (!llvm::isa<IREE::Stream::ResourceType>(result.getType()))
      return failure();
    assert(castOp.getNumOperands() == 2 &&
           "expect resource, index -> resource");
    auto resourceValue = castOp.getOperand(0);
    auto sizeValue = castOp.getOperand(1);
    for (auto &use : llvm::make_early_inc_range(result.getUses())) {
      if (auto sizeOp =
              dyn_cast<IREE::Stream::ResourceSizeOp>(use.getOwner())) {
        rewriter.replaceOp(sizeOp, sizeValue);
      } else {
        rewriter.updateRootInPlace(use.getOwner(),
                                   [&]() { use.set(resourceValue); });
      }
    }
    rewriter.eraseOp(castOp);
    return success();
  }
};

} // namespace

StreamDialect::StreamDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<StreamDialect>()) {
  context->loadDialect<IREE::Util::UtilDialect>();
  context->loadDialect<mlir::complex::ComplexDialect>();

  registerAttributes();
  registerTypes();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/Stream/IR/StreamOps.cpp.inc"
      >();

  addInterfaces<StreamInlinerInterface>();
  addInterfaces<StreamFolderInterface>();
}

void StreamDialect::getCanonicalizationPatterns(
    RewritePatternSet &results) const {
  results.insert<StripResourceConversionCastPattern>(getContext());
}

Operation *StreamDialect::materializeConstant(OpBuilder &builder,
                                              Attribute value, Type type,
                                              Location loc) {
  if (mlir::func::ConstantOp::isBuildableWith(value, type)) {
    return builder.create<mlir::func::ConstantOp>(
        loc, type, llvm::cast<FlatSymbolRefAttr>(value));
  } else if (arith::ConstantOp::isBuildableWith(value, type)) {
    return builder.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(value));
  } else if (llvm::isa<IREE::Stream::TimepointAttr>(value)) {
    return builder.create<IREE::Stream::TimepointImmediateOp>(loc);
  }
  return nullptr;
}

} // namespace mlir::iree_compiler::IREE::Stream
