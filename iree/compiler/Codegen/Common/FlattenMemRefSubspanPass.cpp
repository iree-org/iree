// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- FlattenMemRefSubspanPass.cpp - Flatten n-D MemRef subspan ----------===//
//
// This file implements a pass to flatten n-D MemRef subspan ops to 1-D MemRef
// ones and folds the byte offsets on subspan ops to the consumer load/store
// ops, in preparation for lowering to the final target.
//
// This pass is needed because of how MemRef is used by subspan ops:
//
// 1) Normally MemRef should capture the mapping to the underlying buffer with
// its internal strides and offsets. However, although subspan ops in IREE are
// subview-like constructs, they carry the offset directly on the ops themselves
// and return MemRefs with the identity layout map. This is due to that IREE can
// perform various optimizations over buffer allocation and decide, for example,
// to use the same underlying buffer for two MemRefs, which are converted form
// disjoint tensors initially.
// 2) The byte offset on subspan ops is an offset into the final planned 1-D
// byte buffer, while the MemRef can be n-D without considering a backing
// buffer and its data layout.
//
// So to bridge the gap, we need to linearize the MemRef dimensions to bring it
// onto the same view as IREE: buffers are just a bag of bytes. Then we need to
// fold the byte offset on subspan ops to the consumer load/store ops, so that
// we can rely on transformations in MLIR core, because they assume MemRefs map
// to the underlying buffers with its internal strides and offsets.
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

/// Returns true if the given `type` is a MemRef of rank 0 or 1.
static bool isRankZeroOrOneMemRef(Type type) {
  if (auto memrefType = type.dyn_cast<MemRefType>()) {
    return memrefType.hasRank() && memrefType.getRank() <= 1;
  }
  return false;
}

/// Flattens n-D MemRef to 1-D MemRef and allows other types.
struct FlattenMemRefTypeConverter final : public TypeConverter {
  FlattenMemRefTypeConverter() {
    // Allow all other types.
    addConversion([](Type type) -> Optional<Type> { return type; });

    // Convert n-D MemRef to 1-D MemRef.
    addConversion([](MemRefType type) -> Optional<Type> {
      // 1-D MemRef types are okay.
      if (isRankZeroOrOneMemRef(type)) return type;

      // We can only handle static strides and offsets for now; fail the rest.
      int64_t offset;
      SmallVector<int64_t, 4> strides;
      if (failed(getStridesAndOffset(type, strides, offset))) return Type();

      // Convert to a MemRef with unknown dimension. This is actually more akin
      // to how IREE uses memref types: they are for representing a view from a
      // byte buffer with potentially unknown total size, as transformation
      // passes can concatenate buffers, etc.
      return MemRefType::get(ShapedType::kDynamicSize, type.getElementType(),
                             ArrayRef<AffineMap>(), type.getMemorySpace());
    });
  }
};

//===----------------------------------------------------------------------===//
// Flattening Patterns
//===----------------------------------------------------------------------===//

/// Flattens memref global ops with more than 1 dimensions to 1 dimension.
struct FlattenGlobal final : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  static Attribute flattenAttribute(Attribute value, ShapedType newType) {
    if (!value) return value;
    if (auto splatAttr = value.dyn_cast<SplatElementsAttr>()) {
      return splatAttr.reshape(newType);
    } else if (auto denseAttr = value.dyn_cast<DenseElementsAttr>()) {
      return denseAttr.reshape(newType);
    }
    return {};
  }

  LogicalResult matchAndRewrite(
      memref::GlobalOp globalOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto oldType = globalOp.type().dyn_cast<MemRefType>();
    if (!oldType || !oldType.getAffineMaps().empty()) return failure();

    auto tensorType = RankedTensorType::get({oldType.getNumElements()},
                                            oldType.getElementType());
    auto memRefType =
        MemRefType::get({oldType.getNumElements()}, oldType.getElementType(),
                        {}, oldType.getMemorySpace());
    auto newInitialValue =
        flattenAttribute(globalOp.initial_valueAttr(), tensorType);
    rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        globalOp, globalOp.sym_name(), globalOp.sym_visibilityAttr(),
        memRefType, newInitialValue, globalOp.constant());
    return success();
  }
};

/// Flattens memref global load ops with more than 1 dimensions to 1 dimension.
struct FlattenGetGlobal final
    : public OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::GetGlobalOp getOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto oldType = getOp.getType().dyn_cast<MemRefType>();
    if (!oldType || !oldType.getAffineMaps().empty()) return failure();

    auto globalOp = dyn_cast_or_null<memref::GlobalOp>(
        SymbolTable::lookupNearestSymbolFrom(getOp, getOp.nameAttr()));
    if (!globalOp) return failure();

    auto loadedValue = rewriter.createOrFold<memref::GetGlobalOp>(
        getOp.getLoc(), globalOp.type(), getOp.nameAttr());

    auto newType = getTypeConverter()->convertType(oldType).cast<ShapedType>();
    rewriter.replaceOpWithNewOp<memref::CastOp>(getOp, newType, loadedValue);
    return success();
  }
};

/// Flattens memref subspan ops with more than 1 dimensions to 1 dimension.
struct FlattenBindingSubspan final
    : public OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceBindingSubspanOp subspanOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto oldType = subspanOp.getType().dyn_cast<MemRefType>();
    // IREE subspan ops only use memref types with the default identity
    // layout maps.
    if (!oldType || !oldType.getAffineMaps().empty()) return failure();

    Type newType = getTypeConverter()->convertType(oldType);

    rewriter.replaceOpWithNewOp<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp, newType, subspanOp.binding(), subspanOp.byte_offset(),
        subspanOp.byte_length());
    return success();
  }
};

/// Generates IR to perform index linearization with the given `indices`
/// indexing into the given memref `sourceType`.
static Value linearizeIndices(MemRefType sourceType, ValueRange indices,
                              Location loc, OpBuilder &builder) {
  assert(sourceType.hasRank() && sourceType.getRank() != 0);

  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(sourceType, strides, offset)) ||
      llvm::is_contained(strides, MemRefType::getDynamicStrideOrOffset()) ||
      offset == MemRefType::getDynamicStrideOrOffset()) {
    return nullptr;
  }

  AffineExpr sym0, sym1, sym2;
  bindSymbols(builder.getContext(), sym0, sym1, sym2);
  MLIRContext *context = builder.getContext();
  auto mulAddMap = AffineMap::get(0, 3, {sym0 * sym1 + sym2}, context);

  Value linearIndex = builder.create<ConstantIndexOp>(loc, offset);
  for (auto pair : llvm::zip(indices, strides)) {
    Value stride = builder.create<ConstantIndexOp>(loc, std::get<1>(pair));
    linearIndex = builder.create<AffineApplyOp>(
        loc, mulAddMap, ValueRange{std::get<0>(pair), stride, linearIndex});
  }
  return linearIndex;
}

/// Linearizes indices in memref.load ops.
struct LinearizeLoadIndices final : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::LoadOp loadOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    memref::LoadOp::Adaptor adaptor(operands);
    if (!isRankZeroOrOneMemRef(adaptor.memref().getType())) {
      return rewriter.notifyMatchFailure(
          loadOp, "expected converted memref of rank <= 1");
    }

    Value linearIndex = linearizeIndices(
        loadOp.getMemRefType(), loadOp.getIndices(), loadOp.getLoc(), rewriter);
    if (!linearIndex) {
      return loadOp.emitOpError() << "failed to linearize index";
    }

    rewriter.replaceOpWithNewOp<memref::LoadOp>(loadOp, adaptor.memref(),
                                                linearIndex);
    return success();
  }
};

/// Linearizes indices in memref.store ops.
struct LinearizeStoreIndices final
    : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::StoreOp storeOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    memref::StoreOp::Adaptor adaptor(operands);
    if (!isRankZeroOrOneMemRef(adaptor.memref().getType())) {
      return rewriter.notifyMatchFailure(
          storeOp, "expected converted memref of rank <= 1");
    }

    Value linearIndex =
        linearizeIndices(storeOp.getMemRefType(), storeOp.getIndices(),
                         storeOp.getLoc(), rewriter);
    if (!linearIndex) {
      return storeOp.emitOpError() << "failed to linearize index";
    }

    rewriter.replaceOpWithNewOp<memref::StoreOp>(storeOp, adaptor.value(),
                                                 adaptor.memref(), linearIndex);
    return success();
  }
};

/// Linearizes indices in vector.transfer_read ops.
struct LinearizeTransferReadIndices final
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      vector::TransferReadOp transferReadOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!transferReadOp.permutation_map().isMinorIdentity()) {
      return rewriter.notifyMatchFailure(
          transferReadOp, "cannot convert op with non-minor identity map");
    }
    vector::TransferReadOp::Adaptor adaptor(
        operands, transferReadOp->getAttrDictionary());
    if (!isRankZeroOrOneMemRef(adaptor.source().getType())) {
      return rewriter.notifyMatchFailure(
          transferReadOp, "expected converted memref of rank <= 1");
    }
    Value linearIndex = linearizeIndices(
        transferReadOp.getShapedType().cast<MemRefType>(),
        transferReadOp.indices(), transferReadOp.getLoc(), rewriter);
    if (!linearIndex) {
      return transferReadOp.emitOpError() << "failed to linearize index";
    }

    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        transferReadOp, transferReadOp.getVectorType(), adaptor.source(),
        linearIndex, rewriter.getDimIdentityMap(), transferReadOp.padding(),
        transferReadOp.in_boundsAttr());
    return success();
  }
};

/// Linearizes indices in vector.transfer_write ops.
struct LinearizeTransferWriteIndices final
    : public OpConversionPattern<vector::TransferWriteOp> {
  using OpConversionPattern<vector::TransferWriteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      vector::TransferWriteOp transferWriteOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!transferWriteOp.permutation_map().isMinorIdentity()) {
      return rewriter.notifyMatchFailure(
          transferWriteOp, "cannot convert op with non-minor identity map");
    }
    vector::TransferWriteOp::Adaptor adaptor(
        operands, transferWriteOp->getAttrDictionary());
    if (!isRankZeroOrOneMemRef(adaptor.source().getType())) {
      return rewriter.notifyMatchFailure(
          transferWriteOp, "expected converted memref of rank <= 1");
    }
    Value linearIndex = linearizeIndices(
        transferWriteOp.getShapedType().cast<MemRefType>(),
        transferWriteOp.indices(), transferWriteOp.getLoc(), rewriter);
    if (!linearIndex) {
      return transferWriteOp.emitOpError() << "failed to linearize index";
    }

    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        transferWriteOp, adaptor.vector(), adaptor.source(), linearIndex,
        rewriter.getDimIdentityMap(), transferWriteOp.in_boundsAttr());
    return success();
  }
};

/// Adjusts unrealized_conversion_cast ops' inputs to flattened memref values.
struct AdjustConversionCast final
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      UnrealizedConversionCastOp castOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (castOp->getNumOperands() != 1) return failure();

    Value input = operands.front();
    if (!isRankZeroOrOneMemRef(input.getType())) {
      return rewriter.notifyMatchFailure(
          castOp, "expected converted memref of rank <= 1");
    }
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        castOp, castOp.getResultTypes(), input);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Folding Patterns
//===----------------------------------------------------------------------===//

/// Returns the number of bytes of the given `type`. Returns llvm::None if
/// cannot deduce.
///
/// Note that this should be kept consistent with how the byte offset was
/// calculated in the subspan ops!
Optional<int64_t> getNumBytes(Type type) {
  if (type.isIntOrFloat()) return (type.getIntOrFloatBitWidth() + 7) / 8;
  if (auto vectorType = type.dyn_cast<VectorType>()) {
    auto elementBytes = getNumBytes(vectorType.getElementType());
    if (!elementBytes) return llvm::None;
    return elementBytes.getValue() * vectorType.getNumElements();
  }
  return llvm::None;
}

/// Folds the byte offset on subspan ops into the consumer load/store ops.
template <typename OpType>
struct FoldSubspanOffsetIntoLoadStore final : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    auto memrefType = op.memref().getType().template cast<MemRefType>();
    if (!isRankZeroOrOneMemRef(memrefType)) {
      return rewriter.notifyMatchFailure(op, "expected 0-D or 1-D memref");
    }

    auto subspanOp =
        op.memref()
            .template getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp) return failure();

    // If the subspan op has a zero byte offset then we are done.
    if (matchPattern(subspanOp.byte_offset(), m_Zero())) return failure();
    // byte length is unsupported for now.
    if (subspanOp.byte_length()) {
      return rewriter.notifyMatchFailure(op, "byte length unsupported");
    }

    // Calculate the offset we need to add to the load/store op, in terms of how
    // many elements.
    Optional<int64_t> numBytes = getNumBytes(memrefType.getElementType());
    if (!numBytes) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot deduce element byte count");
    }
    // Create a new subspan op with zero byte offset.
    Value zero = rewriter.create<ConstantIndexOp>(op.memref().getLoc(), 0);
    Value newSubspan = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
        op.memref().getLoc(), subspanOp.getType(), subspanOp.binding(), zero,
        subspanOp.byte_length());

    MLIRContext *context = rewriter.getContext();
    AffineExpr sym0, sym1;
    bindSymbols(context, sym0, sym1);
    auto addMap = AffineMap::get(0, 2, {sym0 + sym1}, context);
    auto divMap = AffineMap::get(0, 2, {sym0.floorDiv(sym1)}, context);

    Value byteValue = rewriter.create<ConstantIndexOp>(op.memref().getLoc(),
                                                       numBytes.getValue());
    // We assume that upper layers guarantee the byte offset is perfectly
    // divisible by the element byte count so the content is well aligned.
    Value offset = rewriter.create<AffineApplyOp>(
        op.getLoc(), divMap, ValueRange{subspanOp.byte_offset(), byteValue});

    // Get the new index by adding the old index with the offset.
    Value newIndex = rewriter.create<AffineApplyOp>(
        op.getLoc(), addMap, ValueRange{op.indices().front(), offset});

    if (std::is_same<OpType, memref::LoadOp>::value) {
      rewriter.replaceOpWithNewOp<memref::LoadOp>(
          op, memrefType.getElementType(), ValueRange{newSubspan, newIndex});
    } else {
      rewriter.replaceOpWithNewOp<memref::StoreOp>(
          op, TypeRange{}, ValueRange{op.getOperand(0), newSubspan, newIndex});
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct FlattenMemRefSubspanPass
    : public FlattenMemRefSubspanBase<FlattenMemRefSubspanPass> {
  FlattenMemRefSubspanPass() {}
  FlattenMemRefSubspanPass(const FlattenMemRefSubspanPass &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    // First flatten the dimensions of subspan op and their consumer load/store
    // ops. This requires setting up conversion targets with type converter.

    MLIRContext &context = getContext();
    FlattenMemRefTypeConverter typeConverter;
    RewritePatternSet flattenPatterns(&context);
    flattenPatterns.add<FlattenGlobal, FlattenGetGlobal, FlattenBindingSubspan,
                        LinearizeLoadIndices, LinearizeStoreIndices,
                        LinearizeTransferReadIndices,
                        LinearizeTransferWriteIndices, AdjustConversionCast>(
        typeConverter, &context);

    ConversionTarget target(context);
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addDynamicallyLegalOp<memref::GlobalOp>(
        [](memref::GlobalOp globalOp) {
          return isRankZeroOrOneMemRef(globalOp.type());
        });
    target.addDynamicallyLegalOp<memref::GetGlobalOp>(
        [](memref::GetGlobalOp getOp) {
          return isRankZeroOrOneMemRef(getOp.getType());
        });
    target.addDynamicallyLegalOp<IREE::HAL::InterfaceBindingSubspanOp>(
        [](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
          return isRankZeroOrOneMemRef(subspanOp.getType());
        });
    target.addDynamicallyLegalOp<memref::LoadOp>([](memref::LoadOp loadOp) {
      return isRankZeroOrOneMemRef(loadOp.getMemRefType());
    });
    target.addDynamicallyLegalOp<memref::StoreOp>([](memref::StoreOp storeOp) {
      return isRankZeroOrOneMemRef(storeOp.getMemRefType());
    });
    target.addDynamicallyLegalOp<vector::TransferReadOp>(
        [](vector::TransferReadOp readOp) {
          return isRankZeroOrOneMemRef(
              readOp.source().getType().cast<MemRefType>());
        });
    target.addDynamicallyLegalOp<vector::TransferWriteOp>(
        [](vector::TransferWriteOp writeOp) {
          return isRankZeroOrOneMemRef(
              writeOp.source().getType().cast<MemRefType>());
        });
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [](UnrealizedConversionCastOp castOp) {
          return castOp->getNumOperands() == 1 &&
                 isRankZeroOrOneMemRef(castOp->getOperandTypes().front());
        });

    // Use partial conversion here so that we can ignore allocations created by
    // promotion and their load/store ops.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(flattenPatterns)))) {
      return signalPassFailure();
    }

    // Then fold byte offset on subspan ops into consumer load/store ops.

    RewritePatternSet foldPatterns(&context);
    foldPatterns.add<FoldSubspanOffsetIntoLoadStore<memref::LoadOp>,
                     FoldSubspanOffsetIntoLoadStore<memref::StoreOp>>(&context);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(foldPatterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createFlattenMemRefSubspanPass() {
  return std::make_unique<FlattenMemRefSubspanPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
