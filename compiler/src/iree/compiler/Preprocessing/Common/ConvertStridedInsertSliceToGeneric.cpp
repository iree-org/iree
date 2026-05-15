// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_CONVERTSTRIDEDINSERTSLICETOGENERICPASS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc"

namespace {

/// Convert a strided `tensor.insert_slice` into a zero-constant destination
/// to a `linalg.generic` with index arithmetic.
///
/// This targets the backward data convolution pattern where the upstream
/// gradient is scattered into a zero buffer at strided positions, producing
/// a separate Memset + slow_memcpy dispatch pair. The replacement generic
/// computes the strided scatter in a single pass and is potentially
/// fusable with consumer ops in later dispatch formation passes.
///
/// For each output position, the generic checks whether the position maps
/// to a valid source element (i.e., (pos - offset) is non-negative,
/// divisible by stride, and the quotient is in-bounds). Source indices are
/// clamped to valid range so the extract is always safe, and arith.select
/// chooses between the extracted value and zero.
class ConvertStridedInsertSliceToGeneric
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::InsertSliceOp op,
                                PatternRewriter &rewriter) const override {
    // Destination must be a zero splat constant.
    Value dest = op.getDest();
    Attribute destAttr;
    if (!matchPattern(dest, m_Constant(&destAttr))) {
      return failure();
    }
    auto splatAttr = dyn_cast<SplatElementsAttr>(destAttr);
    if (!splatAttr) {
      return failure();
    }
    Attribute splatVal = splatAttr.getSplatValue<Attribute>();
    bool isZero =
        TypeSwitch<Attribute, bool>(splatVal)
            .Case<FloatAttr>([](auto a) { return a.getValue().isZero(); })
            .Case<IntegerAttr>([](auto a) { return a.getValue().isZero(); })
            .Default([](auto) { return false; });
    if (!isZero) {
      return failure();
    }

    // All offsets, sizes, and strides must be static, with at least one
    // non-unit stride.
    SmallVector<int64_t> offsets(op.getStaticOffsets());
    SmallVector<int64_t> strides(op.getStaticStrides());
    SmallVector<int64_t> sizes(op.getStaticSizes());
    if (ShapedType::isDynamicShape(offsets) ||
        ShapedType::isDynamicShape(strides) ||
        ShapedType::isDynamicShape(sizes)) {
      return failure();
    }
    if (llvm::all_of(strides, [](int64_t s) { return s == 1; })) {
      return failure();
    }

    Value src = op.getSource();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto destTy = cast<RankedTensorType>(dest.getType());

    unsigned origRank = destTy.getRank();
    auto elemTy = destTy.getElementType();
    Location loc = op.getLoc();

    // A dim is "passthrough" if it has stride 1 and offset 0 — the scatter
    // generic just copies it without any index arithmetic.
    auto isPassthrough = [&](unsigned d) {
      return strides[d] == 1 && offsets[d] == 0;
    };

    // Skip conversion when too many non-batch passthrough elements exist
    // per spatial position. In grouped backward-data convolutions the
    // group+channel dims are passthrough; when their product is large the
    // scatter generic is slower than hardware DMA (Memset + slow_memcpy).
    {
      int64_t passthroughElems = 1;
      constexpr int64_t kMaxPassthroughElems = 256;
      for (unsigned i = 1; i < origRank; i++) {
        if (isPassthrough(i)) {
          passthroughElems *= destTy.getDimSize(i);
        }
      }
      if (passthroughElems > kMaxPassthroughElems) {
        return failure();
      }
    }

    // Collapse contiguous passthrough dims to reduce iteration rank.
    // E.g., grouped convs have trailing [G, C] dims that collapse to [G*C].
    // Without this, high-rank scatter generics (5D+) can fail to compile
    // in downstream passes and produce suboptimal GPU tiling.
    SmallVector<ReassociationIndices> reassociation;
    ReassociationIndices currentGroup = {0};
    for (unsigned i = 1; i < origRank; i++) {
      if (isPassthrough(i - 1) && isPassthrough(i)) {
        currentGroup.push_back(i);
      } else {
        reassociation.push_back(std::move(currentGroup));
        currentGroup = {static_cast<int64_t>(i)};
      }
    }
    reassociation.push_back(std::move(currentGroup));
    bool needsCollapse =
        llvm::any_of(reassociation, [](const auto &g) { return g.size() > 1; });

    // Compute collapsed metadata.
    unsigned rank = reassociation.size();
    SmallVector<int64_t> collapsedOffsets(rank), collapsedStrides(rank);
    SmallVector<int64_t> collapsedDestShape(rank), collapsedSrcShape(rank);
    for (auto [gi, group] : llvm::enumerate(reassociation)) {
      collapsedOffsets[gi] = offsets[group[0]];
      collapsedStrides[gi] = strides[group[0]];
      int64_t destDim = 1, srcDim = 1;
      for (int64_t idx : group) {
        destDim *= destTy.getDimSize(idx);
        srcDim *= srcTy.getDimSize(idx);
      }
      collapsedDestShape[gi] = destDim;
      collapsedSrcShape[gi] = srcDim;
    }
    auto collapsedSrcTy = RankedTensorType::get(collapsedSrcShape, elemTy);
    auto collapsedDestTy = RankedTensorType::get(collapsedDestShape, elemTy);

    Value collapsedSrc = src;
    if (needsCollapse) {
      collapsedSrc = tensor::CollapseShapeOp::create(
          rewriter, loc, collapsedSrcTy, src, reassociation);
    }

    // Build the scatter generic over the (collapsed) dest shape.
    Value empty =
        tensor::EmptyOp::create(rewriter, loc, collapsedDestShape, elemTy);
    AffineMap identityMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap> indexingMaps = {identityMap};
    SmallVector<utils::IteratorType> iterTypes(rank,
                                               utils::IteratorType::parallel);

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, collapsedDestTy, /*inputs=*/ValueRange{},
        /*outputs=*/ValueRange{empty}, indexingMaps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange /*args*/) {
          auto cstIdx = [&](int64_t v) {
            return arith::ConstantOp::create(b, loc, b.getIndexAttr(v));
          };

          Value zero = cstIdx(0);
          Value allValid = nullptr;
          SmallVector<Value> srcIndices;
          for (unsigned i = 0; i < rank; i++) {
            Value idx = linalg::IndexOp::create(b, loc, i);

            // Passthrough dims: source index == dest index.
            if (collapsedStrides[i] == 1 && collapsedOffsets[i] == 0) {
              srcIndices.push_back(idx);
              continue;
            }

            int64_t stride = collapsedStrides[i];
            int64_t srcSize = collapsedSrcTy.getDimSize(i);
            Value shifted =
                arith::SubIOp::create(b, loc, idx, cstIdx(collapsedOffsets[i]));

            Value strVal = cstIdx(stride);
            Value rem = arith::RemSIOp::create(b, loc, shifted, strVal);
            Value srcIdx = arith::DivSIOp::create(b, loc, shifted, strVal);

            // Valid when shifted >= 0 && rem == 0 && srcIdx < srcSize.
            Value dimValid = arith::AndIOp::create(
                b, loc,
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                                      shifted, zero),
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, rem,
                                      zero));
            dimValid = arith::AndIOp::create(
                b, loc, dimValid,
                arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, srcIdx,
                                      cstIdx(srcSize)));
            allValid = allValid
                           ? arith::AndIOp::create(b, loc, allValid, dimValid)
                           : dimValid;

            // Clamp to [0, srcSize-1] so the extract is always in-bounds.
            Value clamped = arith::MaxSIOp::create(b, loc, srcIdx, zero);
            clamped =
                arith::MinSIOp::create(b, loc, clamped, cstIdx(srcSize - 1));
            srcIndices.push_back(clamped);
          }

          Value zeroElem =
              arith::ConstantOp::create(b, loc, rewriter.getZeroAttr(elemTy));
          Value extracted =
              tensor::ExtractOp::create(b, loc, collapsedSrc, srcIndices);
          Value result = allValid ? arith::SelectOp::create(b, loc, allValid,
                                                            extracted, zeroElem)
                                  : extracted;
          linalg::YieldOp::create(b, loc, result);
        });

    Value result = genericOp.getResult(0);
    if (needsCollapse) {
      result = tensor::ExpandShapeOp::create(rewriter, loc, destTy, result,
                                             reassociation);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertStridedInsertSliceToGenericPass
    : impl::ConvertStridedInsertSliceToGenericPassBase<
          ConvertStridedInsertSliceToGenericPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<ConvertStridedInsertSliceToGeneric>(context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace
} // namespace mlir::iree_compiler::Preprocessing
