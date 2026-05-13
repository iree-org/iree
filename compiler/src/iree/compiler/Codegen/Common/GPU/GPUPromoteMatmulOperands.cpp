// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPROMOTEMATMULOPERANDSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
/// Helper to insert copy with derived thread config.
Value promoteValue(OpBuilder &builder, Location loc, Value v,
                   bool useDirectLoad) {
  auto tensorType = cast<RankedTensorType>(v.getType());
  SmallVector<OpFoldResult> mixedSizes = tensor::getMixedSizes(builder, loc, v);

  Value empty = tensor::EmptyOp::create(builder, loc, mixedSizes,
                                        tensorType.getElementType());
  auto copy = linalg::CopyOp::create(builder, loc, v, empty);

  if (useDirectLoad) {
    setLoweringConfig(
        copy, IREE::GPU::UseGlobalLoadDMAAttr::get(builder.getContext()));
  } else {
    setLoweringConfig(
        copy, IREE::GPU::DerivedThreadConfigAttr::get(builder.getContext()));
  }
  return copy.getResult(0);
}

/// Helper to promote results. If the target value is consumed only by a
/// `tensor.extract_slice`, this will promote the result of the slice instead.
void promoteResult(OpBuilder &builder, Operation *op, Value valToMakeShared) {
  IRRewriter rewriter(builder);
  Location loc = op->getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfterValue(valToMakeShared);
  tensor::ExtractSliceOp extractSliceOp;
  SetVector<Operation *> opsToReplaceUseIn;
  Value valueToReplace = valToMakeShared;
  for (auto user : valToMakeShared.getUsers()) {
    extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    if (extractSliceOp) {
      // If the result is consumed by an extract_slice then we expect there to
      // be exactly one extract slice that is then consumed.
      // TODO (nirvedhmeshram) : This is fairly special case. Instead we should
      // just promote results before doing padding which introduces the extract
      // slice.
      if (!valToMakeShared.hasOneUse()) {
        return;
      }
      valueToReplace = extractSliceOp.getResult();
      for (auto user : extractSliceOp->getUsers()) {
        opsToReplaceUseIn.insert(user);
      }
      break;
    }
    opsToReplaceUseIn.insert(user);
  }
  auto tensorType = cast<RankedTensorType>(valToMakeShared.getType());
  if (!tensorType) {
    return;
  }
  SmallVector<Value> dynamicSizes;
  for (auto [idx, size] : llvm::enumerate(tensorType.getShape())) {
    if (ShapedType::isDynamic(size)) {
      dynamicSizes.push_back(
          tensor::DimOp::create(rewriter, loc, valToMakeShared, idx));
    }
  }
  Attribute addressSpace = gpu::AddressSpaceAttr::get(
      rewriter.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  auto alloc = bufferization::AllocTensorOp::create(rewriter, loc, tensorType,
                                                    dynamicSizes);
  alloc.setMemorySpaceAttr(addressSpace);
  auto copy =
      linalg::CopyOp::create(rewriter, loc, valToMakeShared, alloc.getResult());
  Value replacement = copy.getResult(0);

  // Insert a fusion barrier to prevent the copy to workgroup memory from fusing
  // into the promoted copy.
  replacement = IREE::Codegen::FusionBarrierOp::create(
      rewriter, replacement.getLoc(), replacement.getType(), replacement);

  // If in extract slice is present we make it consume the new copy.
  if (extractSliceOp) {
    extractSliceOp.getSourceMutable().assign(replacement);
    replacement = valueToReplace;
  }

  rewriter.setInsertionPointAfterValue(replacement);
  replacement =
      promoteValue(rewriter, loc, replacement, /*useDirectLoad=*/false);
  valueToReplace.replaceUsesWithIf(replacement, [&](OpOperand &use) {
    return opsToReplaceUseIn.contains(use.getOwner());
  });
}

/// Traces through tensor.extract_slice ops to find the
/// iree_codegen.load_from_buffer feeding this value. Returns the slice chain
/// (outermost first) and the load op, or failure if the pattern isn't matched.
static FailureOr<std::pair<SmallVector<tensor::ExtractSliceOp>,
                           IREE::Codegen::LoadFromBufferOp>>
findLoadFromBuffer(Value v) {
  SmallVector<tensor::ExtractSliceOp> slices;
  while (auto sliceOp = v.getDefiningOp<tensor::ExtractSliceOp>()) {
    slices.push_back(sliceOp);
    v = sliceOp.getSource();
  }
  auto loadOp = v.getDefiningOp<IREE::Codegen::LoadFromBufferOp>();
  if (!loadOp) {
    return failure();
  }
  return std::make_pair(slices, loadOp);
}

/// Promotes an operand for global_load_tr by copying to shared memory.
/// Strips fat_raw_buffer so the copy reads from a flat global pointer
/// (required by global_load_tr), then creates a linalg.generic that
/// iterates (K-outer, N-inner) so vectorization produces vector<1x8> reads
/// — 8 contiguous N-direction elements per lane. This is the access pattern
/// that global_load_tr_b128 requires.
/// The matmul's indexing map is updated to read from the transposed [N,K]
/// shared memory layout.
Value transposePromoteOperand(OpBuilder &builder, Operation *op,
                              unsigned index) {
  OpOperand &operand = op->getOpOperand(index);
  Location loc = op->getLoc();

  // Strip fat_raw_buffer if present so global_load_tr can use a flat pointer.
  Value sourceValue = operand.get();
  auto maybeLoad = findLoadFromBuffer(sourceValue);
  if (succeeded(maybeLoad)) {
    auto &[slices, loadOp] = *maybeLoad;
    if (auto castOp =
            loadOp.getBuffer().getDefiningOp<amdgpu::FatRawBufferCastOp>()) {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointAfter(loadOp);
      auto flatMemrefType = cast<MemRefType>(castOp.getSource().getType());
      auto flatTensorType = RankedTensorType::get(
          flatMemrefType.getShape(), flatMemrefType.getElementType());
      Value flatLoad = IREE::Codegen::LoadFromBufferOp::create(
          builder, loadOp.getLoc(), flatTensorType, castOp.getSource());
      sourceValue = flatLoad;
      for (auto sliceOp : llvm::reverse(slices)) {
        builder.setInsertionPointAfter(sliceOp);
        sourceValue = tensor::ExtractSliceOp::create(
            builder, sliceOp.getLoc(), sliceOp.getResultType(), sourceValue,
            sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
            sliceOp.getMixedStrides());
      }
    }
  }

  auto tensorType = cast<RankedTensorType>(sourceValue.getType());
  MLIRContext *ctx = op->getContext();

  // Create the transposed output buffer [N, K].
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, loc, sourceValue);
  SmallVector<OpFoldResult> transposedSizes(mixedSizes.rbegin(),
                                            mixedSizes.rend());
  Value empty = tensor::EmptyOp::create(builder, loc, transposedSizes,
                                        tensorType.getElementType());

  // linalg.generic with (d0=K outer, d1=N inner):
  //   input  map: (d0, d1) -> (d0, d1)  reads src[k, n]
  //   output map: (d0, d1) -> (d1, d0)  writes dst[n, k]
  // With N as the inner (vectorized) dimension, each thread reads
  // vector<1x8> (8 contiguous N elements at fixed K) — the correct
  // access pattern for global_load_tr_b128.
  // Loop iteration order: (d0=N outer, d1=K inner).
  // With K as the inner (fast-varying per-lane) dimension,
  // UseGlobalTransposeLoadAttr's tiling [N=vectorSize, K=1] maps K to
  // linear_dim_0 (fast thread dim): 8 consecutive lanes get 8 consecutive
  // K rows, which is the correct wave-level setup for global_load_tr.
  // Each lane reads vector<1x8> (8 contiguous N from its K row). The tag
  // UseGlobalTransposeLoadAttr drives the thread tiling level sizes.
  //   input  map (d0=N, d1=K) -> (d1, d0)  reads B[K, N]
  //   output map (d0=N, d1=K) -> (d0, d1)  writes alloc[N, K]
  AffineExpr d0 = builder.getAffineDimExpr(0); // N (outer)
  AffineExpr d1 = builder.getAffineDimExpr(1); // K (inner)
  AffineMap inputMap = AffineMap::get(2, 0, {d1, d0}, ctx);
  AffineMap outputMap = AffineMap::get(2, 0, {d0, d1}, ctx);
  SmallVector<utils::IteratorType> iterTypes(2, utils::IteratorType::parallel);

  auto copyOp = linalg::GenericOp::create(
      builder, loc, empty.getType(), sourceValue, empty,
      ArrayRef<AffineMap>{inputMap, outputMap}, iterTypes,
      [](OpBuilder &b, Location l, ValueRange args) {
        linalg::YieldOp::create(b, l, args[0]);
      });
  // Use UseGlobalTransposeLoadAttr as the lowering config so the tiling pass
  // produces K-inner thread assignment via getStaticTilingLevelSizes.
  setLoweringConfig(copyOp, IREE::GPU::UseGlobalTransposeLoadAttr::get(ctx));

  // Update the matmul's indexing map for this operand by reversing its
  // results to reflect the [N, K] shared memory layout.
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    SmallVector<AffineMap> maps(genericOp.getIndexingMapsArray());
    AffineMap oldMap = maps[index];
    SmallVector<AffineExpr> results(oldMap.getResults().rbegin(),
                                    oldMap.getResults().rend());
    maps[index] = AffineMap::get(oldMap.getNumDims(), oldMap.getNumSymbols(),
                                 results, ctx);
    genericOp.setIndexingMapsAttr(builder.getAffineMapArrayAttr(maps));
  }

  return copyOp.getResult(0);
}

void promoteOperand(OpBuilder &builder, Operation *op, unsigned index,
                    IREE::GPU::PromotionAttr promotionAttr) {
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dpsOp) {
    return;
  }
  // We use the convention that if we are passing an index beyond the inputs
  // then we promote the result of the corresponding dps init.
  if (index >= dpsOp.getNumDpsInputs()) {
    index -= dpsOp.getNumDpsInputs();
    assert(index < op->getNumResults() &&
           "trying to promote out of bound result index");
    // TODO(qedawkins): Move result promotion to attribute interface.
    return promoteResult(builder, op, op->getResult(index));
  }

  Value replacement;
  if (isa<IREE::GPU::UseGlobalTransposeLoadAttr>(promotionAttr)) {
    replacement = transposePromoteOperand(builder, op, index);
  } else {
    OpOperand &operand = op->getOpOperand(index);
    replacement = promotionAttr.promoteOperand(builder, operand);
  }
  op->setOperand(index, replacement);
}

struct GPUPromoteMatmulOperandsPass final
    : impl::GPUPromoteMatmulOperandsPassBase<GPUPromoteMatmulOperandsPass> {
  using GPUPromoteMatmulOperandsPassBase::GPUPromoteMatmulOperandsPassBase;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    OpBuilder builder(funcOp);
    WalkResult walkResult = funcOp.walk([&](Operation *op) {
      auto loweringConfig =
          getLoweringConfig<IREE::GPU::LoweringConfigAttr>(op);
      if (!loweringConfig) {
        return WalkResult::advance();
      }

      std::optional<SmallVector<int64_t>> promotedOperands =
          getPromotedOperandList(loweringConfig);
      if (!promotedOperands) {
        return WalkResult::advance();
      }

      std::optional<ArrayRef<Attribute>> maybePromotionTypes =
          getPromotionTypesList(loweringConfig);

      Attribute derived =
          IREE::GPU::DerivedThreadConfigAttr::get(op->getContext());
      ArrayRef<Attribute> promotionTypes =
          maybePromotionTypes.value_or(ArrayRef<Attribute>());

      builder.setInsertionPoint(op);
      for (auto [operand, maybePromotionType] :
           llvm::zip_longest(promotedOperands.value(), promotionTypes)) {
        auto promotionType = dyn_cast<IREE::GPU::PromotionAttr>(
            maybePromotionType.value_or(derived));
        if (!promotionType) {
          op->emitOpError(
              "promotion types does not implement promotion attr interface");
          return WalkResult::interrupt();
        }
        promoteOperand(builder, op, operand.value(), promotionType);
      }
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
