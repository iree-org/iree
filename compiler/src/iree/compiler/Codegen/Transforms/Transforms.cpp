// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Transforms.cpp - Transformations common to all backends ------------===//
//
// Defines transformations that are common to backends
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Transforms/Transforms.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/ScalableValueBoundsConstraintSet.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

#define DEBUG_TYPE "iree-codegen-transforms"

namespace mlir::iree_compiler {

static bool sliceFilter(Operation *op, ValueRange nonIndexComputationOperands,
                        Operation *baseOp) {
  for (auto val : nonIndexComputationOperands) {
    if (op == val.getDefiningOp())
      return false;
  }
  if (op->isProperAncestor(baseOp))
    return false;
  return !isa<IREE::HAL::InterfaceConstantLoadOp>(op);
}

static SliceAndDynamicDims cloneOffsetsSizesAndStridesImpl(
    OpBuilder &builder, Operation *baseOp,
    ValueRange nonIndexComputationOperands, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
    ValueRange dynamicDims) {
  BackwardSliceOptions options;
  options.filter = [&](Operation *op) {
    return sliceFilter(op, nonIndexComputationOperands, baseOp);
  };
  SetVector<Operation *> slice;
  getBackwardSlice(baseOp, &slice, options);
  IRMapping bvm;
  for (auto origOp : slice) {
    builder.clone(*origOp, bvm);
  }
  auto remapOpFoldResult = [&bvm](ArrayRef<OpFoldResult> ofrs) {
    SmallVector<OpFoldResult> clonedOfrs;
    clonedOfrs.reserve(ofrs.size());
    for (auto ofr : ofrs) {
      if (ofr.is<Attribute>()) {
        clonedOfrs.push_back(ofr);
      } else {
        clonedOfrs.push_back(bvm.lookupOrDefault(ofr.get<Value>()));
      }
    }
    return clonedOfrs;
  };
  auto remapValues = [&bvm](ValueRange vals) {
    SmallVector<Value> clonedVals;
    clonedVals.reserve(vals.size());
    for (auto val : vals) {
      clonedVals.push_back(bvm.lookupOrDefault(val));
    }
    return clonedVals;
  };

  SliceAndDynamicDims clonedVals;
  clonedVals.offsets = remapOpFoldResult(offsets);
  clonedVals.sizes = remapOpFoldResult(sizes);
  clonedVals.strides = remapOpFoldResult(strides);
  clonedVals.dynamicDims = remapValues(dynamicDims);
  return clonedVals;
}

SliceAndDynamicDims
cloneOffsetsSizesAndStrides(OpBuilder &builder,
                            IREE::Flow::DispatchTensorStoreOp storeOp) {
  return cloneOffsetsSizesAndStridesImpl(
      builder, storeOp, ValueRange{storeOp.getValue(), storeOp.getTarget()},
      storeOp.getMixedOffsets(), storeOp.getMixedSizes(),
      storeOp.getMixedStrides(), storeOp.getTargetDims());
}

SliceAndDynamicDims
cloneOffsetsSizesAndStrides(OpBuilder &builder,
                            IREE::Flow::DispatchTensorLoadOp loadOp) {
  return cloneOffsetsSizesAndStridesImpl(
      builder, loadOp, ValueRange{loadOp.getSource()}, loadOp.getMixedOffsets(),
      loadOp.getMixedSizes(), loadOp.getMixedStrides(), loadOp.getSourceDims());
}

template <typename AllocLikeOpType>
std::optional<Value> hoistOneStaticallyBoundAllocation(
    mlir::FunctionOpInterface funcOp, OpBuilder &builder, Location loc,
    MemRefType allocLikeType, ValueRange dynamicSizes,
    std::optional<uint64_t> alignment,
    std::optional<vector::VscaleRange> vscaleRange) {
  IntegerAttr alignmentAttr =
      alignment ? builder.getI64IntegerAttr(alignment.value()) : nullptr;
  // For static case just create a new allocation in the entry block of the same
  // size. No need to insert a subview.
  if (dynamicSizes.empty()) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&funcOp.getFunctionBody().front());
    Value allocation =
        builder.create<AllocLikeOpType>(loc, allocLikeType, alignmentAttr);
    if (std::is_same<AllocLikeOpType, memref::AllocOp>::value) {
      builder.setInsertionPoint(
          funcOp.getFunctionBody().front().getTerminator());
      builder.create<memref::DeallocOp>(loc, allocation);
    }
    return allocation;
  }

  Value vscale = nullptr;
  // Use the given vscale range (if provided); otherwise look at the target
  // information.
  if (!vscaleRange.has_value()) {
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    vscaleRange = getDefaultVscaleRange(targetAttr);
  }

  auto computeAllocationBound = [&](Value value) -> FailureOr<OpFoldResult> {
    if (vscaleRange.has_value()) {
      // Scalability supported: Allocations could be scalable.
      FailureOr<vector::ConstantOrScalableBound> ub =
          vector::ScalableValueBoundsConstraintSet::computeScalableBound(
              value, std::nullopt, vscaleRange->vscaleMin,
              vscaleRange->vscaleMax, presburger::BoundType::UB);
      if (failed(ub))
        return failure();

      if (ub->map.isSingleConstant()) {
        auto constantBound = ub->map.getSingleConstantResult();
        return OpFoldResult(builder.getIndexAttr(constantBound));
      }

      if (!vscale)
        vscale = builder.create<vector::VectorScaleOp>(loc);
      return affine::materializeComputedBound(
          builder, loc, ub->map, {std::make_pair(vscale, std::nullopt)});
    }
    // Non-scalable target: Assume everything is fixed-size.
    auto ub = ValueBoundsConstraintSet::computeConstantBound(
        presburger::BoundType::UB, {value, std::nullopt},
        /*stopCondition=*/nullptr,
        /*closedUB=*/true);
    if (failed(ub))
      return failure();

    return OpFoldResult(builder.getIndexAttr(*ub));
  };

  /// For the dynamic but bounded case, insert an allocation of the shape of the
  /// bounds, and a subview of the required size to be used as a replacement.

  SmallVector<OpFoldResult> allocSizes;
  SmallVector<OpFoldResult> subviewSizes;
  allocSizes.reserve(allocLikeType.getRank());
  subviewSizes.reserve(allocLikeType.getRank());

  Value allocation;
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&funcOp.getFunctionBody().front());

    int index = 0;
    for (auto dimSize : allocLikeType.getShape()) {
      if (!ShapedType::isDynamic(dimSize)) {
        auto dimSizeAttr = builder.getIndexAttr(dimSize);
        allocSizes.push_back(dimSizeAttr);
        subviewSizes.push_back(dimSizeAttr);
        continue;
      }

      Value dynamicSize = dynamicSizes[index++];
      auto ub = computeAllocationBound(dynamicSize);
      if (failed(ub))
        return std::nullopt;

      allocSizes.push_back(*ub);
      subviewSizes.push_back(dynamicSize);
    }

    // FIXME: The `AllocLikeOp::build()` method for OpFoldResult drops the
    // layout, so we have to resolve the static/dynamic values here.
    SmallVector<int64_t> staticShape;
    SmallVector<Value> dynamicSizes;
    dispatchIndexOpFoldResults(allocSizes, dynamicSizes, staticShape);
    auto allocationType = allocLikeType.clone(staticShape);

    allocation = builder.create<AllocLikeOpType>(loc, allocationType,
                                                 dynamicSizes, alignmentAttr);
  }

  SmallVector<OpFoldResult> offsets(allocLikeType.getRank(),
                                    builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(allocLikeType.getRank(),
                                    builder.getIndexAttr(1));
  Value subviewOp = builder.create<memref::SubViewOp>(loc, allocation, offsets,
                                                      subviewSizes, strides);

  // Cast it back to the original types to prevent consumer op's verification
  // error. It could happen when the consumer op is a memref.subview op.
  if (subviewOp.getType() != allocLikeType) {
    subviewOp = builder.create<memref::CastOp>(loc, allocLikeType, subviewOp);
  }

  if (std::is_same<AllocLikeOpType, memref::AllocOp>::value) {
    builder.setInsertionPoint(funcOp.getFunctionBody().front().getTerminator());
    builder.create<memref::DeallocOp>(loc, allocation);
  }

  return subviewOp;
}

template <typename AllocLikeOpType>
std::optional<Value> hoistOneStaticallyBoundAllocation(
    mlir::FunctionOpInterface funcOp, OpBuilder &builder,
    AllocLikeOpType allocLikeOp,
    std::optional<vector::VscaleRange> vscaleRange) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(allocLikeOp);
  return hoistOneStaticallyBoundAllocation<AllocLikeOpType>(
      funcOp, builder, allocLikeOp.getLoc(), allocLikeOp.getType(),
      allocLikeOp.getDynamicSizes(), allocLikeOp.getAlignment(), vscaleRange);
}

/// Some uses of a AllocLike can be replaced with a `memref.subview`
/// easily. Other uses (like a use in a `scf.yield` or `func.return`) are
/// non-trivial because of compatibility between types of different SSA values.
static bool isUseReplaceableWithSubview(OpOperand &use) {
  Operation *user = use.getOwner();
  return isa<linalg::LinalgOp, memref::DeallocOp, memref::StoreOp,
             memref::SubViewOp>(user);
}

template <typename AllocLikeOpType>
void hoistStaticallyBoundAllocationsInFunc(
    RewriterBase &rewriter, mlir::FunctionOpInterface funcOp,
    std::optional<vector::VscaleRange> vscaleRange) {
  SmallVector<AllocLikeOpType> allocLikeOps;

  // Collect all allocLikes that are hoistable.
  funcOp.walk([&](AllocLikeOpType allocLikeOp) {
    if (allocLikeOp->getBlock() == &funcOp.getFunctionBody().front())
      return;
    if (allocLikeOp.getDynamicSizes().empty()) {
      allocLikeOps.push_back(allocLikeOp);
      return;
    }
    if (llvm::all_of(allocLikeOp->getUses(), [](OpOperand &use) {
          return isUseReplaceableWithSubview(use);
        })) {
      allocLikeOps.push_back(allocLikeOp);
      return;
    }
  });

  // Hoist the allocLikes and replace all uses.
  for (auto allocLikeOp : allocLikeOps) {
    // Record potential memref::DeallocOps to clean up after hoisting occurs.
    SmallVector<memref::DeallocOp> deallocOps;
    for (Operation *user : allocLikeOp->getUsers()) {
      auto dealloc = dyn_cast<memref::DeallocOp>(user);
      if (dealloc)
        deallocOps.push_back(dealloc);
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Alloca Op : ";
      allocLikeOp->dump();
      int numUses = std::distance(allocLikeOp.getResult().use_begin(),
                                  allocLikeOp.getResult().use_end());
      llvm::dbgs() << " num Uses : " << numUses;
    });
    std::optional<Value> replacement = hoistOneStaticallyBoundAllocation(
        funcOp, rewriter, allocLikeOp, vscaleRange);
    if (!replacement)
      continue;
    LLVM_DEBUG({
      llvm::dbgs() << "Replacement : ";
      replacement->dump();
    });
    Value replacementVal = replacement.value();
    rewriter.replaceOp(allocLikeOp, replacementVal);

    for (memref::DeallocOp deallocOp : deallocOps)
      rewriter.eraseOp(deallocOp);
  }
}

/// Explicit instantiations for `hoistStaticallyBoundAllocationsInFunc` and
/// dependent functions.
template std::optional<Value>
hoistOneStaticallyBoundAllocation<memref::AllocOp>(
    mlir::FunctionOpInterface funcOp, OpBuilder &builder, Location loc,
    MemRefType allocLikeType, ValueRange dynamicSizes,
    std::optional<uint64_t> alignment,
    std::optional<vector::VscaleRange> vscaleRange);
template std::optional<Value>
hoistOneStaticallyBoundAllocation<memref::AllocaOp>(
    mlir::FunctionOpInterface funcOp, OpBuilder &builder, Location loc,
    MemRefType allocLikeType, ValueRange dynamicSizes,
    std::optional<uint64_t> alignment,
    std::optional<vector::VscaleRange> vscaleRange);
template std::optional<Value>
hoistOneStaticallyBoundAllocation<memref::AllocOp>(
    mlir::FunctionOpInterface funcOp, OpBuilder &builder,
    memref::AllocOp allocLikeOp,
    std::optional<vector::VscaleRange> vscaleRange);
template std::optional<Value>
hoistOneStaticallyBoundAllocation<memref::AllocaOp>(
    mlir::FunctionOpInterface funcOp, OpBuilder &builder,
    memref::AllocaOp allocLikeOp,
    std::optional<vector::VscaleRange> vscaleRange);
template void hoistStaticallyBoundAllocationsInFunc<memref::AllocOp>(
    RewriterBase &rewriter, mlir::FunctionOpInterface funcOp,
    std::optional<vector::VscaleRange> vscaleRange);
template void hoistStaticallyBoundAllocationsInFunc<memref::AllocaOp>(
    RewriterBase &rewriter, mlir::FunctionOpInterface funcOp,
    std::optional<vector::VscaleRange> vscaleRange);

//===---------------------------------------------------------------------===//
// Lowering `flow.dispatch.workgroup_count_from_slice` operation.
//===---------------------------------------------------------------------===//

LogicalResult lowerWorkgroupCountFromSliceOp(
    RewriterBase &rewriter,
    IREE::Flow::DispatchWorkgroupCountFromSliceOp workgroupCountOp,
    mlir::FunctionOpInterface entryPointFn,
    ArrayRef<OpFoldResult> workgroupCount, int maxWorkgroupParallelDims) {
  // Compute the backward slice of the workgroup count operations.
  BackwardSliceOptions options;
  options.filter = [](Operation *op) {
    return !isa<IREE::Flow::DispatchWorkloadOrdinalOp>(op);
  };
  options.inclusive = true;
  llvm::SetVector<Operation *> slice;
  for (auto ofr : workgroupCount) {
    if (auto val = dyn_cast<Value>(ofr)) {
      mlir::getBackwardSlice(val, &slice, options);
    }
  }
  // Since there are more than one slices, sort the operations again.
  auto slicedOps = llvm::to_vector(slice);
  mlir::computeTopologicalSorting(slicedOps);

  // Insert the slice into workgroup count region with all `hal.constant.index`
  // operations replaced with arguments (drop the front argument since that is
  // `hal.device`).
  auto workloadVals = workgroupCountOp.getOperands();
  IRMapping map;
  // Map `flow.dispatch.constant_ordinal` op with the corresponding operand of
  // the `flow.dispatch.workgroup_count_default` operation.
  SmallVector<IREE::Flow::DispatchWorkloadOrdinalOp> ordinalOps;
  entryPointFn.walk([&](IREE::Flow::DispatchWorkloadOrdinalOp ordinalOp) {
    ordinalOps.push_back(ordinalOp);
  });
  for (auto ordinalOp : ordinalOps) {
    int64_t ordinal = ordinalOp.getOrdinal().getSExtValue();
    if (ordinal >= workloadVals.size()) {
      ordinalOp.emitOpError(
          "ordinal number is higher than the number of workloads captured in "
          "the workgroup count region");
    }
    map.map(ordinalOp.getResult(),
            workloadVals[ordinalOp.getOrdinal().getSExtValue()]);
  }
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(workgroupCountOp);
  for (auto op : slice) {
    // TODO(#13038) This is a WAR for the these ops ending up in workgroup count
    // computation. They should not. Some pre-processing at MaterializeEncoding
    // time might make these go away.
    if (isa<IREE::Codegen::QueryTileSizesOp>(op)) {
      Value constVal =
          rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 16);
      for (auto result : op->getResults()) {
        map.map(result, constVal);
      }
      continue;
    }
    rewriter.clone(*op, map);
  }
  SmallVector<OpFoldResult> results;
  // Since the workgroup count at HAL level is in x, y, z form, process the
  // workload in reverse.
  for (auto ofr : llvm::reverse(workgroupCount)) {
    if (auto val = dyn_cast<Value>(ofr)) {
      results.push_back(getAsOpFoldResult(map.lookup(val)));
    } else {
      results.push_back(ofr);
    }
  }

  // The `maxWorkgroupParallelDims` represents the maximum dimension number
  // used for distribution. The rest of the workgroups get folded into the
  // `maxWorkgroupParallelDims`
  Location loc = workgroupCountOp.getLoc();
  if (results.size() > maxWorkgroupParallelDims) {
    MutableArrayRef<OpFoldResult> resultsRef =
        llvm::MutableArrayRef<OpFoldResult>(results);
    assert(maxWorkgroupParallelDims != 0 &&
           "unexpected max parallel dimensions being 0");
    AffineExpr s0, s1;
    bindSymbols(rewriter.getContext(), s0, s1);
    AffineMap foldMap = AffineMap::get(0, 2, s0 * s1);
    for (auto [index, foldedResult] : llvm::enumerate(
             resultsRef.take_back(results.size() - maxWorkgroupParallelDims))) {
      resultsRef[maxWorkgroupParallelDims - 1] =
          affine::makeComposedFoldedAffineApply(
              rewriter, loc, foldMap,
              {resultsRef[maxWorkgroupParallelDims - 1],
               resultsRef[maxWorkgroupParallelDims + index]});
    }
    results.resize(maxWorkgroupParallelDims);
  }

  // Fill out the remaining results with 1.
  if (results.size() < workgroupCountOp.getNumResults()) {
    results.resize(workgroupCountOp.getNumResults(), rewriter.getIndexAttr(1));
  }
  rewriter.replaceOp(workgroupCountOp,
                     getValueOrCreateConstantIndexOp(rewriter, loc, results));
  for (auto ordinalOp : ordinalOps) {
    rewriter.replaceOp(ordinalOp, ordinalOp.getOperand());
  }

  return success();
}

LogicalResult lowerWorkgroupCountFromSliceOp(
    RewriterBase &rewriter, mlir::FunctionOpInterface entryPointFn,
    ArrayRef<OpFoldResult> workgroupCount, int maxWorkgroupParallelDims) {
  std::optional<IREE::HAL::ExecutableExportOp> exportOp =
      getEntryPoint(entryPointFn);
  if (!exportOp) {
    return success();
  }
  Block *body = exportOp->getWorkgroupCountBody();
  if (!body) {
    return success();
  }
  auto countOps = body->getOps<IREE::Flow::DispatchWorkgroupCountFromSliceOp>();
  if (countOps.empty()) {
    // If there are no `flow.dispatch.workgroup_count_default` operations
    // do nothing.
    return success();
  }
  if (!llvm::hasSingleElement(countOps)) {
    return exportOp->emitOpError(
        "unexpected multiple flow.dispatch.workgroup_count_default operations "
        "in body");
  }
  return lowerWorkgroupCountFromSliceOp(rewriter, *countOps.begin(),
                                        entryPointFn, workgroupCount,
                                        maxWorkgroupParallelDims);
}

//===---------------------------------------------------------------------===//
// Patterns to fold tensor.expand/collapse_shape into
// `hal.interface.binding.subspan`
//===---------------------------------------------------------------------===//

namespace {

// TODO(antigainst): enable dynamic shape support once they are needed.
template <typename TensorReshapeOp>
static std::optional<Value> getStaticReshapeOpSrc(TensorReshapeOp reshapeOp) {
  auto reshapeSrcType = llvm::cast<ShapedType>(reshapeOp.getSrc().getType());
  auto reshapeDstType = llvm::cast<ShapedType>(reshapeOp.getType());
  if (!reshapeSrcType.hasStaticShape() || !reshapeDstType.hasStaticShape())
    return std::nullopt;
  return reshapeOp.getSrc();
}

/// Folds tensor.expand/collapse_shape into the source
/// hal.interface.binding.subspan.
///
/// For example, this matches the following pattern:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !flow.dispatch.tensor<readonly:tensor<3x3x1x96xf32>>
///   %tensor = flow.dispatch.tensor.load %subspan :
///       !flow.dispatch.tensor<readonly:tensor<3x3x1x96xf32>> ->
///       tensor<3x3x1x96xf32>
///   %0 = linalg.tensor_reshape %tensor [
///         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
///       ] : tensor<3x3x1x96xf32> into tensor<864xf32>
///
/// And turns it into:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !flow.dispatch.tensor<readonly:tensor<864xf32>>
///   %0 = flow.dispatch.tensor.load %subspan :
///       !flow.dispatch.tensor<readonly:tensor<864xf32>> -> tensor<864xf32>
template <typename TensorReshapeOp>
struct FoldReshapeIntoInterfaceTensorLoad : OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    std::optional<Value> reshapeSrc =
        getStaticReshapeOpSrc<TensorReshapeOp>(reshapeOp);
    if (!reshapeSrc)
      return failure();

    auto loadOp =
        reshapeSrc->template getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
    if (!loadOp)
      return failure();

    // Make sure we are loading the full incoming subspan. Otherwise we cannot
    // simply adjust the subspan's resultant type later.
    if (!loadOp.offsets().empty() || !loadOp.sizes().empty() ||
        !loadOp.strides().empty())
      return failure();

    auto subspanOp =
        loadOp.getSource()
            .template getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp)
      return failure();
    assert(subspanOp.getDynamicDims().empty());

    auto tensorAccess =
        llvm::cast<IREE::Flow::DispatchTensorType>(subspanOp.getType())
            .getAccess();
    auto newSubspanType = IREE::Flow::DispatchTensorType::get(
        tensorAccess, reshapeOp.getResultType());

    Value newSubspanOp = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp.getLoc(), newSubspanType, subspanOp.getLayout(),
        subspanOp.getBinding(), subspanOp.getByteOffset(),
        subspanOp.getDynamicDims(), subspanOp.getAlignmentAttr(),
        subspanOp.getDescriptorFlagsAttr());

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
        reshapeOp, reshapeOp.getResultType(), newSubspanOp,
        loadOp.getSourceDims());

    return success();
  }
};

/// Folds tensor.expand into the source hal.interface.binding.subspan.
///
/// For example, this matches the following pattern:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !flow.dispatch.tensor<writeonly:tensor<3x3x1x96xf32>>
///   %0 = tensor.expand_shape %tensor [[0, 1, 2, 3]]
///       : tensor<864xf32> into tensor<3x3x1x96xf32>
///   %tensor = flow.dispatch.tensor.store %0, %subspan :
///       !flow.dispatch.tensor<writeonly:tensor<3x3x1x96xf32>> ->
///       tensor<3x3x1x96xf32>
///
/// And turns it into:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !flow.dispatch.tensor<writeonly:tensor<864xf32>>
///   %0 = flow.dispatch.tensor.store %tensor, %subspan :
///       !flow.dispatch.tensor<writeonly:tensor<864xf32>> -> tensor<864xf32>
struct FoldExpandShapeIntoInterfaceTensorStore
    : OpRewritePattern<IREE::Flow::DispatchTensorStoreOp> {
  using OpRewritePattern<IREE::Flow::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Flow::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Make sure we are storing the full incoming subspan. Otherwise we cannot
    // simply adjust the subspan's resultant type later.
    if (!storeOp.offsets().empty() || !storeOp.sizes().empty() ||
        !storeOp.strides().empty())
      return failure();

    auto reshapeOp = storeOp.getValue().getDefiningOp<tensor::ExpandShapeOp>();
    if (!reshapeOp) {
      return failure();
    }

    // Dynamic shapes are currently unsupported.
    std::optional<Value> reshapeSrc =
        getStaticReshapeOpSrc<tensor::ExpandShapeOp>(reshapeOp);
    if (!reshapeSrc)
      return failure();

    auto subspanOp =
        storeOp.getTarget()
            .template getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp)
      return failure();
    assert(subspanOp.getDynamicDims().empty());

    auto tensorAccess =
        llvm::cast<IREE::Flow::DispatchTensorType>(subspanOp.getType())
            .getAccess();
    auto newSubspanType = IREE::Flow::DispatchTensorType::get(
        tensorAccess, reshapeSrc->getType());

    Value newSubspanOp;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(subspanOp);
      newSubspanOp = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
          subspanOp.getLoc(), newSubspanType, subspanOp.getLayout(),
          subspanOp.getBinding(), subspanOp.getByteOffset(),
          subspanOp.getDynamicDims(), subspanOp.getAlignmentAttr(),
          subspanOp.getDescriptorFlagsAttr());
    }

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        storeOp, *reshapeSrc, newSubspanOp, storeOp.getTargetDims());

    return success();
  }
};

/// Folds tensor.collapse_shape with static shape into the source
/// hal.interface.binding.subspan. The binding is currently required to be
/// static as well, however it is impossible to generate a dispatch where
/// this would not be true today.
///
/// For example, this matches the following pattern:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !flow.dispatch.tensor<writeonly:tensor<2592xf32>>
///   %0 = tensor.collapse_shape %tensor [[0, 1, 2, 3]]
///       : tensor<3x3x1x96xf32> into tensor<864xf32>
///   %tensor = flow.dispatch.tensor.store %0, %subspan,
///       offsets = [%x], sizes = [864], strides = [1]
///       : tensor<864xf32> -> !flow.dispatch.tensor<writeonly:tensor<2592xf32>>
///
/// And turns it into:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !flow.dispatch.tensor<writeonly:tensor<9x3x1x96xf32>>
///   %0 = flow.dispatch.tensor.store %tensor, %subspan :
///       offsets = [%x * 286, 0, 0, 0], sizes = [3, 3, 1, 96]
///       strides = [1, 1, 1, 1] : tensor<3x3x1x96xf32> ->
///       !flow.dispatch.tensor<writeonly:tensor<9x3x1x96xf32>>
struct FoldCollapseShapeIntoInterfaceTensorStore
    : OpRewritePattern<IREE::Flow::DispatchTensorStoreOp> {
  using OpRewritePattern<IREE::Flow::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Flow::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Bail out if the strides aren't unit.
    if (!llvm::all_of(storeOp.getMixedStrides(), [](OpFoldResult s) {
          return isConstantIntValue(s, 1);
        })) {
      return failure();
    }

    auto collapseShape =
        storeOp.getValue().getDefiningOp<tensor::CollapseShapeOp>();
    // TODO: Support dynamic shapes.
    if (!collapseShape || !collapseShape.getSrcType().hasStaticShape()) {
      return failure();
    }

    auto subspanOp =
        storeOp.getTarget()
            .template getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    // TODO: Support dynamic dims.
    if (!subspanOp || !subspanOp.getDynamicDims().empty()) {
      return failure();
    }

    auto subspanType =
        llvm::cast<IREE::Flow::DispatchTensorType>(subspanOp.getType());

    ArrayRef<int64_t> reshapeSrcShape = collapseShape.getSrcType().getShape();

    // Verify the subspan shape against the shape of the slice being inserted.
    for (auto [size, group] : llvm::zip_equal(
             subspanType.getShape(), collapseShape.getReassociationIndices())) {
      if (group.size() == 1) {
        continue;
      }

      int64_t innerDimSize = 1;
      for (auto i : llvm::drop_begin(group)) {
        innerDimSize *= reshapeSrcShape[i];
      }
      if (size % innerDimSize != 0) {
        return rewriter.notifyMatchFailure(
            storeOp, "Subspan type indivisible by expanded shape");
      }
    }

    AffineExpr d0, d1;
    bindDims(rewriter.getContext(), d0, d1);
    AffineExpr div = d0.ceilDiv(d1);

    Location loc = collapseShape.getLoc();
    SmallVector<int64_t> expandedSubspanShape;
    SmallVector<OpFoldResult> expandedOffsets;
    SmallVector<OpFoldResult> expandedSizes;
    OpFoldResult zero = rewriter.getIndexAttr(0);
    for (auto [size, group, offset] : llvm::zip_equal(
             subspanType.getShape(), collapseShape.getReassociationIndices(),
             storeOp.getMixedOffsets())) {
      expandedSizes.push_back(rewriter.getIndexAttr(reshapeSrcShape[group[0]]));

      // Special case for 1 to avoid going through arith folders.
      if (group.size() == 1) {
        expandedOffsets.push_back(offset);
        expandedSubspanShape.push_back(size);
        continue;
      }

      int64_t innerDimSize = 1;
      for (auto i : llvm::drop_begin(group)) {
        innerDimSize *= reshapeSrcShape[i];
      }
      OpFoldResult innerDimSizeAttr = rewriter.getIndexAttr(innerDimSize);
      expandedOffsets.push_back(affine::makeComposedFoldedAffineApply(
          rewriter, loc, div, {offset, innerDimSizeAttr}));
      assert(size % innerDimSize == 0);
      expandedSubspanShape.push_back(size / innerDimSize);
      for (auto i : llvm::drop_begin(group)) {
        expandedOffsets.push_back(zero);
        int64_t dimSize = reshapeSrcShape[i];
        expandedSubspanShape.push_back(dimSize);
        expandedSizes.push_back(rewriter.getIndexAttr(dimSize));
      }
    }

    auto newSubspanTensorType = RankedTensorType::get(
        expandedSubspanShape, collapseShape.getSrcType().getElementType());
    auto newSubspanType = IREE::Flow::DispatchTensorType::get(
        subspanType.getAccess(), newSubspanTensorType);

    Value newSubspanOp;
    {
      // NOTE: If there were any dynamic dims, they would need to be updated
      // based on the newly introduced static sizes as well.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(subspanOp);
      newSubspanOp = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
          subspanOp.getLoc(), newSubspanType, subspanOp.getLayout(),
          subspanOp.getBinding(), subspanOp.getByteOffset(),
          subspanOp.getDynamicDims(), subspanOp.getAlignmentAttr(),
          subspanOp.getDescriptorFlagsAttr());
    }

    SmallVector<OpFoldResult> expandedStrides(reshapeSrcShape.size(),
                                              rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        storeOp, collapseShape.getSrc(), newSubspanOp, storeOp.getTargetDims(),
        expandedOffsets, expandedSizes, expandedStrides);
    return success();
  }
};

} // namespace

void populateReshapeToInterfaceTensorPatterns(RewritePatternSet &patterns) {
  patterns.insert<FoldReshapeIntoInterfaceTensorLoad<tensor::CollapseShapeOp>,
                  FoldReshapeIntoInterfaceTensorLoad<tensor::ExpandShapeOp>>(
      patterns.getContext());
  patterns.insert<FoldExpandShapeIntoInterfaceTensorStore>(
      patterns.getContext());
  patterns.insert<FoldCollapseShapeIntoInterfaceTensorStore>(
      patterns.getContext());
}

//===--------------------------------------------------------------------====//
// Pattern to remove dead allocations
//===--------------------------------------------------------------------====//

namespace {

// Erases the operation if its only users are memref.assume_alignment ops.
static LogicalResult eraseAlignmentOnlyDeadOp(PatternRewriter &rewriter,
                                              Operation *op) {
  SmallVector<Operation *> deadUsers;
  for (OpOperand &use : op->getUses()) {
    if (auto user = dyn_cast<memref::AssumeAlignmentOp>(use.getOwner())) {
      deadUsers.push_back(user);
      continue;
    }
    // For any other use, return failure;
    return failure();
  }
  for (auto user : deadUsers) {
    rewriter.eraseOp(user);
  }
  rewriter.eraseOp(op);
  return success();
}

// Removes operations with Allocate MemoryEffects but no uses.
struct RemoveDeadMemAllocs : RewritePattern {
  RemoveDeadMemAllocs(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memEffect || !memEffect.hasEffect<MemoryEffects::Allocate>()) {
      return failure();
    }
    return eraseAlignmentOnlyDeadOp(rewriter, op);
  }
};

// Removes hal.interface.binding.subspan ops with only assume_alignment uses.
struct RemoveDeadInterfaceBindings
    : OpRewritePattern<IREE::HAL::InterfaceBindingSubspanOp> {
  RemoveDeadInterfaceBindings(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<IREE::HAL::InterfaceBindingSubspanOp>(context,
                                                               benefit) {}

  LogicalResult matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp op,
                                PatternRewriter &rewriter) const override {
    return eraseAlignmentOnlyDeadOp(rewriter, op);
  }
};
} // namespace

void populateRemoveDeadMemAllocPatterns(RewritePatternSet &patterns) {
  patterns.insert<RemoveDeadMemAllocs>(patterns.getContext());
  patterns.insert<RemoveDeadInterfaceBindings>(patterns.getContext());
}

void analyseAllocsForPacking(mlir::FunctionOpInterface funcOp,
                             ArrayRef<Operation *> allocs,
                             SmallVector<AliasGroup> &aliasGroups) {
  // Represent of a group of allocations with overlapping liverange and the
  // liveness of the overall group.
  struct AllocGroup {
    SmallVector<Operation *> allocs;
    // Keep track of every operation where any of the alloc in the group is
    // live.
    // Liveness is represent as a set of Operations where the alloc is alive.
    // To make it merge liveranges and check if a given Operation interfers
    // with the liverange we store it as a DesneSet.
    llvm::DenseSet<Operation *> liveness;
  };
  Liveness liveness(funcOp);
  SmallVector<AllocGroup> groups;
  for (Operation *alloc : allocs) {
    SmallVector<size_t> aliasGroups;
    for (size_t i : llvm::seq<size_t>(0, groups.size())) {
      AllocGroup &group = groups[i];
      for (Operation *user : alloc->getUsers()) {
        // Skip the whole analysis if any user is a subview.
        // TODO: This could be extended if needed by recursively merging
        // liveness.
        if (isa<memref::SubViewOp>(user))
          return;
        if (group.liveness.count(user)) {
          aliasGroups.push_back(i);
          break;
        }
      }
    }
    if (aliasGroups.empty()) {
      // If we didn't find any alias group create a new one.
      AllocGroup &newGroup = groups.emplace_back();
      newGroup.allocs.push_back(alloc);
      Liveness::OperationListT liveInfo =
          liveness.resolveLiveness(alloc->getResult(0));
      newGroup.liveness.insert(liveInfo.begin(), liveInfo.end());
    } else {
      // Merge the alloc into the first alias group it interfers with.
      AllocGroup &mergeGroup = groups[aliasGroups[0]];
      mergeGroup.allocs.push_back(alloc);
      Liveness::OperationListT liveInfo =
          liveness.resolveLiveness(alloc->getResult(0));
      mergeGroup.liveness.insert(liveInfo.begin(), liveInfo.end());
      // Then merge all the other alias groups into the first group.
      for (size_t i = 1, e = aliasGroups.size(); i < e; i++) {
        AllocGroup &group = groups[aliasGroups[i]];
        mergeGroup.allocs.insert(mergeGroup.allocs.end(), group.allocs.begin(),
                                 group.allocs.end());
        mergeGroup.liveness.insert(group.liveness.begin(),
                                   group.liveness.end());
        // For simplicity we leave the group empty and don't remove it.
        group.allocs.clear();
        group.liveness.clear();
      }
    }
  }

  LLVM_DEBUG({
    for (size_t i = 0; i < groups.size(); i++) {
      llvm::dbgs() << "Alias group " << i << ":\n";
      for (Operation *op : groups[i].allocs)
        op->dump();
    }
  });

  for (size_t i = 0; i < groups.size(); i++) {
    if (groups[i].allocs.empty())
      continue;
    aliasGroups.push_back(std::move(groups[i].allocs));
  }
}

static int64_t getAllocSize(Operation *op, DataLayout &dataLayout) {
  auto allocOp = cast<memref::AllocOp>(op);
  int64_t numElements = allocOp.getType().getNumElements();
  return (dataLayout.getTypeSizeInBits(allocOp.getType().getElementType()) *
          numElements) /
         8;
}

void packAllocs(OpBuilder &builder, mlir::FunctionOpInterface funcOp,
                ArrayRef<AliasGroup> aliasGroups) {
  if (aliasGroups.empty())
    return;
  DataLayout dataLayout = DataLayout::closest(funcOp);
  builder.setInsertionPointToStart(&(*funcOp.getFunctionBody().begin()));
  int64_t maxAlloc = 0;
  for (size_t i = 0; i < aliasGroups.size(); i++) {
    int64_t allocSize = 0;
    for (Operation *alloc : aliasGroups[i]) {
      allocSize += getAllocSize(alloc, dataLayout);
    }
    maxAlloc = std::max(maxAlloc, allocSize);
  }
  Attribute memorySpace =
      llvm::cast<MemRefType>(aliasGroups[0][0]->getResultTypes()[0])
          .getMemorySpace();
  MemRefType allocType = MemRefType::get({maxAlloc}, builder.getI8Type(),
                                         AffineMap(), memorySpace);
  Value packedAlloc =
      builder.create<memref::AllocOp>(funcOp.getLoc(), allocType);
  for (size_t i = 0; i < aliasGroups.size(); i++) {
    int64_t offset = 0;
    for (Operation *alloc : aliasGroups[i]) {
      Location loc = alloc->getLoc();
      builder.setInsertionPoint(alloc);
      Value offsetValue = builder.create<arith::ConstantIndexOp>(loc, offset);
      Value newAlloc = builder.create<memref::ViewOp>(
          packedAlloc.getLoc(), alloc->getResultTypes()[0], packedAlloc,
          offsetValue, ArrayRef<Value>({}));
      offset += getAllocSize(alloc, dataLayout);
      alloc->replaceAllUsesWith(ArrayRef<Value>({newAlloc}));
      alloc->erase();
    }
  }
}

LogicalResult tileLinalgOpsWithFilter(mlir::FunctionOpInterface funcOp,
                                      scf::SCFTilingOptions options,
                                      LinalgTransformationFilter filter) {
  IRRewriter rewriter(funcOp.getContext());
  SmallVector<Operation *> candidates;
  funcOp.walk([&](linalg::LinalgOp op) {
    if (succeeded(filter.checkAndNotify(rewriter, op))) {
      candidates.push_back(op);
    }
  });

  for (auto op : candidates) {
    auto target = cast<TilingInterface>(op);

    // tileUsingSCFForOp requires the op is not able to tile op with no
    // iteration domain. Skip the tiling on the op. Otherwise it returns a
    // failure.
    if (target.getLoopIteratorTypes().empty()) {
      continue;
    }

    FailureOr<scf::SCFTilingResult> tiledResults =
        scf::tileUsingSCF(rewriter, target, options);
    if (failed(tiledResults)) {
      return failure();
    }
    for (auto tiledOp : tiledResults->tiledOps) {
      filter.replaceLinalgTransformationFilter(rewriter, tiledOp);
    }
    rewriter.replaceOp(op, tiledResults->replacements);
  }

  return success();
}

LogicalResult
distributeLinalgOpsWithFilter(mlir::FunctionOpInterface funcOp,
                              linalg::LinalgTilingOptions tilingOptions,
                              LinalgTransformationFilter filter) {
  IRRewriter rewriter(funcOp.getContext());
  SmallVector<linalg::LinalgOp> candidates;
  funcOp.walk([&](linalg::LinalgOp op) {
    if (succeeded(filter.checkAndNotify(rewriter, op))) {
      candidates.push_back(op);
    }
  });

  for (auto op : candidates) {
    // TODO: Tile and distribute LinalgOps using interface methods.
    FailureOr<linalg::TiledLinalgOp> res =
        linalg::tileLinalgOp(rewriter, op, tilingOptions);
    if (failed(res)) {
      return failure();
    }
    filter.replaceLinalgTransformationFilter(rewriter, res->op);
    if (res->tensorResults.empty()) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, res->tensorResults);
    }
  }

  return success();
}

//===--------------------------------------------------------------------====//
// Loop hoisting pattern
//===--------------------------------------------------------------------====//

namespace {
struct HoistForallFromFor : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const final {
    if (loop.getBody()->getOperations().size() == 1) {
      return rewriter.notifyMatchFailure(loop,
                                         "Loop only contains a terminator");
    }

    if (loop.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          loop, "unimplemented: multi-result hoisting");
    }

    auto forallOp = dyn_cast<scf::ForallOp>(
        loop.getBody()->getTerminator()->getOperand(0).getDefiningOp());
    if (!forallOp || forallOp.getNumResults() != loop->getNumResults()) {
      return rewriter.notifyMatchFailure(
          loop, "Single for loop return value is not forall result");
    }

    // Verify that all other operations within the loop are only used by
    // implicit capture within the scf.forall.
    Block *loopBody = loop.getBody();
    Value iterArg = loop.getRegionIterArg(0);
    SmallVector<Operation *> operationsToMove;
    for (Operation &op : loopBody->getOperations()) {
      if (&op == forallOp || &op == loopBody->getTerminator()) {
        continue;
      }

      if (op.getNumRegions() != 0 || isa<TilingInterface>(&op)) {
        return rewriter.notifyMatchFailure(
            loop, "unimplemented: hoisting of tilable or region ops.");
      }

      for (Value operand : op.getOperands()) {
        if (operand == iterArg) {
          return rewriter.notifyMatchFailure(
              loop, "Found non forall op that depends on loop iter arg.");
        }
      }

      for (Operation *user : op.getUsers()) {
        if (user == forallOp) {
          return rewriter.notifyMatchFailure(
              loop, "Operation within loop body used by forall op.");
        }
      }
      operationsToMove.push_back(&op);
    }

    // Step 1. Move all other operations into the body of the scf.forall op.
    Block *forallBody = forallOp.getBody();
    for (auto op : llvm::reverse(operationsToMove)) {
      rewriter.moveOpBefore(op, &forallBody->getOperations().front());
    }

    bool isSingleTripLoop = forallOp.isNormalized() &&
                            llvm::all_of(forallOp.getStaticUpperBound(),
                                         [](int64_t i) { return i == 1; });

    // Step 2. Collect the set of tensor.parallel_insert_slice ops in the
    // terminator and their paired extract_slice ops from the for loop iter arg.
    SmallVector<Operation *> sliceOperandProducers;

    BackwardSliceOptions backwardOptions;
    backwardOptions.inclusive = true;
    backwardOptions.filter = [&](Operation *op) -> bool {
      return forallOp->isProperAncestor(op);
    };
    SetVector<Operation *> slice;

    scf::InParallelOp parallelTerminator = forallOp.getTerminator();
    SmallVector<tensor::ParallelInsertSliceOp> terminators(
        forallOp.getNumResults());
    SmallVector<std::optional<tensor::ExtractSliceOp>> pairedSlices(
        forallOp.getNumResults(), std::nullopt);
    int64_t numInductionVars = forallOp.getInductionVars().size();
    for (auto &yieldingOp : parallelTerminator.getYieldingOps()) {
      auto parallelInsert = cast<tensor::ParallelInsertSliceOp>(&yieldingOp);
      BlockArgument destBbArg =
          llvm::cast<BlockArgument>(parallelInsert.getDest());
      tensor::ExtractSliceOp destSlice;
      for (auto user : destBbArg.getUsers()) {
        if (user == parallelInsert)
          continue;
        auto maybeSlice = dyn_cast<tensor::ExtractSliceOp>(user);
        if (!maybeSlice) {
          // Fail if the destination has more users than a direct insert and
          // extract slice unless it is a single trip loop.
          if (!isSingleTripLoop) {
            return failure();
          }
          continue;
        }
        // Require at most one extract per destination.
        if (destSlice) {
          return failure();
        }
        destSlice = maybeSlice;
      }

      // Verify they operate on equivalent subsets, ensuring the slices are
      // hoistable. It is still possible to hoist the loop if this is not true,
      // however in such cases we likely formed the loops in the wrong order.
      if (destSlice && !cast<SubsetOpInterface>(*destSlice)
                            .operatesOnEquivalentSubset(
                                cast<SubsetOpInterface>(*parallelInsert),
                                [](Value v1, Value v2) { return v1 == v2; })) {
        return failure();
      }

      auto isOverwritingFullDestination =
          [](tensor::ParallelInsertSliceOp insert) {
            // TODO: Handle rank reducing case.
            if (insert.getSourceType().getRank() !=
                insert.getDestType().getRank()) {
              return false;
            }
            for (auto [dim, size] : llvm::enumerate(insert.getMixedSizes())) {
              FailureOr<bool> equalDimSize = ValueBoundsConstraintSet::areEqual(
                  {size}, {insert.getDest(), static_cast<int64_t>(dim)});
              if (failed(equalDimSize) || !*equalDimSize)
                return false;
            }
            return true;
          };

      // For single trip loops, verify that the parallel_insert_slice is
      // overwriting the full destination.
      if (!destSlice && !isOverwritingFullDestination(parallelInsert)) {
        return failure();
      }

      int64_t argId = destBbArg.getArgNumber() - numInductionVars;
      terminators[argId] = parallelInsert;
      if (destSlice) {
        pairedSlices[argId] = destSlice;
      }

      // Collect all of the offset/size/stride operands for both slices and
      // compute a backwards slice of the program from them. Fail if any of
      // them depend on the serial loop iterator.
      llvm::SmallDenseSet<Value> sliceOperands;
      sliceOperands.insert(
          parallelInsert.getOperands().begin() +
              parallelInsert.getOffsetSizeAndStrideStartOperandIndex(),
          parallelInsert.getOperands().end());
      if (destSlice) {
        sliceOperands.insert(
            destSlice.getOperands().begin() +
                destSlice.getOffsetSizeAndStrideStartOperandIndex(),
            destSlice.getOperands().end());
      }
      for (Value operand : sliceOperands) {
        if (auto bbArg = dyn_cast<BlockArgument>(operand)) {
          if (bbArg.getOwner()->getParentOp() == loop) {
            return rewriter.notifyMatchFailure(
                loop, "Slice operand producers depend on loop");
          }
        }
        SetVector<Operation *> tmpBackwardSlice;
        getBackwardSlice(operand, &tmpBackwardSlice, backwardOptions);
        slice.set_union(tmpBackwardSlice);
      }
    }

    // An operation is safe to hoist if it is speculatable and none of its
    // operands depend on the serial loop.
    auto isSafeToHoist = [&](Operation *op) {
      return op->getBlock() == forallOp.getBody() && isMemoryEffectFree(op) &&
             isSpeculatable(op) &&
             llvm::none_of(op->getOperands(), [&](Value operand) {
               if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
                 return blockArg.getOwner() == loop.getBody();
               }
               return false;
             });
    };

    if (!llvm::all_of(slice, isSafeToHoist)) {
      return rewriter.notifyMatchFailure(
          loop, "Slice operand producers not safe to hoist out of loop");
    }

    // Sort the backwards slice of the producers for the insertion/extraction
    // indices by the block order in the scf.forall body. This ensures that we
    // hoist operations in the same order they started. Any topological ordering
    // would work too because the operations are speculatable.
    slice = mlir::topologicalSort(slice);

    // Step 3. Create the ForallOp.
    Location loc = forallOp.getLoc();
    scf::ForallOp newForallOp = rewriter.create<scf::ForallOp>(
        loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
        forallOp.getMixedStep(), loop.getInitArgs(), forallOp.getMappingAttr());

    {
      // RAII guard, inserting within forallOp, before terminator.
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(newForallOp.getTerminator());
      SmallVector<Value> newInits;
      for (auto [iterArgId, slice] : llvm::enumerate(pairedSlices)) {
        if (slice) {
          newInits.push_back(slice.value().getResult());
          continue;
        }

        // If there is no paired slice (for a single trip count loop) then
        // use the iter arg of the forall op directly.
        newInits.push_back(newForallOp.getRegionIterArgs()[iterArgId]);
      }
      // Step 4. Create a new for loop with new inits for the result of the
      // extracted slices.
      auto newLoop = rewriter.create<scf::ForOp>(
          loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
          loop.getStep(), newInits,
          [](OpBuilder &, Location, Value, ValueRange) {});

      {
        // Step 5. Inline the body of the original forall into the new for loop.
        OpBuilder::InsertionGuard g2(rewriter);
        SmallVector<Value> argReplacements(newForallOp.getInductionVars());
        // If there is no paired slice (for a single trip count loop) then
        // replace the iter arg of the forall op directly.
        for (auto [forallIterArg, forIterArg, maybeSlice] :
             llvm::zip_equal(newForallOp.getRegionIterArgs(),
                             newLoop.getRegionIterArgs(), pairedSlices)) {
          if (maybeSlice) {
            argReplacements.push_back(forallIterArg);
          } else {
            argReplacements.push_back(forIterArg);
          }
        }
        rewriter.mergeBlocks(forallOp.getBody(), newLoop.getBody(),
                             argReplacements);
        rewriter.replaceAllUsesWith(loop.getInductionVar(),
                                    newLoop.getInductionVar());
        // Replace the users of the to-be-hoisted slices with the loop iter
        // args.
        for (auto [hoistedSlice, iterArg] :
             llvm::zip_equal(pairedSlices, newLoop.getRegionIterArgs())) {
          if (hoistedSlice) {
            rewriter.replaceAllUsesExcept(hoistedSlice.value(), iterArg,
                                          newLoop);
          }
        }

        // Create the terminator for the new loop using the sources of the
        // parallel inserts.
        SmallVector<Value> newYields;
        for (auto parallelSlice : terminators) {
          newYields.push_back(parallelSlice.getSource());
        }
        rewriter.setInsertionPointToEnd(newLoop.getBody());
        rewriter.create<scf::YieldOp>(loop.getLoc(), newYields);
      }

      // Move all producers for the indices of the slices outside of the body
      // of the loop (and the extract_slice ops themselves).
      for (auto sliceOperandProducer : slice) {
        rewriter.moveOpBefore(sliceOperandProducer, newLoop);
      }
      for (auto slice : pairedSlices) {
        if (slice) {
          rewriter.moveOpBefore(slice.value(), newLoop);
        }
      }

      // Create the new terminator for the hoisted forall loop using the results
      // of the new for loop.
      rewriter.setInsertionPointToEnd(newForallOp.getTerminator().getBody());
      for (auto [parallelSlice, source, dest] :
           llvm::zip_equal(terminators, newLoop.getResults(),
                           newForallOp.getRegionIterArgs())) {
        rewriter.create<tensor::ParallelInsertSliceOp>(
            parallelSlice.getLoc(), source, dest, parallelSlice.getOffsets(),
            parallelSlice.getSizes(), parallelSlice.getStrides(),
            parallelSlice.getStaticOffsets(), parallelSlice.getStaticSizes(),
            parallelSlice.getStaticStrides());
      }
    }

    // Step 6. Erase the original terminator and replace the loop with
    // the hoisted loop.
    for (auto parallelSlice : terminators) {
      rewriter.eraseOp(parallelSlice);
    }
    rewriter.eraseOp(parallelTerminator);
    rewriter.replaceOp(loop, newForallOp);
    return success();
  }
};
} // namespace

void populateForallLoopHoistingPattern(RewritePatternSet &patterns) {
  patterns.insert<HoistForallFromFor>(patterns.getContext());
}

} // namespace mlir::iree_compiler
