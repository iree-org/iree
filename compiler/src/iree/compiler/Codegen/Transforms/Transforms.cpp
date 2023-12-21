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

#include "iree/compiler/Codegen/Dialect/IREECodegenOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

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
std::optional<Value>
hoistOneStaticallyBoundAllocation(func::FuncOp funcOp, OpBuilder &builder,
                                  Location loc, MemRefType allocLikeType,
                                  ValueRange dynamicSizes,
                                  std::optional<uint64_t> alignment) {
  IntegerAttr alignmentAttr =
      alignment ? builder.getI64IntegerAttr(alignment.value()) : nullptr;
  // For static case just create a new allocation in the entry block of the same
  // size. No need to insert a subview.
  if (dynamicSizes.empty()) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    Value allocation =
        builder.create<AllocLikeOpType>(loc, allocLikeType, alignmentAttr);
    if (std::is_same<AllocLikeOpType, memref::AllocOp>::value) {
      builder.setInsertionPoint(funcOp.getBody().front().getTerminator());
      builder.create<memref::DeallocOp>(loc, allocation);
    }
    return allocation;
  }

  /// For the dynamic but bounded case, insert an allocation of the shape of the
  /// bounds, and a subview of the required size to be used as a replacement.
  SmallVector<int64_t> staticShape;
  SmallVector<OpFoldResult> subviewSizes;
  staticShape.reserve(allocLikeType.getRank());
  subviewSizes.reserve(allocLikeType.getRank());

  int index = 0;
  for (auto dimSize : allocLikeType.getShape()) {
    if (!ShapedType::isDynamic(dimSize)) {
      staticShape.push_back(dimSize);
      subviewSizes.push_back(builder.getIndexAttr(dimSize));
      continue;
    }
    Value dynamicSize = dynamicSizes[index++];
    auto ub = ValueBoundsConstraintSet::computeConstantBound(
        presburger::BoundType::UB, dynamicSize, /*dim=*/std::nullopt,
        /*stopCondition=*/nullptr, /*closedUB=*/true);
    if (failed(ub)) {
      return std::nullopt;
    }
    staticShape.push_back(ub.value());
    subviewSizes.push_back(dynamicSize);
  }
  SmallVector<OpFoldResult> offsets(allocLikeType.getRank(),
                                    builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(allocLikeType.getRank(),
                                    builder.getIndexAttr(1));

  Value allocation;
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    auto allocationType =
        MemRefType::get(staticShape, allocLikeType.getElementType());
    allocation =
        builder.create<AllocLikeOpType>(loc, allocationType, alignmentAttr);
  }

  Value subviewOp = builder.create<memref::SubViewOp>(loc, allocation, offsets,
                                                      subviewSizes, strides);

  if (std::is_same<AllocLikeOpType, memref::AllocOp>::value) {
    builder.setInsertionPoint(funcOp.getBody().front().getTerminator());
    builder.create<memref::DeallocOp>(loc, allocation);
  }
  return subviewOp;
}

template <typename AllocLikeOpType>
std::optional<Value>
hoistOneStaticallyBoundAllocation(func::FuncOp funcOp, OpBuilder &builder,
                                  AllocLikeOpType allocLikeOp) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(allocLikeOp);
  return hoistOneStaticallyBoundAllocation<AllocLikeOpType>(
      funcOp, builder, allocLikeOp.getLoc(), allocLikeOp.getType(),
      allocLikeOp.getDynamicSizes(), allocLikeOp.getAlignment());
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
void hoistStaticallyBoundAllocationsInFunc(RewriterBase &rewriter,
                                           func::FuncOp funcOp) {
  SmallVector<AllocLikeOpType> allocLikeOps;

  // Collect all allocLikes that are hoistable.
  funcOp.walk([&](AllocLikeOpType allocLikeOp) {
    if (allocLikeOp->getBlock() == &funcOp.getBody().front())
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
    std::optional<Value> replacement =
        hoistOneStaticallyBoundAllocation(funcOp, rewriter, allocLikeOp);
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
    func::FuncOp funcOp, OpBuilder &builder, Location loc,
    MemRefType allocLikeType, ValueRange dynamicSizes,
    std::optional<uint64_t> alignment);
template std::optional<Value>
hoistOneStaticallyBoundAllocation<memref::AllocaOp>(
    func::FuncOp funcOp, OpBuilder &builder, Location loc,
    MemRefType allocLikeType, ValueRange dynamicSizes,
    std::optional<uint64_t> alignment);
template std::optional<Value>
hoistOneStaticallyBoundAllocation<memref::AllocOp>(func::FuncOp funcOp,
                                                   OpBuilder &builder,
                                                   memref::AllocOp allocLikeOp);
template std::optional<Value>
hoistOneStaticallyBoundAllocation<memref::AllocaOp>(
    func::FuncOp funcOp, OpBuilder &builder, memref::AllocaOp allocLikeOp);
template void
hoistStaticallyBoundAllocationsInFunc<memref::AllocOp>(RewriterBase &rewriter,
                                                       func::FuncOp funcOp);
template void
hoistStaticallyBoundAllocationsInFunc<memref::AllocaOp>(RewriterBase &rewriter,
                                                        func::FuncOp funcOp);

//===---------------------------------------------------------------------===//
// Lowering `flow.dispatch.workgroup_count_from_slice` operation.
//===---------------------------------------------------------------------===//

LogicalResult lowerWorkgroupCountFromSliceOp(
    RewriterBase &rewriter,
    IREE::Flow::DispatchWorkgroupCountFromSliceOp workgroupCountOp,
    func::FuncOp entryPointFn, ArrayRef<OpFoldResult> workgroupCount,
    int maxWorkgroupParallelDims) {
  // Compute the backward slice of the workgroup count operations.
  BackwardSliceOptions options;
  options.filter = [](Operation *op) {
    return !isa<IREE::Flow::DispatchWorkloadOrdinalOp>(op);
  };
  options.inclusive = true;
  llvm::SetVector<Operation *> slice;
  for (auto ofr : workgroupCount) {
    if (auto val = ofr.dyn_cast<Value>()) {
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
    if (auto val = ofr.dyn_cast<Value>()) {
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
    RewriterBase &rewriter, func::FuncOp entryPointFn,
    ArrayRef<OpFoldResult> workgroupCount, int maxWorkgroupParallelDims) {
  FailureOr<IREE::HAL::ExecutableExportOp> exportOp =
      getEntryPoint(entryPointFn);
  if (failed(exportOp)) {
    return entryPointFn.emitOpError(
        "expected function to be entry point function");
  }
  Block *body = exportOp->getWorkgroupCountBody();
  if (!body) {
    return exportOp->emitOpError("unexpected empty workgroup count region");
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
        subspanOp.getLoc(), newSubspanType, subspanOp.getSet(),
        subspanOp.getBinding(), subspanOp.getDescriptorType(),
        subspanOp.getByteOffset(), subspanOp.getDynamicDims(),
        subspanOp.getAlignmentAttr(), subspanOp.getDescriptorFlagsAttr());

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
        reshapeOp, reshapeOp.getResultType(), newSubspanOp,
        loadOp.getSourceDims());

    return success();
  }
};

/// Folds tensor.expand/collapse_shape into the source
/// hal.interface.binding.subspan.
///
/// For example, this matches the following pattern:
///
///   %subspan = hal.interface.binding.subspan ... :
///       !flow.dispatch.tensor<writeonly:tensor<3x3x1x96xf32>>
///   %0 = linalg.tensor_reshape %tensor [
///         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
///       ] : tensor<864xf32> into tensor<3x3x1x96xf32>
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
struct FoldReshapeIntoInterfaceTensorStore
    : OpRewritePattern<IREE::Flow::DispatchTensorStoreOp> {
  using OpRewritePattern<IREE::Flow::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Flow::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Make sure we are storing the full incoming subspan. Otherwise we cannot
    // simply adjust the subspan's resultant type later.
    if (!storeOp.offsets().empty() || !storeOp.sizes().empty() ||
        !storeOp.strides().empty())
      return failure();

    auto reshapeOp = storeOp.getValue().getDefiningOp();
    if (!isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(reshapeOp))
      return failure();

    // Dynamic shapes are currently unsupported.
    std::optional<Value> reshapeSrc =
        isa<tensor::CollapseShapeOp>(reshapeOp)
            ? getStaticReshapeOpSrc<tensor::CollapseShapeOp>(
                  cast<tensor::CollapseShapeOp>(reshapeOp))
            : getStaticReshapeOpSrc<tensor::ExpandShapeOp>(
                  cast<tensor::ExpandShapeOp>(reshapeOp));
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
          subspanOp.getLoc(), newSubspanType, subspanOp.getSet(),
          subspanOp.getBinding(), subspanOp.getDescriptorType(),
          subspanOp.getByteOffset(), subspanOp.getDynamicDims(),
          subspanOp.getAlignmentAttr(), subspanOp.getDescriptorFlagsAttr());
    }

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorStoreOp>(
        storeOp, *reshapeSrc, newSubspanOp, storeOp.getTargetDims());

    return success();
  }
};
} // namespace

void populateReshapeToInterfaceTensorPatterns(RewritePatternSet &patterns) {
  patterns.insert<FoldReshapeIntoInterfaceTensorLoad<tensor::CollapseShapeOp>,
                  FoldReshapeIntoInterfaceTensorLoad<tensor::ExpandShapeOp>>(
      patterns.getContext());
  patterns.insert<FoldReshapeIntoInterfaceTensorStore>(patterns.getContext());
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

void analyseAllocsForPacking(func::FuncOp funcOp, ArrayRef<Operation *> allocs,
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

void packAllocs(OpBuilder &builder, func::FuncOp funcOp,
                ArrayRef<AliasGroup> aliasGroups) {
  if (aliasGroups.empty())
    return;
  DataLayout dataLayout = DataLayout::closest(funcOp);
  builder.setInsertionPointToStart(&(*funcOp.getBody().begin()));
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

LogicalResult
tileLinalgOpsWithFilter(func::FuncOp funcOp, scf::SCFTilingOptions options,
                        IREE::LinalgExt::LinalgTransformationFilter filter) {
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
        scf::tileUsingSCFForOp(rewriter, target, options);
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

LogicalResult distributeLinalgOpsWithFilter(
    func::FuncOp funcOp, linalg::LinalgTilingOptions tilingOptions,
    IREE::LinalgExt::LinalgTransformationFilter filter) {
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

} // namespace mlir::iree_compiler
