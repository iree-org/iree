// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "CommonExtensions.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Interfaces/BufferizationInterfaces.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::IREE;

iree_compiler::IREE::transform_dialect::CommonExtensions::CommonExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsOps.cpp.inc"
      >();
}

void mlir::iree_compiler::registerTransformDialectCommonExtension(
    DialectRegistry &registry) {
  registry.addExtensions<
      mlir::iree_compiler::IREE::transform_dialect::CommonExtensions>();
}

//===---------------------------------------------------------------------===//
// ApplyIREELinalgElementwiseGreedyFusionPatternsOp
//===---------------------------------------------------------------------===//

static void addOperands(Operation *op, SetVector<Value> &operandSet) {
  if (!op)
    return;
  TypeSwitch<Operation *, void>(op)
      .Case<linalg::LinalgOp>([&](linalg::LinalgOp linalgOp) {
        SmallVector<Value> inputOperands = linalgOp.getDpsInputs();
        operandSet.insert(inputOperands.begin(), inputOperands.end());
      })
      .Default([&](Operation *operation) {
        operandSet.insert(operation->operand_begin(), operation->operand_end());
      });
}

template <int limit = 3>
static bool setFusedOpOperandLimit(OpOperand *fusedOperand) {
  Operation *producer = fusedOperand->get().getDefiningOp();
  if (!producer)
    return false;
  Operation *consumer = fusedOperand->getOwner();
  SetVector<Value> fusedOpOperands;
  if (producer->getNumResults() != 1)
    return false;
  addOperands(consumer, fusedOpOperands);
  fusedOpOperands.remove(producer->getResult(0));
  addOperands(producer, fusedOpOperands);
  return fusedOpOperands.size() <= limit;
}

void transform_dialect::ApplyIREELinalgElementwiseGreedyFusionPatternsOp::
    populatePatterns(RewritePatternSet &patterns) {
  linalg::populateElementwiseOpsFusionPatterns(patterns,
                                               setFusedOpOperandLimit<3>);
}

//===---------------------------------------------------------------------===//
// ApplyFoldFillIntoPadPatternsOp
//===---------------------------------------------------------------------===//

namespace {
/// Fold `tensor.pad(cst, tensor.extract*(linalg.fill(cst)))` into
/// `linalg.fill(cst, empty)` when the padding constant and the fill constant
/// are the same.
/// This seems generally desirable as a folding but may be too intrusive, so we
/// only apply it selectively for now.
// TODO: atm hardcoded on linalg.fill but we could take any result of any
// generic that yields a constant in that result.
struct FoldFillIntoPad : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const final {
    Operation *currentOp = padOp.getSource().getDefiningOp();
    auto maybeExtractSlice =
        dyn_cast_or_null<tensor::ExtractSliceOp>(currentOp);
    while (currentOp && maybeExtractSlice) {
      currentOp = maybeExtractSlice.getSource().getDefiningOp();
      maybeExtractSlice = dyn_cast_or_null<tensor::ExtractSliceOp>(currentOp);
    }
    auto fillOp = dyn_cast_or_null<linalg::FillOp>(currentOp);
    if (!fillOp) {
      return rewriter.notifyMatchFailure(
          padOp, "not coming from a linalg.fill op via tensor.extract_slice*");
    }

    Value padValue = padOp.getConstantPaddingValue();
    RankedTensorType resultType = padOp.getResultType();
    if (!padValue ||
        getAsOpFoldResult(padValue) !=
            getAsOpFoldResult(fillOp.getDpsInputOperand(0)->get())) {
      return rewriter.notifyMatchFailure(
          padOp, "not a constant value matching the fill value");
    }

    Location loc = padOp.getLoc();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(
        loc, tensor::getMixedSizes(rewriter, loc, padOp),
        resultType.getElementType());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(padOp, padValue,
                                                emptyOp.getResult());

    return success();
  }
};
} // namespace

void transform_dialect::ApplyFoldFillIntoPadPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  patterns.insert<FoldFillIntoPad>(patterns.getContext());
}

//===---------------------------------------------------------------------===//
// ApplyHoistForallFromForPatternsOp
//===---------------------------------------------------------------------===//

namespace {
struct HoistForallFromFor : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const final {
    if (loop.getBody()->getOperations().size() != 2) {
      return rewriter.notifyMatchFailure(
          loop, "Body of the loop contains more than one op");
    }

    // TODO(qedawkins): It should be fine to hoist as long as there is a single
    // forall op such that any operation within the block of the parent scf.for
    // is independent of the destination of the forall.
    auto forallOp =
        dyn_cast<scf::ForallOp>(loop.getBody()->without_terminator().begin());
    if (!forallOp) {
      return rewriter.notifyMatchFailure(
          loop, "Loop single contained op is not scf.forall op");
    }

    if (forallOp.getNumResults() != 1 || loop.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          loop, "unimplemented: multi-result hoisting");
    }

    if (loop.getBody()->getTerminator()->getOperand(0).getDefiningOp() !=
        forallOp) {
      return rewriter.notifyMatchFailure(
          loop, "Single for loop return is not forall result");
    }

    // Step 1. Collect the set of tensor.parallel_insert_slice ops in the
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
    SmallVector<tensor::ExtractSliceOp> pairedSlices(forallOp.getNumResults());
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
        // Fail if the destination has more users than a direct insert and
        // extract slice.
        if (!maybeSlice) {
          return failure();
        }
        // Require a single extract per destination.
        if (destSlice) {
          return failure();
        }
        destSlice = maybeSlice;
      }
      // Verify they operate on equivalent subsets, ensuring the slices are
      // hoistable. It is still possible to hoist the loop if this is not true,
      // however in such cases we likely formed the loops in the wrong order.
      if (!cast<SubsetOpInterface>(*destSlice)
               .operatesOnEquivalentSubset(
                   cast<SubsetOpInterface>(*parallelInsert),
                   [](Value v1, Value v2) { return v1 == v2; })) {
        return failure();
      }
      terminators[destBbArg.getArgNumber() - numInductionVars] = parallelInsert;
      pairedSlices[destBbArg.getArgNumber() - numInductionVars] = destSlice;

      // Collect all of the offset/size/stride operands for both slices and
      // compute a backwards slice of the program from them. Fail if any of
      // them depend on the serial loop iterator.
      llvm::SmallDenseSet<Value> sliceOperands;
      sliceOperands.insert(
          parallelInsert.getOperands().begin() +
              parallelInsert.getOffsetSizeAndStrideStartOperandIndex(),
          parallelInsert.getOperands().end());
      sliceOperands.insert(
          destSlice.getOperands().begin() +
              destSlice.getOffsetSizeAndStrideStartOperandIndex(),
          destSlice.getOperands().end());
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

    // Step 2. Create the ForallOp.
    Location loc = forallOp.getLoc();
    scf::ForallOp newForallOp = rewriter.create<scf::ForallOp>(
        loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
        forallOp.getMixedStep(), loop.getInitArgs(), forallOp.getMappingAttr());

    {
      // RAII guard, inserting within forallOp, before terminator.
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(newForallOp.getTerminator());
      SmallVector<Value> newInits;
      for (auto slice : pairedSlices) {
        newInits.push_back(slice.getResult());
      }
      // Step 3. Create a new for loop with new inits for the result of the
      // extracted slices.
      auto newLoop = rewriter.create<scf::ForOp>(
          loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
          loop.getStep(), newInits,
          [](OpBuilder &, Location, Value, ValueRange) {});

      {
        // Step 4. Inline the body of the original forall into the new for loop.
        OpBuilder::InsertionGuard g2(rewriter);
        SmallVector<Value> argReplacements(newForallOp.getInductionVars());
        argReplacements.append(newForallOp.getRegionIterArgs().begin(),
                               newForallOp.getRegionIterArgs().end());
        rewriter.mergeBlocks(forallOp.getBody(), newLoop.getBody(),
                             argReplacements);
        rewriter.replaceAllUsesWith(loop.getInductionVar(),
                                    newLoop.getInductionVar());
        // Replace the users of the to-be-hoisted slices with the loop iter
        // args.
        for (auto [hoistedSlice, iterArg] :
             llvm::zip_equal(pairedSlices, newLoop.getRegionIterArgs())) {
          rewriter.replaceAllUsesExcept(hoistedSlice, iterArg, newLoop);
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
        rewriter.moveOpBefore(slice, newLoop);
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

    // Step 5. Erase the original terminator and replace the loop with
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

void transform_dialect::ApplyHoistForallFromForPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  patterns.insert<HoistForallFromFor>(patterns.getContext());
}

//===---------------------------------------------------------------------===//
// ApplyLowerShuffleTensorPatternsOp
//===---------------------------------------------------------------------===//

namespace {
struct LowerShuffleTensor
    : public OpRewritePattern<IREE::GPU::ShuffleTensorOp> {
  using OpRewritePattern<IREE::GPU::ShuffleTensorOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::GPU::ShuffleTensorOp shuffleOp,
                                PatternRewriter &rewriter) const final {
    Location loc = shuffleOp.getLoc();

    // Step 1. Insert the source slice into the intermediate tensor.
    SmallVector<OpFoldResult, 4> sourceOffsets = shuffleOp.getMixedOffsets();
    SmallVector<OpFoldResult, 4> sourceSizes = shuffleOp.getMixedSizes();
    SmallVector<OpFoldResult, 4> sourceStrides = shuffleOp.getMixedStrides();
    Value insertedSlice = rewriter.create<tensor::InsertSliceOp>(
        loc, shuffleOp.getSource(), shuffleOp.getDest(), sourceOffsets,
        sourceSizes, sourceStrides);

    // Step 2. Synchronize the workers.
    rewriter.create<gpu::BarrierOp>(loc);

    auto terminator = shuffleOp.getBody()->getTerminator();
    Value replacement = terminator->getOperand(0);
    rewriter.inlineBlockBefore(shuffleOp.getBody(), shuffleOp, {insertedSlice});
    rewriter.replaceAllUsesWith(shuffleOp.getResult(), replacement);
    rewriter.setInsertionPointAfterValue(replacement);

    // Step 2. Synchronize the workers again after reading the shuffled values.
    // TODO: This barrier is an approximation for what we expect bufferization +
    // vectorization to produce. There is no guarantee that this barrier is
    // adhered to, but the way that bufferization and vectorization works
    // is unfriendly towards barrier-like constructs.
    rewriter.create<gpu::BarrierOp>(loc);
    rewriter.eraseOp(terminator);
    return success();
  }
};
} // namespace

void transform_dialect::ApplyLowerShuffleTensorPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  patterns.insert<LowerShuffleTensor>(patterns.getContext());
}

//===---------------------------------------------------------------------===//
// ApplyUnrollVectorsGpuMmaSyncPatternsOp
//===---------------------------------------------------------------------===//

static std::optional<SmallVector<int64_t>>
getGPUTensorCoreNativeMmaSyncVectorSize(Operation *op) {
  return getMmaNativeVectorSize(op);
}

void transform_dialect::ApplyUnrollVectorsGpuMmaSyncPatternsOp::
    populatePatterns(RewritePatternSet &patterns) {
  auto unrollOrder = [](Operation *op) -> std::optional<SmallVector<int64_t>> {
    auto contract = dyn_cast<vector::ContractionOp>(op);
    if (!contract)
      return std::nullopt;
    return mlir::iree_compiler::gpuMmaUnrollOrder(contract);
  };
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getGPUTensorCoreNativeMmaSyncVectorSize)
                    .setUnrollTraversalOrderFn(unrollOrder));
}

//===---------------------------------------------------------------------===//
// ApplyUnrollVectorsGpuWmmaSyncPatternsOp
//===---------------------------------------------------------------------===//

static std::optional<SmallVector<int64_t>>
getGPUTensorCoreNativeWmmaVectorSize(Operation *op) {
  return getWmmaNativeVectorSize(op);
}

void transform_dialect::ApplyUnrollVectorsGpuWmmaSyncPatternsOp::
    populatePatterns(RewritePatternSet &patterns) {
  auto unrollOrder = [](Operation *op) -> std::optional<SmallVector<int64_t>> {
    auto contract = dyn_cast<vector::ContractionOp>(op);
    if (!contract)
      return std::nullopt;
    return mlir::iree_compiler::gpuMmaUnrollOrder(contract);
  };
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getGPUTensorCoreNativeWmmaVectorSize)
                    .setUnrollTraversalOrderFn(unrollOrder));
}

//===---------------------------------------------------------------------===//
// Remaining Apply...PatternsOp
//===---------------------------------------------------------------------===//

void transform_dialect::ApplyBubbleCollapsePatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  linalg::populateFoldReshapeOpsByCollapsingPatterns(
      patterns, [](OpOperand *) { return true; });
}

void transform_dialect::ApplyBubbleExpandPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  linalg::populateFoldReshapeOpsByExpansionPatterns(
      patterns, [](OpOperand *) { return true; });
}

void transform_dialect::ApplyBubblePackUnpackPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  linalg::populateDataLayoutPropagationPatterns(
      patterns, [](Operation *op) { return true; });
}

void transform_dialect::ApplyFoldReshapeIntoTensorHalInterfacePatternsOp::
    populatePatterns(RewritePatternSet &patterns) {
  populateReshapeToInterfaceTensorPatterns(patterns);
}

void transform_dialect::ApplyFoldTensorSliceIntoTransferPatternsOp::
    populatePatterns(RewritePatternSet &patterns) {
  populateVectorTransferTensorSliceTransforms(patterns);
}

void transform_dialect::ApplyPrepareVectorToMMAPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  populatePrepareVectorToMMAPatterns(patterns, getUseNvGpu());
}

//===----------------------------------------------------------------------===//
// CopyTensorOperandOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_dialect::CopyTensorOperandOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  int64_t operandIndex = getOperandIndex();
  if (operandIndex > target->getNumOperands()) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "Operand index out of range");
  }
  Value operand = target->getOperand(operandIndex);
  auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
  if (!tensorType) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "Non tensor type operand to copy");
  }
  rewriter.setInsertionPoint(target);
  Value empty = rewriter.create<tensor::EmptyOp>(
      target->getLoc(),
      tensor::getMixedSizes(rewriter, target->getLoc(), operand),
      tensorType.getElementType());
  Operation *copy =
      rewriter.create<linalg::CopyOp>(target->getLoc(), operand, empty);
  target->setOperand(operandIndex, copy->getResult(0));
  results.push_back(copy);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::CopyTensorOperandOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::producesHandle(getResult(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// FlattenForallOp
//===---------------------------------------------------------------------===//

template <typename MappingTy>
static bool isAscendingRelativeMapping(ArrayRef<Attribute> mapping) {
  auto start = cast<MappingTy>(*mapping.begin());
  if (start.getMappingId() !=
      static_cast<int64_t>(gpu::MappingId::LinearDim0)) {
    return false;
  }
  int64_t prev = start.getMappingId();
  for (auto id : llvm::drop_begin(mapping)) {
    int64_t nextId = cast<MappingTy>(id).getMappingId();
    if (nextId != prev + 1) {
      return false;
    }
    prev = nextId;
  }
  return true;
}

static FailureOr<scf::ForallOp> flattenForallOp(RewriterBase &rewriter,
                                                scf::ForallOp forallOp) {
  if (!forallOp.getMapping().has_value())
    return forallOp->emitError("mapping must be present");
  SmallVector<Attribute> mapping =
      llvm::to_vector(forallOp.getMapping()->getValue());
  if (!(llvm::all_of(mapping, llvm::IsaPred<gpu::GPUThreadMappingAttr>) ||
        llvm::all_of(mapping, llvm::IsaPred<gpu::GPUWarpMappingAttr>))) {
    return forallOp->emitError("mapping must be #gpu.thread or #gpu.warp");
  }

  if (forallOp.getRank() == 1) {
    return forallOp;
  }

  bool isThreadMapping = isa<gpu::GPUThreadMappingAttr>(*mapping.begin());

  if (isThreadMapping &&
      !isAscendingRelativeMapping<gpu::GPUThreadMappingAttr>(mapping)) {
    return forallOp->emitError("mapping must be ascending linear thread ids");
  }
  if (!isThreadMapping &&
      !isAscendingRelativeMapping<gpu::GPUWarpMappingAttr>(mapping)) {
    return forallOp->emitError("mapping must be ascending linear warp ids");
  }

  auto isAll = [](ArrayRef<int64_t> array, int64_t cmp) {
    return llvm::all_of(array, [cmp](int64_t x) { return x == cmp; });
  };

  if (!isAll(forallOp.getStaticStep(), 1) ||
      !isAll(forallOp.getStaticLowerBound(), 0)) {
    return forallOp->emitError(
        "unimplemented: trying to flatten non-normalized forall op");
  }

  MLIRContext *context = rewriter.getContext();
  Location loc = forallOp.getLoc();

  // Step 1. Construct the new mapping attribute as a single linear dim of the
  // original type.
  Attribute flatMapping;
  if (isThreadMapping) {
    flatMapping =
        gpu::GPUThreadMappingAttr::get(context, gpu::MappingId::LinearDim0);
  } else {
    flatMapping =
        gpu::GPUWarpMappingAttr::get(context, gpu::MappingId::LinearDim0);
  }

  // Step 2. Compute the flat lower bound/upper bound/step.
  AffineExpr d0, d1;
  bindDims(context, d0, d1);
  AffineExpr mulExpr = d0 * d1;
  SmallVector<OpFoldResult> upperBounds = forallOp.getMixedUpperBound();
  OpFoldResult newUpperBound = upperBounds.front();
  for (OpFoldResult ub : llvm::drop_begin(upperBounds)) {
    newUpperBound = affine::makeComposedFoldedAffineApply(
        rewriter, loc, mulExpr, {newUpperBound, ub});
  }

  OpFoldResult zero = rewriter.getIndexAttr(0);
  OpFoldResult one = rewriter.getIndexAttr(1);

  // Step 3. Create a new parallel loop with a single mapping id.
  auto newForallOp = rewriter.create<scf::ForallOp>(
      loc, ArrayRef<OpFoldResult>{zero}, ArrayRef<OpFoldResult>{newUpperBound},
      ArrayRef<OpFoldResult>{one}, forallOp.getOutputs(),
      rewriter.getArrayAttr({flatMapping}));

  rewriter.setInsertionPointToStart(newForallOp.getBody());
  Value linearId = newForallOp.getInductionVar(0);

  // Step 4. Delinearize the flat ID to the original basis.
  auto ids = rewriter.create<affine::AffineDelinearizeIndexOp>(
      loc, linearId, forallOp.getMixedUpperBound());

  // Step 5. Inline the region of the original forall op.
  SmallVector<Value> newArgs(ids.getResults());
  newArgs.append(newForallOp.getRegionIterArgs().begin(),
                 newForallOp.getRegionIterArgs().end());
  rewriter.eraseOp(newForallOp.getTerminator());
  rewriter.mergeBlocks(forallOp.getBody(), newForallOp.getBody(), newArgs);
  rewriter.replaceOp(forallOp, newForallOp);
  return newForallOp;
}

DiagnosedSilenceableFailure
transform_dialect::FlattenForallMappingOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::scf::ForallOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {

  rewriter.setInsertionPoint(target);
  auto newForallOp = flattenForallOp(rewriter, target);
  if (failed(newForallOp)) {
    return mlir::emitDefiniteFailure(target, "flattenForallOp failed");
  }

  results.push_back(*newForallOp);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::FlattenForallMappingOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::producesHandle(getResult(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// ForallToWorkgroupOp
//===---------------------------------------------------------------------===//

LogicalResult rewriteForallToWorkgroup(RewriterBase &rewriter,
                                       scf::ForallOp forallOp) {
  // Step 0. Target-specific verifications. There is no good place to anchor
  // those right now: the ForallOp is target-independent and the
  // transform op does not apply to individual ForallOp.
  MLIRContext *ctx = forallOp->getContext();
  Location loc = forallOp->getLoc();
  // TODO iree should have own device mapping like #hal.workgroup<x/y/z>
  Attribute bX = gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::DimX);
  Attribute bY = gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::DimY);
  Attribute bZ = gpu::GPUBlockMappingAttr::get(ctx, gpu::MappingId::DimZ);
  if (forallOp.getNumResults() > 0)
    return forallOp->emitError(
        "only bufferized scf.forall lowers to workgroup");
  if (forallOp.getRank() > 3)
    return forallOp->emitError(
        "scf.forall with rank > 3 does not lower to workgroup");

  if (!forallOp.getMapping().has_value())
    return forallOp->emitError("mapping must be present");
  SmallVector<Attribute> blockMapping =
      llvm::to_vector(forallOp.getMapping()->getValue());
  if (llvm::any_of(blockMapping, [](Attribute map) {
        return !llvm::isa<gpu::GPUBlockMappingAttr>(map);
      })) {
    return forallOp->emitError("mapping must be #gpu.block<x/y/z/>");
  }

  // Step 1. Complete the blockMapping to a full mapping (with 1s) if
  // necessary.
  SmallVector<Value> numBlocks =
      llvm::to_vector(forallOp.getUpperBound(rewriter));
  // Ensure we have 3 block sizes, one for each id.
  Value one;
  for (auto attr : {bX, bY, bZ}) {
    if (!llvm::is_contained(blockMapping, attr)) {
      blockMapping.push_back(attr);
      one = one ? one : rewriter.create<arith::ConstantIndexOp>(loc, 1);
      numBlocks.push_back(one);
    }
  }
  // Step 2. sort the values by the corresponding GPUBlockMappingAttr.
  auto comparator = [](Attribute a, Attribute b) -> bool {
    return static_cast<int64_t>(
               llvm::cast<gpu::GPUBlockMappingAttr>(a).getBlock()) <
           static_cast<int64_t>(
               llvm::cast<gpu::GPUBlockMappingAttr>(b).getBlock());
  };
  SmallVector<Value> gridDimValues =
      getValuesSortedByKey(blockMapping, numBlocks, comparator);

  // Step 3. Create the workgroup id and count ops.
  IRMapping bvm;
  SmallVector<Value> workgroupIdOps, workgroupCountOps;
  for (Attribute attr : blockMapping) {
    auto idx = static_cast<int64_t>(
        llvm::cast<gpu::GPUBlockMappingAttr>(attr).getBlock());
    workgroupIdOps.push_back(
        rewriter.create<HAL::InterfaceWorkgroupIDOp>(loc, idx));
    workgroupCountOps.push_back(
        rewriter.create<HAL::InterfaceWorkgroupCountOp>(loc, idx));
  }
  bvm.map(forallOp.getInductionVars(), workgroupIdOps);
  bvm.map(forallOp.getUpperBound(rewriter), workgroupCountOps);

  // Step 4. Predicate omitted given unique topLevel scf::ForallOp.

  // Step 5. Move the body of forallOp.
  // Erase the terminator first, it will not be used since we are on buffers.
  rewriter.eraseOp(forallOp.getTerminator());
  Block *targetBlock;
  Block::iterator insertionPoint;
  targetBlock = forallOp->getBlock();
  insertionPoint = Block::iterator(forallOp);
  Block &sourceBlock = forallOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 6. RAUW thread indices to thread ops.
  for (Value blockIdx : forallOp.getInductionVars()) {
    for (Operation *user : llvm::make_early_inc_range(blockIdx.getUsers())) {
      rewriter.modifyOpInPlace(user, [&]() {
        user->replaceUsesOfWith(blockIdx, bvm.lookup(blockIdx));
      });
    }
  }

  // Step 6. Barriers omitted given unique topLevel scf::ForallOp.

  // Step 7. Erase old op.
  rewriter.eraseOp(forallOp);

  return success();
}

DiagnosedSilenceableFailure transform_dialect::ForallToWorkgroupOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::FunctionOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {

  scf::ForallOp topLevelForallOp;
  auto walkResult = target->walk([&](scf::ForallOp forallOp) {
    if (forallOp->getParentOfType<scf::ForallOp>())
      return WalkResult::advance();
    if (topLevelForallOp)
      return WalkResult::interrupt();
    topLevelForallOp = forallOp;
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return mlir::emitSilenceableFailure(
        target, "could not find a unique topLevel scf.forall");
  }

  rewriter.setInsertionPoint(topLevelForallOp);
  if (failed(rewriteForallToWorkgroup(rewriter, topLevelForallOp)))
    return mlir::emitDefiniteFailure(target, "rewriteForallToWorkgroup failed");

  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::ForallToWorkgroupOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// FuseForallOp
//===---------------------------------------------------------------------===//

FailureOr<int64_t> getTripCount(scf::ForallOp loop) {
  ArrayRef<int64_t> lbs = loop.getStaticLowerBound();
  ArrayRef<int64_t> ubs = loop.getStaticUpperBound();
  ArrayRef<int64_t> steps = loop.getStaticStep();

  if (ShapedType::isDynamicShape(lbs) || ShapedType::isDynamicShape(ubs) ||
      ShapedType::isDynamicShape(steps)) {
    return failure();
  }

  int64_t tripCount = 1;
  for (auto [lb, ub, step] : llvm::zip_equal(lbs, ubs, steps)) {
    tripCount *= mlir::ceilDiv((ub - lb), step);
  }
  return tripCount;
}

LogicalResult compareWorkerCountsAndTypes(scf::ForallOp producer,
                                          scf::ForallOp consumer) {
  FailureOr<int64_t> producerTripCount = getTripCount(producer);
  FailureOr<int64_t> consumerTripCount = getTripCount(consumer);
  if (failed(producerTripCount) || failed(consumerTripCount) ||
      *producerTripCount != *consumerTripCount) {
    return failure();
  }

  auto checkMappingTypes = [&](ArrayAttr array) {
    return llvm::all_of(array.getValue(),
                        llvm::IsaPred<gpu::GPUThreadMappingAttr>) ||
           llvm::all_of(array.getValue(),
                        llvm::IsaPred<gpu::GPUWarpMappingAttr>);
  };

  if (producer.getMappingAttr() != consumer.getMappingAttr() ||
      !checkMappingTypes(producer.getMappingAttr()) ||
      !checkMappingTypes(consumer.getMappingAttr())) {
    return failure();
  }
  return success();
}

void replaceExtractSlice(RewriterBase &rewriter, Location loc,
                         tensor::ParallelInsertSliceOp parallelInsert,
                         tensor::ExtractSliceOp extractSlice) {
  OpBuilder::InsertionGuard g(rewriter);
  auto shuffleOp = rewriter.create<IREE::GPU::ShuffleTensorOp>(
      loc, extractSlice.getType(), parallelInsert.getSource(),
      parallelInsert.getOffsets(), parallelInsert.getSizes(),
      parallelInsert.getStrides(), parallelInsert.getStaticOffsets(),
      parallelInsert.getStaticSizes(), parallelInsert.getStaticStrides(),
      parallelInsert.getDest());
  Region *region = &shuffleOp.getRegion();
  rewriter.createBlock(region, region->end(),
                       ArrayRef<Type>{parallelInsert.getDestType()},
                       ArrayRef<Location>{loc});
  rewriter.setInsertionPointToStart(shuffleOp.getBody());
  auto terminator =
      rewriter.create<IREE::GPU::YieldOp>(loc, extractSlice.getResult());
  rewriter.moveOpBefore(extractSlice, terminator);
  extractSlice.getSourceMutable().assign(shuffleOp.getBody()->getArgument(0));
  rewriter.replaceAllUsesExcept(extractSlice.getResult(), shuffleOp,
                                terminator);
}

LogicalResult fuseForallIntoSlice(RewriterBase &rewriter,
                                  scf::ForallOp producer,
                                  scf::ForallOp consumer,
                                  tensor::ExtractSliceOp slice,
                                  std::optional<Attribute> addressSpace) {
  if (producer->getNumResults() != 1) {
    return failure();
  }

  if (failed(compareWorkerCountsAndTypes(producer, consumer))) {
    return failure();
  }

  auto isAll = [](ArrayRef<OpFoldResult> array, int64_t cmp) {
    return llvm::all_of(array, [cmp](OpFoldResult val) {
      return isConstantIntValue(val, cmp);
    });
  };

  if (!isAll(producer.getMixedStep(), 1) ||
      !isAll(producer.getMixedLowerBound(), 0) ||
      !isAll(consumer.getMixedStep(), 1) ||
      !isAll(consumer.getMixedLowerBound(), 0)) {
    return failure();
  }

  rewriter.setInsertionPoint(slice);

  // Step 1. Compute the producer IDs in terms of the consumer IDs.

  MLIRContext *context = rewriter.getContext();
  Location loc = producer.getLoc();

  AffineExpr d0, d1, d2;
  bindDims(context, d0, d1, d2);
  AffineExpr mulAdd = d0 * d1 + d2;
  OpFoldResult linearId = rewriter.getIndexAttr(0);
  for (auto [inductionVar, workerCount] :
       llvm::zip_equal(getAsOpFoldResult(consumer.getInductionVars()),
                       consumer.getMixedUpperBound())) {
    linearId = affine::makeComposedFoldedAffineApply(
        rewriter, loc, mulAdd, {linearId, workerCount, inductionVar});
  }

  Value linearThreadIdVal =
      getValueOrCreateConstantIndexOp(rewriter, loc, linearId);
  SmallVector<Value> ranges;
  for (auto workerCount : producer.getStaticUpperBound()) {
    ranges.push_back(rewriter.create<arith::ConstantIndexOp>(loc, workerCount));
  }
  ValueRange newIds = rewriter
                          .create<affine::AffineDelinearizeIndexOp>(
                              loc, linearThreadIdVal, ranges)
                          .getResults();

  // Step 2. Inline the region of the producer.
  SmallVector<Value> bbArgReplacements(newIds);
  bbArgReplacements.append(producer.getOutputs().begin(),
                           producer.getOutputs().end());

  scf::InParallelOp terminator = producer.getTerminator();
  rewriter.inlineBlockBefore(producer.getBody(), slice, bbArgReplacements);

  rewriter.setInsertionPointAfter(terminator);
  auto parallelInsert =
      cast<tensor::ParallelInsertSliceOp>(*terminator.getYieldingOps().begin());

  replaceExtractSlice(rewriter, loc, parallelInsert, slice);

  rewriter.eraseOp(parallelInsert);
  rewriter.eraseOp(terminator);
  rewriter.eraseOp(producer);
  return success();
}

DiagnosedSilenceableFailure
transform_dialect::FuseForallOp::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
                                       transform::TransformState &state) {
  auto producers = state.getPayloadOps(getProducer());
  auto consumers = state.getPayloadOps(getConsumer());

  int64_t numProducers = llvm::range_size(producers);
  int64_t numConsumers = llvm::range_size(consumers);
  if (numProducers != 1 || numConsumers != 1) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "More than one producer or consumer");
  }

  auto producer = dyn_cast<scf::ForallOp>(*producers.begin());
  auto consumer = dyn_cast<scf::ForallOp>(*consumers.begin());
  if (!producer || !consumer) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "Non-forall producer or consumer");
  }

  if (!producer->hasOneUse()) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "non-single use producer");
  }

  auto sliceConsumer =
      dyn_cast<tensor::ExtractSliceOp>(*producer->user_begin());
  if (!sliceConsumer || sliceConsumer->getParentOp() != consumer) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "producer loop sole consumer is not an "
                                     "extracted slice from the consumer loop");
  }

  if (failed(fuseForallIntoSlice(rewriter, producer, consumer, sliceConsumer,
                                 getAddressSpace()))) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "failed to fuse forall ops");
  }

  results.set(getOperation()->getOpResult(0), {consumer});
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::FuseForallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getProducer(), effects);
  transform::consumesHandle(getConsumer(), effects);
  transform::producesHandle(getResult(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// GpuDistributeSharedMemoryCopyOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::GpuDistributeSharedMemoryCopyOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::FunctionOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {

  // Look for ops that move to workgroup memory and mark as copies for
  // distribution.
  target.walk([&](linalg::GenericOp copyOp) {
    if (copyOp.getNumDpsInputs() != 1 || copyOp.getNumDpsInits() != 1)
      return;
    auto dest =
        dyn_cast<TypedValue<MemRefType>>(copyOp.getDpsInitOperand(0)->get());
    if (!dest)
      return;

    MemRefType destType = dest.getType();

    // Check if the only operation in the possible copy op region is a
    // terminator.
    Block &body = copyOp.getRegion().front();
    if (!std::begin(body)->hasTrait<OpTrait::IsTerminator>())
      return;

    auto destSpace =
        dyn_cast_or_null<gpu::AddressSpaceAttr>(destType.getMemorySpace());
    if (!destSpace)
      return;

    // The destination space must be shared memory.
    if (destSpace.getValue() != gpu::GPUDialect::getWorkgroupAddressSpace())
      return;

    // Mark this copy operation as a copy to workgroup memory.
    setMarker(copyOp, getCopyToWorkgroupMemoryMarker());
  });

  if (failed(mlir::iree_compiler::gpuDistributeSharedMemoryCopy(target))) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "Pattern failed to apply");
  }
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::GpuDistributeSharedMemoryCopyOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// HoistStaticAllocOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_dialect::HoistStaticAllocOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::FunctionOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  mlir::iree_compiler::hoistStaticallyBoundAllocationsInFunc<memref::AllocOp>(
      rewriter, target);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::HoistStaticAllocOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// PopulateWorkgroupCountRegionUsingNumThreadsSliceOp
//===---------------------------------------------------------------------===//

void transform_dialect::PopulateWorkgroupCountRegionUsingNumThreadsSliceOp::
    getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getForAllOp(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure
transform_dialect::PopulateWorkgroupCountRegionUsingNumThreadsSliceOp::
    applyToOne(transform::TransformRewriter &rewriter, Operation *target,
               transform::ApplyToEachResultList &results,
               transform::TransformState &state) {
  auto forAllOp = dyn_cast<scf::ForallOp>(target);
  if (!forAllOp) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "expected scf.forall operation handle");
  }
  if (!forAllOp.isNormalized()) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "Expect the for op to be normalized");
  }
  auto workgroupCount =
      getMixedValues(forAllOp.getStaticUpperBound(),
                     forAllOp.getDynamicUpperBound(), rewriter);

  // Account for mapping attribute if present. The attribute used for mapping
  // provides a mapping ID that is ordered in `x` = 0, `y`=1, and `z` = 2. Use
  // this to shuffle the workgroup count around.
  if (auto blockMapping = forAllOp.getMapping()) {
    // Get the mapping IDs.
    auto mappingIds = llvm::map_to_vector(
        blockMapping.value(), [](Attribute mappingAttr) -> int {
          return llvm::cast<DeviceMappingAttrInterface>(mappingAttr)
              .getMappingId();
        });
    int maxId = 0;
    for (auto id : mappingIds) {
      maxId = std::max(maxId, id);
    }
    SmallVector<OpFoldResult> workgroupCountOrdered(maxId + 1,
                                                    rewriter.getIndexAttr(1));
    for (auto [index, mapId] : llvm::enumerate(mappingIds)) {
      workgroupCountOrdered[maxId - mapId] = workgroupCount[index];
    }
    workgroupCount = workgroupCountOrdered;
  }

  auto funcOp = forAllOp->getParentOfType<mlir::FunctionOpInterface>();
  if (failed(
          lowerWorkgroupCountFromSliceOp(rewriter, funcOp, workgroupCount))) {
    return mlir::emitDefiniteFailure(state.getTopLevel(),
                                     "failed to lower workgroup count region");
  }
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// IREEApplyLoopIndependentCodeMotionOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::IREEApplyLoopIndependentCodeMotionOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  ErrorCheckingTrackingListener listener(state, *this);
  target->walk([&](mlir::FunctionOpInterface funcOp) {
    // This assumes LICM never removes operations so we don't need tracking.
    // TODO: confirm / revisit this assumption and plumb a rewriter through
    // upstream moveLoopInvariantCode if necessary.
    funcOp->walk([](LoopLikeOpInterface loopLike) {
      // Do not hoist from scf.forall ops. These capture isolated computations
      // that will be mapped to a certain level in the GPU hierarchy (e.g.,
      // GPU blocks), so hoisting is not desired.
      if (!isa<scf::ForallOp>(loopLike.getOperation()))
        moveLoopInvariantCode(loopLike);
    });
    // For now, put single loop promotion as part of licm. Underlying
    // implementations perform splice operations which shouldn't need
    // tracking.
    // TODO: confirm / revisit this assumption and plumb a rewriter through
    // upstream moveLoopInvariantCode if necessary.
    funcOp->walk([&](Operation *op) {
      (void)llvm::TypeSwitch<Operation *, LogicalResult>(op)
          .Case<affine::AffineForOp, scf::ForOp>([&](auto loop) {
            return loop.promoteIfSingleIteration(rewriter);
          })
          .Default([](Operation *) { return success(); });
    });
  });

  return listener.checkAndResetError();
}

void transform_dialect::IREEApplyLoopIndependentCodeMotionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// IREEBufferizeOp
//===---------------------------------------------------------------------===//

// Important note: this transform is load-bearing and is the glue between
// different dialects that want to operate on tensors.
// Originaly, it used to just call `addIREEComprehensiveBufferizePasses` but
// this introduces a lot of complexity in the registration process due to the
// use of nested pass pipelines, to a point that it is a major endeavor to
// connect a new dialect.
// Instead, avoid calling the passes and only take what we need from them.
//
// TODO: Maybe we need both a transform.iree.cpu.bufferize and a
// transform.iree.gpu.bufferize rather than a single common bufferize op?
//
// Note: This has become so specific that it may be worth it to separate in
// its own .cpp file.
using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::OneShotAnalysisState;
using mlir::bufferization::OneShotBufferizationOptions;

void transform_dialect::IREEBufferizeOp::build(OpBuilder &builder,
                                               OperationState &result,
                                               Value target, bool targetGpu,
                                               bool testAnalysisOnly,
                                               bool printConflicts) {
  result.addOperands(target);
  if (targetGpu) {
    result.addAttribute(IREEBufferizeOp::getTargetGpuAttrName(result.name),
                        builder.getUnitAttr());
  }
  if (testAnalysisOnly) {
    result.addAttribute(
        IREEBufferizeOp::getTestAnalysisOnlyAttrName(result.name),
        builder.getUnitAttr());
  }
  if (printConflicts) {
    result.addAttribute(IREEBufferizeOp::getPrintConflictsAttrName(result.name),
                        builder.getUnitAttr());
  }
  MLIRContext *ctx = builder.getContext();
  result.addTypes(transform::AnyOpType::get(ctx));
}

//===---------------------------------------------------------------------===//
// Default allocation functions for CPU backend
// TODO: register the bufferization behavior in a target-specific way.
// TODO: Maybe bufferize should have a separate cpu and a gpu version. This is
// unclear though: what happens on heterogeneous HW ?
//===---------------------------------------------------------------------===//

// Allocation callbacks to use with upstream comprehensive bufferization
static FailureOr<Value> cpuComprehensiveBufferizeAllocationFn(
    OpBuilder &builder, Location loc, MemRefType memRefType,
    ValueRange dynamicSizes, unsigned alignment) {
  return builder
      .create<memref::AllocaOp>(loc, memRefType, dynamicSizes,
                                builder.getI64IntegerAttr(alignment))
      .getResult();
}

static LogicalResult cpuComprehensiveBufferizeCopyFn(OpBuilder &builder,
                                                     Location loc, Value from,
                                                     Value to) {
  // TODO: ideally we should use linalg.copy which was recently reintroduced
  // as an OpDSL named op. However, IREE-specific patterns to cleanup spurious
  // post-bufferization copies do not trigger properly.
  // So we keep using `createLinalgCopyOp` which builds a GenericOp.
  // builder.create<linalg::CopyOp>(loc, from, to);
  mlir::iree_compiler::createLinalgCopyOp(builder, loc, from, to);
  return success();
}

static FailureOr<Value> gpuComprehensiveBufferizeAllocationFn(
    OpBuilder &builder, Location loc, MemRefType memRefType,
    ValueRange dynamicSizes, unsigned alignment) {
  OpBuilder::InsertionGuard g(builder);
  auto addressSpaceAttr = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  MemRefType allocType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      AffineMap(), addressSpaceAttr);
  return builder
      .create<memref::AllocOp>(loc, allocType, dynamicSizes,
                               builder.getI64IntegerAttr(alignment))
      .getResult();
}

static LogicalResult gpuComprehensiveBufferizeCopyFn(OpBuilder &builder,
                                                     Location loc, Value from,
                                                     Value to) {
  // Insert barriers for copies from and to shared memory.
  bool needsBarrier = false;
  if (hasSharedMemoryAddressSpace(llvm::cast<MemRefType>(from.getType())) !=
      hasSharedMemoryAddressSpace(llvm::cast<MemRefType>(to.getType()))) {
    needsBarrier = true;
  }
  if (needsBarrier)
    builder.create<gpu::BarrierOp>(loc);
  // TODO: ideally we should use linalg.copy which was recently reintroduced
  // as an OpDSL named op. However, IREE-specific patterns to cleanup spurious
  // post-bufferization copies do not trigger properly.
  // So we keep using `createLinalgCopyOp` which builds a GenericOp.
  // builder.create<linalg::CopyOp>(loc, from, to);
  mlir::iree_compiler::createLinalgCopyOp(builder, loc, from, to);
  if (needsBarrier)
    builder.create<gpu::BarrierOp>(loc);
  return success();
}

static IREEOneShotBufferizationOptions getBufferizationOptions() {
  IREEOneShotBufferizationOptions options;
  // options.testAnalysisOnly = testAnalysisOnly;
  // options.printConflicts = printConflicts;

  // bufferization.to_memref is used to bufferize constants in IREE. IREE has
  // it's own logic to handle constants. We'd like to leave the arith.constant
  // as is and insert bufferization.to_memref to convert the tensor to memref.
  options.opFilter.denyOperation<arith::ConstantOp>();
  options.opFilter.denyOperation<bufferization::ToMemrefOp>();

  // This type converter converts tensor types to memref types when no exact
  // memref type can be inferred from the context.
  options.unknownTypeConverterFn = [](Value value, Attribute memorySpace,
                                      const BufferizationOptions &options) {
    auto tensorType = llvm::cast<TensorType>(value.getType());

    // Special rule for ConstantOps: These always lower to some memref with a
    // static identity layout.
    if (value.getDefiningOp<arith::ConstantOp>())
      return bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType,
                                                                  memorySpace);

    // Default case: Fully dynamic layout map for best compatibility.
    return bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType,
                                                              memorySpace);
  };

  return options;
}

namespace {
/// Pattern to rewrite tensor.empty to tensor.alloc.
struct EmptyTensorLoweringPattern : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<bufferization::AllocTensorOp>(
        op, op.getType(), op.getDynamicSizes());
    return success();
  }
};
} // namespace

DiagnosedSilenceableFailure transform_dialect::IREEBufferizeOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  auto payload = state.getPayloadOps(getTarget());

  // Bufferize the dispatch.
  using mlir::bufferization::BufferizationOptions;
  BufferizationOptions::AllocationFn allocationFn =
      cpuComprehensiveBufferizeAllocationFn;
  BufferizationOptions::MemCpyFn memCpyFn = cpuComprehensiveBufferizeCopyFn;
  if (getTargetGpu()) {
    allocationFn = gpuComprehensiveBufferizeAllocationFn;
    memCpyFn = gpuComprehensiveBufferizeCopyFn;
  }

  Operation *target = *payload.begin();
  ErrorCheckingTrackingListener listener(state, *this);
  //   1. Rewrite tensor.empty to tensor.alloc, without the pass baggage.
  {
    RewritePatternSet patterns(getContext());
    patterns.add<EmptyTensorLoweringPattern>(patterns.getContext());
    GreedyRewriteConfig config;
    config.listener = &listener;
    // Manually gather list of ops because the other GreedyPatternRewriteDriver
    // overloads only accepts ops that are isolated from above.
    LogicalResult result =
        applyOpPatternsAndFold(target, std::move(patterns), config);
    if (failed(result)) {
      return mlir::emitDefiniteFailure(target,
                                       "greedy pattern application failed");
    }
    if (listener.failed())
      return listener.checkAndResetError();
  }

  //   2. Run one-shot-bufferize, without the pass baggage.
  IREEOneShotBufferizationOptions options = getBufferizationOptions();
  options.allocationFn = allocationFn;
  options.memCpyFn = memCpyFn;
  options.testAnalysisOnly = getTestAnalysisOnly();
  options.printConflicts = getPrintConflicts();

  if (getTargetGpu()) {
    options.defaultMemorySpaceFn =
        [&](TensorType t) -> std::optional<Attribute> {
      Attribute addressSpaceAttr = gpu::AddressSpaceAttr::get(
          t.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
      return addressSpaceAttr;
    };
  }
  if (failed(runIREEOneShotBufferize(target, options))) {
    return mlir::emitDefiniteFailure(target, "bufferization failed");
  }

  // Early exit if test_analysis_only is set.
  if (getTestAnalysisOnly()) {
    results.set(getOperation()->getOpResult(0), {*payload.begin()});
    return listener.checkAndResetError();
  }

  //   3. Post-bufferization passes are fine.
  PassManager pm(getContext());
  addIREEPostBufferizationPasses(pm);
  if (failed(pm.run(target))) {
    return mlir::emitDefiniteFailure(target)
           << "post-bufferization passes failed";
  }

  results.set(getOperation()->getOpResult(0), {target});
  return listener.checkAndResetError();
}

//===---------------------------------------------------------------------===//
// IREEEliminateEmptyTensorsOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::IREEEliminateEmptyTensorsOp::applyToOne(
    transform::TransformRewriter &rewriter, ::mlir::Operation *target,
    ::mlir::transform::ApplyToEachResultList &results,
    ::mlir::transform::TransformState &state) {
  if (failed(
          eliminateEmptyTensors(rewriter, target, getBufferizationOptions())))
    return emitDefaultDefiniteFailure(target)
           << "failed to eliminate tensor.empty ops";
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::IREEEliminateEmptyTensorsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ReduceSharedMemoryBankConflictsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::ReduceSharedMemoryBankConflictsOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::FunctionOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (failed(reduceSharedMemoryBankConflicts(target, getPaddingSizeBits()))) {
    return emitDefaultDefiniteFailure(target)
           << "failed to reduce shared memory bank conflicts";
  }
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::ReduceSharedMemoryBankConflictsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ShareForallOperandsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform_dialect::ShareForallOperandsOp::applyToOne(
    transform::TransformRewriter &rewriter, scf::ForallOp forallOp,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  SmallVector<int64_t> shareOperands(getShareOperands());
  // Empty case: consider all operands need to be shared.
  if (shareOperands.empty()) {
    shareOperands =
        llvm::to_vector(llvm::seq<int64_t>(0, forallOp.getOutputs().size()));
  }
  for (int64_t outputIdx : getShareOperands()) {
    if (outputIdx < 0 || outputIdx >= forallOp.getOutputs().size())
      return mlir::emitDefiniteFailure(forallOp, "operand idx overflow");
    Value toShare = forallOp.getOutputs()[outputIdx];
    if (std::distance(toShare.getUses().begin(), toShare.getUses().end()) !=
        2) {
      /*return mlir::emitSilenceableFailure(
          forallOp,
          "operand to share must have exactly 2 uses, the forall op "
          "and an extract_slice op.");*/
      continue;
    }
    tensor::ExtractSliceOp extractSliceOp;
    for (Operation *user : toShare.getUsers()) {
      extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (extractSliceOp)
        break;
    }
    if (!extractSliceOp) {
      /*return mlir::emitSilenceableFailure(
        forallOp,
        "shared operands use must be extractSliceOp.");*/
      continue;
    }
    // Get the corresponding bbArg.
    BlockArgument bbArg = forallOp.getRegionIterArgs()[outputIdx];

    // Check if the extract_slice has a matching parallel_insert_slice
    // (i.e., same source/target, offsets, sizes and strides).
    auto isMatchingParallelInsertSlice = [&](Operation &op) {
      auto insertSlice = dyn_cast<tensor::ParallelInsertSliceOp>(&op);
      if (!insertSlice)
        return false;
      if (insertSlice.getDest() != bbArg)
        return false;
      return llvm::equal(insertSlice.getMixedOffsets(),
                         extractSliceOp.getMixedOffsets()) &&
             llvm::equal(insertSlice.getMixedSizes(),
                         extractSliceOp.getMixedSizes()) &&
             llvm::equal(insertSlice.getMixedStrides(),
                         extractSliceOp.getMixedStrides());
    };
    if (llvm::none_of(forallOp.getTerminator().getYieldingOps(),
                      isMatchingParallelInsertSlice)) {
      continue;
    }

    // Promote extract_slice source to bbArg.
    rewriter.modifyOpInPlace(extractSliceOp, [&]() {
      extractSliceOp.getSourceMutable().assign(bbArg);
    });
  }

  results.push_back(forallOp);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// TestGpuVectorDistribution
//===----------------------------------------------------------------------===//

class TestVectorLayoutOptions : public VectorLayoutOptions {
public:
  TestVectorLayoutOptions(Operation *root)
      : VectorLayoutOptions(root, /*fullConversion=*/false) {}

  LogicalResult setAnchorOps(VectorLayoutAnalysis &analysis) override {
    return setAnchorOpsFromAttributes(analysis, root);
  }
};

DiagnosedSilenceableFailure
transform_dialect::TestGpuVectorDistribution::applyToOne(
    transform::TransformRewriter &rewriter, mlir::FunctionOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  TestVectorLayoutOptions options(target);
  RewritePatternSet patterns(target.getContext());

  rewriter.setInsertionPointToStart(&target.getFunctionBody().front());
  // This is a test op so we unsafely use thread_id x as the lane ID. In
  // general this should linearize the thread IDs based on the workgroup size
  // and divide by the subgroup size. i.e.
  //
  // lane_id = (tid_x + tid_y * dim_x + tid_z * dim_y * dim_x) / subgroup_size;
  Value laneId =
      rewriter.create<gpu::ThreadIdOp>(target.getLoc(), gpu::Dimension::x);

  populateGPUDistributionPatterns(patterns);
  populateGPUDistributionLayoutAttrPatterns(laneId, patterns);
  populateGPUReductionDistributionPatterns(patterns);
  // For testing we use subgroup size = 64.
  populateGPUDistributeNestedLayoutAttrPatterns(patterns, laneId,
                                                /*subgroupSize=*/64);
  populateGPUDistributeNestedLayoutContractAMDGPUPatterns(patterns);
  if (getExperimental())
    populateGPULayoutResolutionDistributionPatterns(patterns);
  if (failed(distributeVectorOps(target, patterns, options))) {
    return emitDefaultDefiniteFailure(target);
  }
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::TestGpuVectorDistribution::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// TestVectorLayoutAnalysisOp
//===----------------------------------------------------------------------===//

static void emitLayoutRemarks(VectorLayoutAnalysis &analysis,
                              mlir::FunctionOpInterface funcOp) {
  funcOp.walk([&](Operation *op) {
    // Do not emit remarks for conflict operations.
    if (isa<VectorExt::LayoutConflictResolutionOp>(op)) {
      return;
    }

    for (OpResult result : op->getOpResults()) {
      if (auto layout = analysis.getLayout<Attribute>(result)) {
        // Print layout attr to a string.
        std::string layoutStr;
        llvm::raw_string_ostream s(layoutStr);
        s << layout;
        // Emit remark.
        op->emitRemark("layout of result #" + Twine(result.getResultNumber()) +
                       " is " + s.str());
      }
    }
  });
}

DiagnosedSilenceableFailure
transform_dialect::TestVectorLayoutAnalysisOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::FunctionOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  VectorLayoutAnalysis analysis(target);
  if (setAnchorOpsFromAttributes(analysis, target).failed()) {
    return emitDefaultSilenceableFailure(target);
  }
  if (failed(analysis.run())) {
    target.emitError("layout analysis failed");
    return emitDefaultSilenceableFailure(target);
  }
  emitLayoutRemarks(analysis, target);
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::TestVectorLayoutAnalysisOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// WorkgroupSwizzleOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform_dialect::WorkgroupSwizzleOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::FunctionOpInterface target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  (void)swizzleWorkgroupsInFunc(target, getLogTile());
  return DiagnosedSilenceableFailure::success();
}

void transform_dialect::WorkgroupSwizzleOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTarget(), effects);
  transform::modifiesPayload(effects);
}

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsOps.cpp.inc"
