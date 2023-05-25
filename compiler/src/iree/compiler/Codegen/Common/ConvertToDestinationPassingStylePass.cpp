// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- ConvertToDestinationPassingStylePass.cpp ---------------------------===//
//
// Transformations that are performed before calling upstream Comprehensive
// Bufferization pass. These change the dispatch region to use destination
// passing style, mostly to get rid of `empty` ops that result in an
// allocation.
//
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/BufferizationAnalysis.h"
#include "iree/compiler/Codegen/Common/CommonPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {
class ConvertToDestinationPassingStylePass
    : public ConvertToDestinationPassingStyleBase<
          ConvertToDestinationPassingStylePass> {
 public:
  ConvertToDestinationPassingStylePass() = default;
  ConvertToDestinationPassingStylePass(bool useWARForCooperativeMatrixCodegen) {
    this->useWARForCooperativeMatrixCodegen = useWARForCooperativeMatrixCodegen;
  }
  ConvertToDestinationPassingStylePass(
      const ConvertToDestinationPassingStylePass &pass) {
    useWARForCooperativeMatrixCodegen = pass.useWARForCooperativeMatrixCodegen;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<linalg::LinalgDialect, bufferization::BufferizationDialect>();
  }
  void runOnOperation() override;
};
}  // namespace

/// Returns the subview into the buffer that is supposed to be populated with
/// the `value` of the `flow.dispatch.tensor.store` operation. This can be used
/// to compute the results in place.
static Value getTensorLoadOpForTensorStoreOp(
    OpBuilder &b, IREE::Flow::DispatchTensorStoreOp storeOp) {
  // Clone the offset, size and stride values. They will be CSE-ed later.
  SliceAndDynamicDims clonedVals = cloneOffsetsSizesAndStrides(b, storeOp);
  Value tensorLoadOp = b.create<IREE::Flow::DispatchTensorLoadOp>(
      storeOp.getLoc(),
      llvm::cast<RankedTensorType>(storeOp.getValue().getType()),
      storeOp.getTarget(), clonedVals.dynamicDims, clonedVals.offsets,
      clonedVals.sizes, clonedVals.strides);
  return tensorLoadOp;
}

/// Gets the reverse of a `tensor.expand_shape`/`tensor.collapse_shape` op to
/// get a memref type that can be used for in-place computation of the result
/// of a dispatch region.
template <typename TensorReshapeOpTy>
static Value getReverseOfReshapeOp(OpBuilder &b, TensorReshapeOpTy reshapeOp,
                                   Value resultBuffer) {
  using ReverseReshapeOpTy = typename std::conditional<
      std::is_same<TensorReshapeOpTy, tensor::CollapseShapeOp>::value,
      tensor::ExpandShapeOp, tensor::CollapseShapeOp>::type;
  return b.create<ReverseReshapeOpTy>(reshapeOp.getLoc(),
                                      reshapeOp.getSrcType(), resultBuffer,
                                      reshapeOp.getReassociation());
}

/// Gets the reverse of a `tensor.cast` op to get a memref type that
/// can be used for in-place computation of the result of a disaptch region.
static Value getReverseOfCastOp(OpBuilder &b, tensor::CastOp castOp,
                                Value resultBuffer) {
  return b.create<tensor::CastOp>(castOp.getLoc(), castOp.getSource().getType(),
                                  resultBuffer);
}

/// Returns a tied result value give the operand. If no such result exists,
/// returns `nullptr`.
static Value getTiedResultForOperand(OpOperand &operand,
                                     const BufferizationPlan &plan) {
  for (Value result : operand.getOwner()->getResults()) {
    if (plan.isEquivalent(operand.get(), result)) {
      return result;
    }
  }
  if (auto yieldOp = dyn_cast<scf::YieldOp>(operand.getOwner())) {
    Operation *parentOp = yieldOp->getParentOp();
    if (isa<scf::ForOp, scf::IfOp>(parentOp)) {
      Value result = parentOp->getResult(operand.getOperandNumber());
      if (plan.isEquivalent(result, operand.get())) {
        return result;
      }
    }
  }
  return nullptr;
}

/// To perform updates directly into the result buffer, the uses need to be
/// walked to get to a value already mapped to a buffer or a
/// `flow.dispatch.tensor.store` operation. For each use, gets the tied result
/// and follow its uses. The traversed uses and thir tied results are returned
/// in `traversedUses`.
static IREE::Flow::DispatchTensorStoreOp walkUseToGetDispatchTensorStoreOp(
    Value value, const BufferizationPlan &plan,
    SmallVectorImpl<OpOperand *> &traversedUses,
    llvm::DenseSet<Value> &processed) {
  Operation *user = nullptr;
  while (value.hasOneUse()) {
    processed.insert(value);
    OpOperand &use = *value.use_begin();
    user = use.getOwner();
    if (auto storeOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(user)) {
      return storeOp;
    }
    value = getTiedResultForOperand(use, plan);
    if (!value) return nullptr;
    traversedUses.push_back(&use);
  }
  // If the value has a use which is a store, then use that directly.
  for (Operation *user : value.getUsers()) {
    if (auto storeOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(user)) {
      return storeOp;
    }
  }
  return nullptr;
}

/// For the operation that produces `resultValue`, replaces the operand whose
/// buffer can be reused for the result with `destinationValue`. This makes the
/// dispatch region use destination passing style.
/// TODO(ravishankarm): This could just use the tied operand information from
/// BufferizationPlan object constructed.
static LogicalResult replaceDestinationBuffer(OpResult resultValue,
                                              Value destinationValue) {
  Operation *op = resultValue.getOwner();
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<DestinationStyleOpInterface>([&](auto op) {
        op.setDpsInitOperand(resultValue.getResultNumber(), destinationValue);
        return success();
      })
      .Case<tensor::EmptyOp>([&](auto emptyOp) {
        emptyOp.replaceAllUsesWith(destinationValue);
        return success();
      })
      .Default([](auto defaultOp) {
        return defaultOp->emitOpError("failed to update destination value");
      });
}

/// For an operation whose `resultValue` is the result of the dispatch region,
/// gets the buffer to use to compute the value in-place.
static LogicalResult modifyResultToUseStoreBuffer(
    OpBuilder &b, OpResult resultValue, const BufferizationPlan &plan,
    llvm::DenseSet<Value> &processed) {
  // Traverse the use-def chains to get the `flow.dispatch.tensor.store`
  // operation keeping track of all the traversed operations. Note that the
  // equivalence set construction should ensure that all operations traversed
  // here have a single use.
  Operation *resultValueOp = resultValue.getOwner();
  SmallVector<OpOperand *> traversedUses;
  IREE::Flow::DispatchTensorStoreOp storeOp = walkUseToGetDispatchTensorStoreOp(
      resultValue, plan, traversedUses, processed);
  if (!storeOp) {
    return resultValueOp->emitOpError(
        "failed walk of uses to get to flow.dispatch.tensor.store op");
  }

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(storeOp->getBlock());
  if (auto sourceDefiningOp = storeOp.getTarget().getDefiningOp()) {
    if (sourceDefiningOp->getBlock() == storeOp->getBlock()) {
      b.setInsertionPointAfter(sourceDefiningOp);
    }
  }
  Value resultBuffer = getTensorLoadOpForTensorStoreOp(b, storeOp);

  // Now replay the instructions that are essentially doing type-conversion, in
  // reverse, to get the type needed for the operation computing the value.
  for (auto &it : llvm::reverse(traversedUses)) {
    Operation *op = it->getOwner();
    resultBuffer =
        TypeSwitch<Operation *, Value>(op)
            .Case<DestinationStyleOpInterface>(
                [&](auto) { return resultBuffer; })
            .Case<scf::IfOp, scf::ForOp, tensor::InsertSliceOp,
                  vector::TransferWriteOp>([&](auto) { return resultBuffer; })
            .Case<tensor::InsertSliceOp>([&](auto insertSliceOp) -> Value {
              if (it->get() == insertSliceOp.getDest()) {
                return resultBuffer;
              }
              return nullptr;
            })
            .Case<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
                [&](auto reshapeOp) {
                  return getReverseOfReshapeOp(b, reshapeOp, resultBuffer);
                })
            .Case<tensor::CastOp>([&](tensor::CastOp castOp) {
              return getReverseOfCastOp(b, castOp, resultBuffer);
            })
            .Default([&](Operation *) { return nullptr; });
    if (!resultBuffer) {
      return op->emitOpError(
          "unhandled operation when converting to destination passing style");
    }
  }
  if (failed(replaceDestinationBuffer(resultValue, resultBuffer))) {
    return failure();
  }
  return success();
}

/// Main entry point to convert dispatch region to use destination passing
/// style.
static LogicalResult convertToDestinationPassingStyle(OpBuilder &b,
                                                      func::FuncOp funcOp) {
  BufferizationPlan plan;
  if (failed(createTensorEquivalenceClasses(funcOp, plan))) {
    return failure();
  }

  llvm::DenseSet<Value> processed;
  auto walkResult = funcOp.walk<WalkOrder::PreOrder>(
      [&](tensor::EmptyOp emptyOp) -> WalkResult {
        for (auto result : emptyOp->getResults()) {
          if (!llvm::isa<RankedTensorType>(result.getType())) continue;
          if (plan.isInStoreSet(result) && !processed.count(result)) {
            return modifyResultToUseStoreBuffer(b, result, plan, processed);
          }
        }
        return success();
      });
  return success(!walkResult.wasInterrupted());
}

/// Multiple uses of `tensor.empty()` results in a copy since upstream
/// treats `tensor.empty()` as an allocation and sees uses as a data-hazard
/// creating copies/allocations. Since the `empty` op is a proxy for
/// undef, these could just be duplicated to have a single use. This removes
/// unnecessary data-hazards.
static LogicalResult duplicateTensorEmptyOps(OpBuilder &b,
                                             tensor::EmptyOp emptyOp) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(emptyOp);
  SmallVector<OpOperand *> uses = llvm::to_vector(
      llvm::map_range(emptyOp->getUses(), [](OpOperand &use) { return &use; }));
  for (auto use : llvm::make_range(std::next(uses.begin()), uses.end())) {
    auto newOp = cast<tensor::EmptyOp>(b.clone(*emptyOp.getOperation()));
    Operation *user = use->getOwner();
    user->setOperand(use->getOperandNumber(), newOp);
  }
  return success();
}

// Checks if the `inOperand` can be used in place of the `initOperand`
// to mimic in-place update behavior for parallel elementwise ops.
static bool canUseInOperandAsInitOperand(
    OpOperand *inOperand, OpOperand *initOperand,
    bool useWARForCooperativeMatrixCodegen = false) {
  if (isReadOnly(inOperand->get())) {
    return false;
  }

  if (inOperand->getOwner() != initOperand->getOwner()) return false;

  auto linalgOp = dyn_cast<linalg::LinalgOp>(inOperand->getOwner());
  if (!linalgOp) return false;

  if (linalgOp.getMatchingIndexingMap(inOperand) !=
      linalgOp.getMatchingIndexingMap(initOperand)) {
    return false;
  }

  if (inOperand->get().getType() != initOperand->get().getType()) return false;

  if (useWARForCooperativeMatrixCodegen) {
    return true;
  }

  if (auto producerOp = inOperand->get().getDefiningOp<linalg::LinalgOp>()) {
    if (succeeded(linalg::vectorizeLinalgOpPrecondition(linalgOp)) &&
        succeeded(linalg::vectorizeLinalgOpPrecondition(producerOp))) {
      return false;
    }
  }
  return true;
}

/// Checks if the use of a result of a compute op can be modified
/// so that it can be moved into a store set.
static std::optional<OpOperand *> canModifyUseToGetValueIntoStoreSet(
    BufferizationPlan &plan, OpOperand *use,
    bool useWARForCooperativeMatrixCodegen) {
  assert(!plan.isInStoreSet(use->get()) &&
         "attempting to move a value into a store set, when it is already part "
         "of one");

  // Currently only look at use in linalg.generic ops.
  auto genericOpConsumer = dyn_cast<linalg::GenericOp>(use->getOwner());
  if (!genericOpConsumer) return std::nullopt;

  // All loops need to be parallel.
  if (genericOpConsumer.getNumLoops() !=
      genericOpConsumer.getNumParallelLoops()) {
    return std::nullopt;
  }

  if (genericOpConsumer.isDpsInit(use)) return std::nullopt;

  for (auto [index, initOperand] :
       llvm::enumerate(genericOpConsumer.getDpsInitOperands())) {
    // Output tensor is unused in the body computation.
    if (genericOpConsumer.payloadUsesValueFromOperand(initOperand)) continue;
    // The result of this operation needs to be in a store set.
    if (!plan.isInStoreSet(genericOpConsumer->getResult(index))) continue;
    if (!canUseInOperandAsInitOperand(use, initOperand,
                                      useWARForCooperativeMatrixCodegen)) {
      continue;
    }
    return initOperand;
  }
  return std::nullopt;
}

/// For a compute op which has a result not in the store set, but has a user
/// with an `inOperand`/`initOperand` pair (`inOperand` being the use of result
/// of the compute op) modify the user to replace `initOperand` with a use of
/// the result. This avoids the need for a temporary stack for result of the
/// `initOperand`.
static LogicalResult modifyUseToGetValueIntoStoreSet(RewriterBase &rewriter,
                                                     OpOperand *inOperand,
                                                     OpOperand *initOperand) {
  /// Currently handle only uses as `ins` of `linalg.generic`. Replace the
  /// `initOperand` with the `inOperand`, and drop its use from the `ins`
  /// operand list.
  auto genericOp = cast<linalg::GenericOp>(inOperand->getOwner());
  assert(genericOp == initOperand->getOwner() &&
         "expected in operand and out operand to be the same op");
  SmallVector<Value> newInputs;
  SmallVector<Value> newOutputs;
  SmallVector<Type> newResultTypes;
  SmallVector<AffineMap> maps;
  for (OpOperand *in : genericOp.getDpsInputOperands()) {
    if (in != inOperand) {
      newInputs.push_back(in->get());
      maps.push_back(genericOp.getMatchingIndexingMap(in));
    }
  }
  for (OpOperand *out : genericOp.getDpsInitOperands()) {
    maps.push_back(genericOp.getMatchingIndexingMap(out));
    if (initOperand == out) {
      newOutputs.push_back(inOperand->get());
      newResultTypes.push_back(inOperand->get().getType());
    } else {
      newOutputs.push_back(out->get());
      newResultTypes.push_back(out->get().getType());
    }
  }
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(genericOp);

  Location loc = genericOp.getLoc();
  SmallVector<utils::IteratorType> iterTypes(genericOp.getNumLoops(),
                                             utils::IteratorType::parallel);
  auto newOp = rewriter.create<linalg::GenericOp>(
      loc, newResultTypes, newInputs, newOutputs, maps, iterTypes,
      /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(genericOp));
  rewriter.inlineRegionBefore(genericOp.getRegion(), newOp.getRegion(),
                              newOp.getRegion().begin());

  // Repair the payload entry block.
  Block &payload = newOp.getRegion().front();
  payload.getArgument(inOperand->getOperandNumber())
      .replaceAllUsesWith(payload.getArgument(initOperand->getOperandNumber()));
  payload.eraseArgument(inOperand->getOperandNumber());

  rewriter.replaceOp(genericOp, newOp.getResults());
  return success();
}

/// For results of compute ops that are not part of a store set, tries to modify
/// its users to get the result into a store set, avoiding the use of stack
/// allocation. This is done by looking for users
/// 1) The result of the user is in the store set.
/// 2) The use of the result of the compute op can be updated such that
///    the new use is tied to the result of the user.
/// This makes the result of the compute op be in the store set, and
/// bufferizable without using a new stack. See
/// https://github.com/openxla/iree/issues/8303.
static LogicalResult adaptComputeConsumerToAvoidStackAllocation(
    func::FuncOp funcOp, bool useWARForCooperativeMatrixCodegen) {
  IRRewriter rewriter(funcOp.getContext());

  constexpr int kMaxNumIterations = 6;
  int numIterations = 0;
  while (numIterations < kMaxNumIterations) {
    numIterations++;
    BufferizationPlan plan;
    if (failed(createTensorEquivalenceClasses(funcOp, plan))) {
      return funcOp.emitOpError("failed to create tensor equivalance classes");
    }

    auto resultMovedIntoStoreSet =
        [&](TilingInterface computeOp) -> WalkResult {
      for (auto result : computeOp->getResults()) {
        // If result is already in a store set. Nothing to do.
        if (plan.isInStoreSet(result)) continue;

        // Check if there are any uses that can be modified to reuse the output
        // buffer.
        for (OpOperand &use : result.getUses()) {
          std::optional<OpOperand *> reusableOperand =
              canModifyUseToGetValueIntoStoreSet(
                  plan, &use, useWARForCooperativeMatrixCodegen);
          if (!reusableOperand) continue;
          if (failed(modifyUseToGetValueIntoStoreSet(rewriter, &use,
                                                     reusableOperand.value())))
            continue;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    };

    // If walk wasnt interrupted, then there wasnt any modifications. So break;
    if (!funcOp.walk(resultMovedIntoStoreSet).wasInterrupted()) {
      break;
    }
  }
  if (numIterations >= kMaxNumIterations) {
    return funcOp.emitOpError(
        "Hit maximum number of iterations to avoid stack allocation");
  }
  return success();
}

/// Replaces a tensor.empty op with bufferization.alloc_tensor op which is
/// created by tiling tensor.unpack op. It is intended because tiling unpack ops
/// with non-perfect sizes needs extra elements. See the tiling implementation
/// of tensor.unpack op for more details.
static LogicalResult replaceUnpackEmptyWithAllocTensor(OpBuilder &b,
                                                       func::FuncOp funcOp) {
  funcOp.walk([&](tensor::UnPackOp unpackOp) {
    if (!unpackOp->hasOneUse() ||
        !isa<tensor::ExtractSliceOp>(*(unpackOp->user_begin()))) {
      return;
    }
    auto emptyOp = unpackOp.getDest().getDefiningOp<tensor::EmptyOp>();
    if (!emptyOp) return;

    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointAfter(emptyOp);
    auto allocTensor = b.create<bufferization::AllocTensorOp>(
        emptyOp.getLoc(), emptyOp.getType(), emptyOp.getDynamicSizes());
    emptyOp.replaceAllUsesWith(allocTensor.getResult());
  });

  return success();
}

namespace {
struct RemoveCstOutsDependency
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    bool modifiedOutput = false;
    Location loc = op.getLoc();
    for (OpOperand *opOperand : op.getDpsInitOperands()) {
      DenseElementsAttr attr;
      if (!matchPattern(opOperand->get(), m_Constant(&attr))) continue;
      if (!attr.isSplat()) continue;
      auto type = llvm::dyn_cast<RankedTensorType>(attr.getType());
      if (!type) continue;
      TypedAttr scalarAttr = attr.getValues<TypedAttr>()[0];

      modifiedOutput = true;
      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, type.getShape(), type.getElementType());
      Value cstOp = rewriter.create<arith::ConstantOp>(loc, scalarAttr);
      Value fillOp =
          rewriter.create<linalg::FillOp>(loc, cstOp, emptyTensor).result();
      op->setOperand(opOperand->getOperandNumber(), fillOp);
    }
    if (!modifiedOutput) {
      rewriter.cancelRootUpdate(op);
      return failure();
    }
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

/// Add a pattern to switch
/// ```mlir
///  %0 = scf.if %cond {
///    ...
///    scf.yield %true
///  } else {
///    ...
///    scf.yield %false
///  }
///  flow.dispatch.tensor.store %0, %target, ...
/// ```
///
/// to
///
/// ```mlir
///  scf.if %cond {
///    ...
///    flow.dispatch.tensor.store %true, %target
///  } else {
///    ...
///    flow.dispatch.tensor.store %true, %target
///  }
/// ```
/// This is a workaround for #11273 while a proper fix lands.
struct SwitchStoreOfIfResultValue
    : public OpRewritePattern<IREE::Flow::DispatchTensorStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Flow::DispatchTensorStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto ifOp = storeOp.getValue().getDefiningOp<scf::IfOp>();
    if (!ifOp) {
      return rewriter.notifyMatchFailure(storeOp,
                                         "store source is not an if statement");
    }

    auto resultNumber =
        llvm::cast<OpResult>(storeOp.getValue()).getResultNumber();
    auto moveStoreInsideBody = [&](Block *body) {
      OpBuilder::InsertionGuard guard(rewriter);
      auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
      rewriter.setInsertionPoint(yieldOp);
      auto yieldedVal = yieldOp.getOperand(resultNumber);
      SliceAndDynamicDims sliceAndDynamicDims =
          cloneOffsetsSizesAndStrides(rewriter, storeOp);
      rewriter.create<IREE::Flow::DispatchTensorStoreOp>(
          storeOp.getLoc(), yieldedVal, storeOp.getTarget(),
          sliceAndDynamicDims.dynamicDims, sliceAndDynamicDims.offsets,
          sliceAndDynamicDims.sizes, sliceAndDynamicDims.strides);
    };

    moveStoreInsideBody(&ifOp.getThenRegion().front());
    moveStoreInsideBody(&ifOp.getElseRegion().front());
    rewriter.eraseOp(storeOp);
    return success();
  }
};

}  // namespace

void ConvertToDestinationPassingStylePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  MLIRContext *context = &getContext();

  OpBuilder b(context);
  SmallVector<tensor::EmptyOp> emptyOps;
  funcOp.walk([&](tensor::EmptyOp emptyOp) { emptyOps.push_back(emptyOp); });
  if (llvm::any_of(emptyOps, [&](tensor::EmptyOp emptyOp) {
        return failed(duplicateTensorEmptyOps(b, emptyOp));
      })) {
    return signalPassFailure();
  }

  if (failed(adaptComputeConsumerToAvoidStackAllocation(
          funcOp, useWARForCooperativeMatrixCodegen))) {
    return signalPassFailure();
  }

  {
    RewritePatternSet patterns(context);
    patterns.insert<RemoveCstOutsDependency>(context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  if (failed(replaceUnpackEmptyWithAllocTensor(b, funcOp))) {
    return signalPassFailure();
  }

  if (failed(convertToDestinationPassingStyle(b, funcOp))) {
    return signalPassFailure();
  }

  // Add patterns to remove unused operands and results
  {
    RewritePatternSet patterns(context);
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  {
    RewritePatternSet patterns(context);
    patterns.insert<SwitchStoreOfIfResultValue>(context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertToDestinationPassingStylePass(
    bool useWARForCooperativeMatrixCodegen) {
  return std::make_unique<ConvertToDestinationPassingStylePass>(
      useWARForCooperativeMatrixCodegen);
}

}  // namespace iree_compiler
}  // namespace mlir
