// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- ConvertToDestinationPassingStylePass.cpp ---------------------------===//
//
// Transformations that are performed before calling upstream Comprehensive
// Bufferization pass. These change the dispatch region to use destination
// passing style, mostly to get rid of `init_tensor` ops that result in an
// allocation.
//
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/BufferizationAnalysis.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
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
    registry.insert<linalg::LinalgDialect>();
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
      storeOp.getLoc(), storeOp.getValue().getType().cast<RankedTensorType>(),
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
      .Case<tensor::EmptyOp>([&](auto emptyTensorOp) {
        emptyTensorOp.replaceAllUsesWith(destinationValue);
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
      [&](tensor::EmptyOp emptyTensorOp) -> WalkResult {
        for (auto result : emptyTensorOp->getResults()) {
          if (!result.getType().isa<RankedTensorType>()) continue;
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
/// creating copies/allocations. Since the `init_tensor` op is a proxy for
/// undef, these could just be duplicated to have a single use. This removes
/// unnecessary data-hazards.
static LogicalResult duplicateInitTensorOps(OpBuilder &b,
                                            tensor::EmptyOp emptyTensorOp) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(emptyTensorOp);
  SmallVector<OpOperand *> uses = llvm::to_vector(llvm::map_range(
      emptyTensorOp->getUses(), [](OpOperand &use) { return &use; }));
  for (auto use : llvm::make_range(std::next(uses.begin()), uses.end())) {
    auto newOp = cast<tensor::EmptyOp>(b.clone(*emptyTensorOp.getOperation()));
    Operation *user = use->getOwner();
    user->setOperand(use->getOperandNumber(), newOp);
  }
  return success();
}

// Checks if the `inOperand` can be used in place of the `outOperand`
// to mimic in-place update behavior for parallel elementwise ops.
static bool canUseInOperandAsOutOperand(
    OpOperand *inOperand, OpOperand *outOperand,
    bool useWARForCooperativeMatrixCodegen = false) {
  if (isReadOnly(inOperand->get())) {
    return false;
  }

  if (inOperand->getOwner() != outOperand->getOwner()) return false;

  auto linalgOp = dyn_cast<linalg::LinalgOp>(inOperand->getOwner());
  if (!linalgOp) return false;

  if (linalgOp.getMatchingIndexingMap(inOperand) !=
      linalgOp.getMatchingIndexingMap(outOperand)) {
    return false;
  }

  if (inOperand->get().getType() != outOperand->get().getType()) return false;

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

namespace {
/// Adapts Linalg ops input operand to output operand. This is required for not
/// creating extra alloca ops. For more details, see
/// https://github.com/iree-org/iree/issues/8303
struct AdaptLinalgInputOperandToOutputOperand
    : public OpRewritePattern<linalg::GenericOp> {
  AdaptLinalgInputOperandToOutputOperand(MLIRContext *context,
                                         bool useWARForCooperativeMatrixCodegen,
                                         PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit),
        useWARForCooperativeMatrixCodegen(useWARForCooperativeMatrixCodegen) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // All the loops should be parallel loops.
    if (op.getNumLoops() != op.getNumParallelLoops()) return failure();
    // There is only one result tensor.
    if (op->getNumResults() != 1) return failure();
    // The output tensor is unused in the body computation.
    auto outputOperand = op.getDpsInitOperand(0);
    if (op.payloadUsesValueFromOperand(outputOperand)) return failure();

    // Find an input operand which meets:
    //   1. It has the same indexing map and type.
    //   2. It is not from a readonly tensor.
    OpOperand *operand = nullptr;
    SmallVector<Value> newOperands;
    SmallVector<AffineMap> maps;
    for (auto in : op.getDpsInputOperands()) {
      if (!operand &&
          canUseInOperandAsOutOperand(in, outputOperand,
                                      useWARForCooperativeMatrixCodegen)) {
        operand = in;
      } else {
        newOperands.push_back(in->get());
        maps.push_back(op.getMatchingIndexingMap(in));
      }
    }
    if (!operand) return failure();
    maps.push_back(op.getMatchingIndexingMap(operand));

    Location loc = op.getLoc();
    SmallVector<utils::IteratorType> iterTypes(op.getNumLoops(),
                                               utils::IteratorType::parallel);
    auto newOp = rewriter.create<linalg::GenericOp>(
        loc, op.getResultTypes(), newOperands, operand->get(), maps, iterTypes,
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(op));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().begin());

    // Repair the payload entry block.
    Block &payload = newOp.getRegion().front();
    payload.getArgument(operand->getOperandNumber())
        .replaceAllUsesWith(payload.getArgument(op.getNumDpsInputs()));
    payload.eraseArgument(operand->getOperandNumber());

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }

 private:
  bool useWARForCooperativeMatrixCodegen;
};

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
      auto type = attr.getType().dyn_cast<RankedTensorType>();
      if (!type) continue;
      Attribute scalarAttr = attr.getValues<Attribute>()[0];

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
}  // namespace

void ConvertToDestinationPassingStylePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  MLIRContext *context = &getContext();

  {
    RewritePatternSet patterns(context);
    patterns.insert<AdaptLinalgInputOperandToOutputOperand>(
        context, useWARForCooperativeMatrixCodegen);
    patterns.insert<RemoveCstOutsDependency>(context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  OpBuilder b(context);
  SmallVector<tensor::EmptyOp> emptyTensorOps;
  funcOp.walk([&](tensor::EmptyOp emptyTensorOp) {
    emptyTensorOps.push_back(emptyTensorOp);
  });
  if (llvm::any_of(emptyTensorOps, [&](tensor::EmptyOp emptyTensorOp) {
        return failed(duplicateInitTensorOps(b, emptyTensorOp));
      })) {
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
}

std::unique_ptr<OperationPass<func::FuncOp>>
createConvertToDestinationPassingStylePass(
    bool useWARForCooperativeMatrixCodegen) {
  return std::make_unique<ConvertToDestinationPassingStylePass>(
      useWARForCooperativeMatrixCodegen);
}

}  // namespace iree_compiler
}  // namespace mlir
