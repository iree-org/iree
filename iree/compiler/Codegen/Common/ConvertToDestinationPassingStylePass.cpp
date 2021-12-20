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
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {
class ConvertToDestinationPassingStylePass
    : public ConvertToDestinationPassingStyleBase<
          ConvertToDestinationPassingStylePass> {
 public:
  ConvertToDestinationPassingStylePass() = default;
  void runOnOperation() override;
};
}  // namespace

/// Returns the subview into the buffer that is supposed to be populated with
/// the `value` of the `flow.dispatch.tensor.store` operation. This can be used
/// to compute the results in place.
static Value getTensorLoadOpForTensorStoreOp(
    OpBuilder &b, IREE::Flow::DispatchTensorStoreOp storeOp) {
  // Clone the offset, size and stride values. They will be CSE-ed later.
  Operation *parentOp = storeOp->getParentOp();
  BlockAndValueMapping indexValMap;
  llvm::SetVector<Operation *> slice;
  auto cloneIndexValues = [&](ArrayRef<OpFoldResult> ofrs) {
    SmallVector<OpFoldResult> clonedVals;
    for (auto ofr : ofrs) {
      // Just copy the attributes.
      if (auto attr = ofr.dyn_cast<Attribute>()) {
        clonedVals.push_back(attr);
        continue;
      }
      Value val = ofr.get<Value>();
      // If it is a block argument use the same value.
      if (val.isa<BlockArgument>()) {
        clonedVals.push_back(val);
        continue;
      }
      // The slice of ops needed for index computation need to be cloned to
      // avoid use-def violations. If the value has been cloned already, reuse
      // that.
      if (auto lookupVal = indexValMap.lookupOrNull(val)) {
        clonedVals.push_back(lookupVal);
        continue;
      }
      slice.clear();
      getBackwardSlice(val, &slice, [&](Operation *sliceOp) {
        return sliceOp->getParentOp() == parentOp;
      });
      for (auto sliceOp : slice) {
        if (!indexValMap.contains(sliceOp->getResult(0))) {
          b.clone(*sliceOp, indexValMap);
        }
      }
      if (Operation *definingOp = val.getDefiningOp()) {
        b.clone(*definingOp, indexValMap);
      }
      clonedVals.push_back(indexValMap.lookup(val));
    }
    return clonedVals;
  };
  SmallVector<OpFoldResult> loadOffsets, loadSizes, loadStrides;
  loadOffsets = cloneIndexValues(storeOp.getMixedOffsets());
  loadSizes = cloneIndexValues(storeOp.getMixedSizes());
  loadStrides = cloneIndexValues(storeOp.getMixedStrides());
  Value tensorLoadOp = b.create<IREE::Flow::DispatchTensorLoadOp>(
      storeOp.getLoc(), storeOp.value().getType().cast<RankedTensorType>(),
      storeOp.target(), storeOp.target_dims(), loadOffsets, loadSizes,
      loadStrides);
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
                                      reshapeOp.reassociation());
}

/// Gets the reverse of a `tensor.cast` op to get a memref type that
/// can be used for in-place computation of the result of a disaptch region.
static Value getReverseOfCastOp(OpBuilder &b, tensor::CastOp castOp,
                                Value resultBuffer) {
  return b.create<tensor::CastOp>(castOp.getLoc(), castOp.source().getType(),
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
      .Case<linalg::LinalgOp>([&](auto linalgOp) {
        unsigned resultNumber = resultValue.getResultNumber();
        linalgOp.setOutputOperand(resultNumber, destinationValue);
        return success();
      })
      .Case<linalg::InitTensorOp>([&](auto initTensorOp) {
        initTensorOp.replaceAllUsesWith(destinationValue);
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
  if (auto sourceDefiningOp = storeOp.target().getDefiningOp()) {
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
            .Case<scf::IfOp, scf::ForOp, linalg::LinalgOp,
                  IREE::LinalgExt::LinalgExtOp, tensor::InsertSliceOp,
                  vector::TransferWriteOp>(
                [&](auto caseOp) { return resultBuffer; })
            .Case<tensor::InsertSliceOp>([&](auto insertSliceOp) -> Value {
              if (it->get() == insertSliceOp.dest()) {
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
                                                      FuncOp funcOp) {
  BufferizationPlan plan;
  if (failed(createTensorEquivalenceClasses(funcOp, plan))) {
    return failure();
  }

  llvm::DenseSet<Value> processed;
  auto walkResult = funcOp.walk<WalkOrder::PreOrder>(
      [&](linalg::InitTensorOp initTensorOp) -> WalkResult {
        for (auto result : initTensorOp->getResults()) {
          if (!result.getType().isa<RankedTensorType>()) continue;
          if (plan.isInStoreSet(result) && !processed.count(result)) {
            return modifyResultToUseStoreBuffer(b, result, plan, processed);
          }
        }
        return success();
      });
  return success(!walkResult.wasInterrupted());
}

/// Multiple uses of `linalg.init_tensor` results in a copy since upstream
/// treats `linalg.init_tensor` as an allocation and sees uses as a data-hazard
/// creating copies/allocations. Since the `init_tensor` op is a proxy for
/// undef, these could just be duplicated to have a single use. This removes
/// unnecessary data-hazards.
static LogicalResult duplicateInitTensorOps(OpBuilder &b,
                                            linalg::InitTensorOp initTensorOp) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(initTensorOp);
  for (auto &use : llvm::make_range(std::next(initTensorOp->use_begin()),
                                    initTensorOp->use_end())) {
    auto newOp =
        cast<linalg::InitTensorOp>(b.clone(*initTensorOp.getOperation()));
    Operation *user = use.getOwner();
    user->setOperand(use.getOperandNumber(), newOp);
  }
  return success();
}

void ConvertToDestinationPassingStylePass::runOnOperation() {
  FuncOp funcOp = getOperation();
  MLIRContext *context = &getContext();

  OpBuilder b(context);
  SmallVector<linalg::InitTensorOp> initTensorOps;
  funcOp.walk([&](linalg::InitTensorOp initTensorOp) {
    initTensorOps.push_back(initTensorOp);
  });
  if (llvm::any_of(initTensorOps, [&](linalg::InitTensorOp initTensorOp) {
        return failed(duplicateInitTensorOps(b, initTensorOp));
      })) {
    return signalPassFailure();
  }

  if (failed(convertToDestinationPassingStyle(b, funcOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<FuncOp>>
createConvertToDestinationPassingStylePass() {
  return std::make_unique<ConvertToDestinationPassingStylePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
