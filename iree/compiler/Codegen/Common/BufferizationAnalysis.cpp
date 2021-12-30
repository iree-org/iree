// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- BufferizationAnalysis.cpp - Pre bufferization analysis -------------===//
//
// Analysis to group together tensors within a dispatch region into an
// equivalance class such that all members of a set can be mapped to the same
// memory region.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Codegen/Common/BufferizationAnalysis.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "iree-codegen-bufferization-analysis"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Analysis to compute equivalence sets.
//
// These functions compute the equivalence relationships between all tensors in
// the program. Two tensors are equivalent if they are to be mapped to the same
// buffer. For every operation, based on the operation semantics the result of
// the operation can reuse the buffer for an operand of the operation. This
// information is captured by adding these two tensors to the same equivalence
// class. Eventually the result of the dispatch tensor is added to some
// equivalence set. All tensors in that equivalence set can reuse the result
// buffer and compute the values in place. You can add tensors to equivalence
// set only if
// - They have a single use
// - They are derived from a read-only buffer.
//
//===----------------------------------------------------------------------===//

/// Check if all users of an op that lowers to a subview eventually can use the
/// subview when converted to buffers. For example `linalg.reshape` (which is
/// the buffer version of `linalg.tensor_reshape`) cannot handle subviews.
static bool canUsersHandleSubviews(Operation *op) {
  // TODO(ravishankarm): Maybe this is too aggressive, might have to switch this
  // to have a white-list instead of blacklist.
  for (Operation *user : op->getUsers()) {
    if (isa<IREE::Flow::DispatchTensorStoreOp, tensor::CollapseShapeOp,
            tensor::ExpandShapeOp>(user)) {
      return false;
    }
  }
  return true;
}

/// Walks the use-def chain and see if this value comes from a read-only tensor.
static bool isFromReadOnlyTensor(Value v, const BufferizationPlan &plan) {
  auto definingOp = v.getDefiningOp();
  if (!definingOp) {
    auto arg = v.cast<BlockArgument>();
    return TypeSwitch<Operation *, bool>(arg.getOwner()->getParentOp())
        .Case<scf::ForOp>([&](scf::ForOp forOp) {
          Value initOperand = forOp.getOpOperandForRegionIterArg(arg).get();
          if (plan.isEquivalent(arg, initOperand)) {
            return isFromReadOnlyTensor(initOperand, plan);
          }
          return false;
        })
        .Default([&](Operation *op) { return false; });
  }
  return TypeSwitch<Operation *, bool>(definingOp)
      .Case<arith::ConstantOp>(
          [&](arith::ConstantOp constantOp) { return true; })
      .Case<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
          [&](auto op) { return isFromReadOnlyTensor(op.src(), plan); })
      .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp sliceOp) {
        return isFromReadOnlyTensor(sliceOp.source(), plan);
      })
      .Case<IREE::Flow::DispatchTensorLoadOp>(
          [&](IREE::Flow::DispatchTensorLoadOp loadOp) {
            return loadOp.source()
                       .getType()
                       .cast<IREE::Flow::DispatchTensorType>()
                       .getAccess() == IREE::Flow::TensorAccess::ReadOnly;
          })
      .Default([&](Operation *op) { return false; });
}

/// Adds the result of `std.constant` to its set (there is nothing to tie to
/// here).
static LogicalResult analyseConstantOp(arith::ConstantOp constantOp,
                                       BufferizationPlan &plan) {
  if (!constantOp.getResult().getType().isa<ShapedType>()) return success();
  plan.insert(constantOp.getResult());
  return success();
}

/// Adds the result of the `flow.dispatch.tensor.load` op to the same
/// equivalence class as the source.
static LogicalResult analyseInterfaceLoadTensorOp(
    IREE::Flow::DispatchTensorLoadOp loadOp, BufferizationPlan &plan) {
  plan.unionSets(loadOp.result(), loadOp.source());
  return success();
}

/// Helper method to returns an operation of type `OpType` whose result is in
/// the same equivalence set as `value`. Returns an operation if there is only
/// one such op in the equivalence set or nullptr in all other cases.
template <typename OpType>
static OpType getEquivalentOpOfType(Value value, BufferizationPlan &plan) {
  OpType equivalentOp;
  SmallVector<Value> mappedTensors = plan.getTensorsMappedToSameSet(value);
  for (auto v : mappedTensors) {
    auto definingOp = v.getDefiningOp<OpType>();
    if (!definingOp) continue;
    assert((!equivalentOp || equivalentOp == definingOp) &&
           "found two interface binding ops marked as equivalent");
    if (!equivalentOp) equivalentOp = definingOp;
  }
  return equivalentOp;
}

/// Returns true if the value and target of a `flow.dispatch.tensor.store`
/// operation can be added to the same equivalence set. This can be done only if
/// - The `value` is not from a equivalence set that contains a read-only
///   tensor.
/// - All `hal.interface.binding.subspan` operations in the equivalence class of
///   `value` and `target` have the same binding and offset. For now, it is
///   assumed that the equivalence classes contain only 1 such instruction.
/// This method asserts that the `target` equivalence class already contains a
/// `hal.interface.binding.subspan` op.'
static bool canSetStoreValueAndTargetAsEquivalent(
    IREE::Flow::DispatchTensorStoreOp storeOp, BufferizationPlan &plan) {
  Value value = storeOp.value();
  Value target = storeOp.target();
  auto targetInterfaceOp =
      getEquivalentOpOfType<IREE::HAL::InterfaceBindingSubspanOp>(target, plan);
  assert(targetInterfaceOp);
  if (auto valueConstantOp =
          getEquivalentOpOfType<arith::ConstantOp>(value, plan)) {
    return false;
  }
  if (auto valueInterfaceOp =
          getEquivalentOpOfType<IREE::HAL::InterfaceBindingSubspanOp>(value,
                                                                      plan)) {
    if (targetInterfaceOp.binding() != valueInterfaceOp.binding() ||
        targetInterfaceOp.byte_offset() != valueInterfaceOp.byte_offset()) {
      // If the binding and offsets are different, map these to different
      // memrefs.
      return false;
    }
    // If the binding and offsets are the same, make sure that the
    // !flow.dispatch.tensor is read-write.
    auto sourceType =
        valueInterfaceOp.getType().dyn_cast<IREE::Flow::DispatchTensorType>();
    return sourceType &&
           sourceType.getAccess() == IREE::Flow::TensorAccess::ReadWrite;
  }
  return true;
}

/// Tries to add the `value` and `target` to the same equivalence class.
static LogicalResult analyseInterfaceStoreTensorOp(
    IREE::Flow::DispatchTensorStoreOp storeOp, BufferizationPlan &plan) {
  // The value and target can be union-ed if the set the value is part of does
  // not contain any hal.interface.binding.subspan from a different binding.
  Value value = storeOp.value();
  Value target = storeOp.target();
  if (!getEquivalentOpOfType<IREE::HAL::InterfaceBindingSubspanOp>(target,
                                                                   plan)) {
    return storeOp.emitError(
        "expected target of store op to already be added to an equivalence "
        "set");
  }
  if (canSetStoreValueAndTargetAsEquivalent(storeOp, plan)) {
    plan.unionSets(value, target);
  } else {
    plan.insert(value);
  }
  plan.storeSet(target);
  return success();
}

static LogicalResult analyseInterfaceBindingSubspanOp(
    IREE::HAL::InterfaceBindingSubspanOp subspanOp, BufferizationPlan &plan) {
  plan.insert(subspanOp.getResult());
  return success();
}

static LogicalResult analysePadTensorOp(linalg::PadTensorOp padTensorOp,
                                        BufferizationPlan &plan) {
  plan.insert(padTensorOp.source());
  plan.insert(padTensorOp.result());
  return success();
}

/// For every result of the LinalgOp, gets the operands (`ins` or `outs`) whose
/// buffer can be reused for the result.
static SmallVector<Value> getTiedOperandsForLinalgOps(
    linalg::LinalgOp linalgOp, const BufferizationPlan &plan) {
  SmallVector<Value> tiedOperands(linalgOp.getOperation()->getNumResults());
  auto outputOperands = linalgOp.getOutputOperands();
  for (auto outTensor : llvm::enumerate(outputOperands)) {
    // If the `outs` tensor has a single use (this op) and is not from a
    // read-only buffer, the `outs` tensor can be tied to the result.
    if (outTensor.value()->get().hasOneUse() &&
        !isFromReadOnlyTensor(outTensor.value()->get(), plan)) {
      tiedOperands[outTensor.index()] = outTensor.value()->get();
    }
  }
  return tiedOperands;
}

static LogicalResult analyseLinalgExtOps(IREE::LinalgExt::LinalgExtOp op,
                                         BufferizationPlan &plan) {
  if (!op.hasTensorSemantics()) return success();
  // TODO(hanchung): Revisit if we can tie together op.getOutputOperands() with
  // the corresponding op.getInputOperands(). For now we have limit LinalgExt
  // ops, and there is no use case. So we ignore it.
  // Note: this is what should be done for LinalgOps, except for a what is done
  // for operand fusion today.
  for (auto input : op.getInputOperands()) {
    plan.insert(input->get());
  }
  for (auto output : op.getOutputOperands()) {
    plan.insert(output->get());
  }
  for (auto result : op->getResults()) {
    plan.insert(result);
  }
  return success();
}

/// Adds the corresponding `outs` and result tensors of the linalg op into the
/// same equivalence class.
static LogicalResult analyseLinalgOps(linalg::LinalgOp linalgOp,
                                      BufferizationPlan &plan) {
  if (!linalgOp.hasTensorSemantics()) return success();
  auto results = linalgOp->getResults();
  auto tiedOperands = getTiedOperandsForLinalgOps(linalgOp, plan);
  for (auto it : llvm::enumerate(llvm::zip(results, tiedOperands))) {
    Value resultTensor = std::get<0>(it.value());
    Value tiedOperand = std::get<1>(it.value());
    if (tiedOperand) {
      plan.unionSets(resultTensor, tiedOperand);
    }
    plan.insert(linalgOp.getOutputOperand(it.index())->get());
    plan.insert(resultTensor);
  }
  return success();
}

/// Returns true if there is a single use of the `value` that is "real",
/// i.e. where the value itself is used, and not the type of the value. For
/// example, a use in a `memref.dim` is only looking at the type and not the
/// value.
static bool hasSingleRealUse(Value value) {
  int numUsers = 0;
  for (OpOperand &use : value.getUses()) {
    if (!isa<memref::DimOp, tensor::DimOp>(use.getOwner())) {
      numUsers++;
    }
  }
  return numUsers == 1;
}

/// For operations that have a single operand and result, adds both to the same
/// equivalence class.
static LogicalResult analyseSingleOperandResultOp(Value source, Value result,
                                                  BufferizationPlan &plan) {
  if (hasSingleRealUse(source) || isFromReadOnlyTensor(source, plan)) {
    plan.unionSets(source, result);
    return success();
  }
  plan.insert(source);
  plan.insert(result);
  return success();
}

static LogicalResult analyseSubTensorOp(tensor::ExtractSliceOp subTensorOp,
                                        BufferizationPlan &plan) {
  if (!canUsersHandleSubviews(subTensorOp)) {
    plan.insert(subTensorOp.source());
    plan.insert(subTensorOp.result());
    return success();
  }
  return analyseSingleOperandResultOp(subTensorOp.source(),
                                      subTensorOp.result(), plan);
}

/// Adds the `dest` and `result` tensor of a subtensor insert operation into the
/// same equivalence class. If `source` is not null also checks that the
/// `source` and `dest` are not equivalent.
static LogicalResult analyseDestructiveUpdateOp(Operation *op, Value source,
                                                Value dest, Value result,
                                                BufferizationPlan &plan) {
  if (hasSingleRealUse(dest) && !isFromReadOnlyTensor(dest, plan)) {
    plan.unionSets(dest, result);
  } else if (source && plan.isEquivalent(source, dest)) {
    // The destructive update pattern can put the source and dest in the same
    // equivalence class, but that is checked explicitly later on. So at this
    // stage this shouldnt happen.
    return op->emitError(
        "unexpected source and dest being equivalent in destructive update op");
  }
  plan.insert(dest);
  plan.insert(result);
  return success();
}

static LogicalResult analyseScfIfOp(scf::IfOp ifOp, BufferizationPlan &plan) {
  if (!ifOp.getNumResults()) return success();
  for (auto it : llvm::zip(ifOp.getResults(), ifOp.thenYield().getOperands(),
                           ifOp.elseYield().getOperands())) {
    Value result = std::get<0>(it);
    if (!result.getType().isa<RankedTensorType>()) continue;
    // All results and yields of the if-then-else are tied together.
    plan.unionSets(result, std::get<1>(it));
    plan.unionSets(result, std::get<2>(it));
  }
  return success();
}

static LogicalResult analyseScfForOp(scf::ForOp forOp,
                                     BufferizationPlan &plan) {
  if (forOp.results().empty()) return success();
  if (!llvm::all_of(forOp->getResultTypes(), [](Type resultType) {
        return resultType.isa<RankedTensorType>();
      })) {
    return success();
  }

  auto yeildOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  auto regionArgs = forOp.getRegionIterArgs();
  auto initArgs = forOp.initArgs();
  for (int i = 0; i < yeildOp.results().size(); ++i) {
    Value yieldTensor = yeildOp.results()[i];
    Value resultTensor = forOp.results()[i];
    Value initArg = initArgs[i];
    Value arg = regionArgs[i];
    // Always tie the yield, the result tensor, and the region arg
    plan.unionSets(yieldTensor, resultTensor);
    plan.unionSets(yieldTensor, arg);

    // If the init value is not read-only and has single use, the tie the init
    // and result (and by extension all 4 tensors here).
    if (hasSingleRealUse(initArg) && !isFromReadOnlyTensor(initArg, plan)) {
      plan.unionSets(initArg, resultTensor);
    }
  }
  return success();
}

/// Look for destructive update loop pattern.
///
/// ```mlir
///   %result = scf.for %arg0 = ... iter_args(%arg1 = %init) {
///     %st = subtensor %arg1[...]
///
///     %yieldVal = tensor.insert_slice %val, %arg1[...]
///     scf.yield %yieldVal
///   }
///
/// `%result`, `%arg1` and `%yieldVal` are all already in the same equivalence
/// class. `%st` and `%arg` can be added to the same equivalence class even
/// though `%arg1` has multiple uses. Same is true for `%yieldVal` and
/// `%arg1`. Here we also verify there are no other "value" uses of
/// `%arg1`. This might be overly constraining, but we can relax gradually.
static LogicalResult hasDestructiveUpdateLoopPattern(scf::ForOp forOp,
                                                     BufferizationPlan &plan) {
  for (BlockArgument arg : forOp.getRegionIterArgs()) {
    auto isDestructiveUpdateUses = [&](OpOperand &use) -> bool {
      Operation *user = use.getOwner();
      return TypeSwitch<Operation *, bool>(user)
          .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp sliceOp) {
            return sliceOp.source() == arg;
          })
          .Case<tensor::InsertSliceOp>(
              [&](tensor::InsertSliceOp subTensorInsertOp) {
                return subTensorInsertOp.dest() == arg;
              })
          .Case<memref::DimOp, scf::YieldOp, tensor::DimOp>(
              [&](auto op) { return true; })
          .Default([&](Operation *op) { return false; });
    };
    if (llvm::all_of(arg.getUses(), isDestructiveUpdateUses)) {
      for (Operation *user : arg.getUsers()) {
        TypeSwitch<Operation *>(user)
            .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp sliceOp) {
              plan.unionSets(sliceOp.source(), sliceOp.result());
            })
            .Case<tensor::InsertSliceOp>(
                [&](tensor::InsertSliceOp subTensorInsertOp) {
                  if (!isFromReadOnlyTensor(subTensorInsertOp.source(), plan)) {
                    plan.unionSets(subTensorInsertOp.source(),
                                   subTensorInsertOp.dest());
                  }
                })
            .Default([&](Operation *) {});
      }
    }
  }
  return success();
}

/// Ties together operands for operand fusion as exists today by reusing buffer
/// for the result for one of the inputs to do in-place update. Ideally we dont
/// need to do this if the fusion just happens at vector level. To be removed
/// when that is worked out and can be load-bearing. Conditions checked here are
/// 1) the result does not use the value of the `outs` buffer.
/// 2) the input has a single use (this op) and has the same indexing map as the
///    result.
/// 3) the input equivalence set does not have an interface binding, i.e. it is
///    not using a buffer from the dispatch ABI.
static void tieOperandsForOperandFusion(linalg::LinalgOp linalgOp,
                                        BufferizationPlan &plan) {
  for (auto result : enumerate(linalgOp.getOutputOperands())) {
    if (linalgOp.payloadUsesValueFromOperand(result.value())) {
      continue;
    }
    for (OpOperand *input : linalgOp.getInputTensorOperands()) {
      Type inputElementType =
          input->get().getType().cast<RankedTensorType>().getElementType();
      Type resultElementType = result.value()
                                   ->get()
                                   .getType()
                                   .cast<RankedTensorType>()
                                   .getElementType();
      if (input->get().hasOneUse() && (inputElementType == resultElementType) &&
          linalgOp.getTiedIndexingMap(input) ==
              linalgOp.getTiedIndexingMap(result.value()) &&
          !getEquivalentOpOfType<IREE::HAL::InterfaceBindingSubspanOp>(
              input->get(), plan) &&
          !isFromReadOnlyTensor(input->get(), plan)) {
        plan.unionSets(linalgOp->getResult(result.index()), input->get());
        break;
      }
    }
  }
}

void BufferizationPlan::dump() {
  llvm::dbgs() << "BufferMappings : \n";
  unsigned numSets = 0;
  for (auto it = mappedTensors.begin(), ie = mappedTensors.end(); it != ie;
       ++it) {
    if (!it->isLeader()) continue;
    llvm::dbgs() << "\tSet " << numSets << ":\n";
    for (auto member : llvm::make_range(mappedTensors.member_begin(it),
                                        mappedTensors.member_end())) {
      llvm::dbgs() << "\t\t";
      getValue(member).print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
    numSets++;
  }
}

LogicalResult createTensorEquivalenceClasses(FuncOp funcOp,
                                             BufferizationPlan &plan) {
  auto bufferMappingFn = [&](Operation *op) -> WalkResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<arith::ConstantOp>([&](arith::ConstantOp constantOp) {
          return analyseConstantOp(constantOp, plan);
        })
        .Case<IREE::Flow::DispatchTensorLoadOp>(
            [&](IREE::Flow::DispatchTensorLoadOp loadOp) {
              return analyseInterfaceLoadTensorOp(loadOp, plan);
            })
        .Case<IREE::Flow::DispatchTensorStoreOp>(
            [&](IREE::Flow::DispatchTensorStoreOp storeOp) {
              return analyseInterfaceStoreTensorOp(storeOp, plan);
            })
        .Case<IREE::HAL::InterfaceBindingSubspanOp>(
            [&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
              return analyseInterfaceBindingSubspanOp(subspanOp, plan);
            })
        .Case<linalg::PadTensorOp>([&](linalg::PadTensorOp padTensorOp) {
          return analysePadTensorOp(padTensorOp, plan);
        })
        .Case<linalg::LinalgOp>([&](linalg::LinalgOp linalgOp) {
          return analyseLinalgOps(linalgOp, plan);
        })
        .Case<IREE::LinalgExt::LinalgExtOp>(
            [&](IREE::LinalgExt::LinalgExtOp linalgExtOp) {
              return analyseLinalgExtOps(linalgExtOp, plan);
            })
        .Case<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
            [&](auto reshapeOp) {
              return analyseSingleOperandResultOp(reshapeOp.src(),
                                                  reshapeOp.result(), plan);
            })
        .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp sliceOp) {
          return analyseSubTensorOp(sliceOp, plan);
        })
        .Case<tensor::InsertSliceOp>(
            [&](tensor::InsertSliceOp subTensorInsertOp) {
              return analyseDestructiveUpdateOp(
                  subTensorInsertOp, subTensorInsertOp.source(),
                  subTensorInsertOp.dest(), subTensorInsertOp.result(), plan);
            })
        .Case<tensor::CastOp>([&](tensor::CastOp castOp) {
          return analyseSingleOperandResultOp(castOp.source(), castOp.dest(),
                                              plan);
        })
        .Case<tensor::InsertOp>([&](tensor::InsertOp insertOp) {
          return analyseDestructiveUpdateOp(insertOp, /*source =*/nullptr,
                                            insertOp.dest(), insertOp.result(),
                                            plan);
        })
        .Case<vector::TransferReadOp>(
            [&](vector::TransferReadOp transferReadOp) {
              plan.insert(transferReadOp.source());
              return success();
            })
        .Case<vector::TransferWriteOp>(
            [&](vector::TransferWriteOp transferWriteOp) {
              return analyseDestructiveUpdateOp(transferWriteOp, nullptr,
                                                transferWriteOp.source(),
                                                transferWriteOp.result(), plan);
            })
        .Case<scf::IfOp>(
            [&](scf::IfOp ifOp) { return analyseScfIfOp(ifOp, plan); })
        .Case<scf::ForOp>(
            [&](scf::ForOp forOp) { return analyseScfForOp(forOp, plan); })
        .Default([&](Operation *op) { return success(); });
  };
  if (funcOp.walk<WalkOrder::PreOrder>(bufferMappingFn).wasInterrupted()) {
    return failure();
  }
  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs() << "After First walk ";
    plan.dump();
  });

  if (funcOp
          .walk([&](scf::ForOp forOp) -> WalkResult {
            return hasDestructiveUpdateLoopPattern(forOp, plan);
          })
          .wasInterrupted()) {
    return failure();
  }
  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs() << "After Destructive update walk ";
    plan.dump();
  });

  // Tie operands to allow for operand fusion support. To be dropped once the
  // operand fusion is generalized in IREE.
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    return tieOperandsForOperandFusion(linalgOp, plan);
  });
  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs() << "After union for supporting operand fusion";
    plan.dump();
  });

  if (funcOp
          .walk([&](IREE::Flow::DispatchTensorStoreOp storeOp) -> WalkResult {
            return analyseInterfaceStoreTensorOp(storeOp, plan);
          })
          .wasInterrupted()) {
    return failure();
  }

  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
