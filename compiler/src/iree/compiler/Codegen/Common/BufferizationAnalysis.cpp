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
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "iree-codegen-bufferization-analysis"

namespace mlir::iree_compiler {

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
  Operation *definingOp = v.getDefiningOp();
  if (!definingOp) {
    auto arg = llvm::cast<BlockArgument>(v);
    return TypeSwitch<Operation *, bool>(arg.getOwner()->getParentOp())
        .Case<scf::ForOp>([&](scf::ForOp forOp) {
          Value initOperand = forOp.getTiedLoopInit(arg)->get();
          if (plan.isEquivalent(arg, initOperand)) {
            return isFromReadOnlyTensor(initOperand, plan);
          }
          return false;
        })
        .Default([&](Operation *op) { return false; });
  }
  return isReadOnly(v);
}

/// Adds the result of `std.constant` to its set (there is nothing to tie to
/// here).
static LogicalResult analyseConstantOp(arith::ConstantOp constantOp,
                                       BufferizationPlan &plan) {
  if (!llvm::isa<ShapedType>(constantOp.getResult().getType()))
    return success();
  plan.insert(constantOp.getResult());
  return success();
}

/// Adds the result of the `flow.dispatch.tensor.load` op to the same
/// equivalence class as the source.
static LogicalResult
analyseInterfaceLoadTensorOp(IREE::Flow::DispatchTensorLoadOp loadOp,
                             BufferizationPlan &plan) {
  plan.unionSets(loadOp.getResult(), loadOp.getSource());
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
    if (!definingOp)
      continue;
    assert((!equivalentOp || equivalentOp == definingOp) &&
           "found two interface binding ops marked as equivalent");
    if (!equivalentOp)
      equivalentOp = definingOp;
  }
  return equivalentOp;
}

/// Check if two sets can be merged based on what operations exist in that set.
static bool canSetsBeMerged(Value v1, Value v2, BufferizationPlan &plan) {
  // Dont merge two sets if one of the sets is a constant.
  if (getEquivalentOpOfType<arith::ConstantOp>(v1, plan) ||
      getEquivalentOpOfType<arith::ConstantOp>(v2, plan)) {
    return false;
  }
  auto v1InterfaceBinding =
      getEquivalentOpOfType<IREE::HAL::InterfaceBindingSubspanOp>(v1, plan);
  auto v2InterfaceBinding =
      getEquivalentOpOfType<IREE::HAL::InterfaceBindingSubspanOp>(v2, plan);
  // If any of these sets do not have a interface binding, they can be merged.
  if (!v1InterfaceBinding || !v2InterfaceBinding) {
    return true;
  }
  if (v1InterfaceBinding.getSet() != v2InterfaceBinding.getSet() ||
      v1InterfaceBinding.getBinding() != v2InterfaceBinding.getBinding() ||
      v1InterfaceBinding.getByteOffset() !=
          v2InterfaceBinding.getByteOffset()) {
    // If the set, binding or offsets are different, map these to different
    // memrefs.
    return false;
  }
  return true;
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
static bool
canSetStoreValueAndTargetAsEquivalent(IREE::Flow::DispatchTensorStoreOp storeOp,
                                      BufferizationPlan &plan) {
  if (!canSetsBeMerged(storeOp.getValue(), storeOp.getTarget(), plan)) {
    return false;
  }
  auto valueInterfaceBinding =
      getEquivalentOpOfType<IREE::HAL::InterfaceBindingSubspanOp>(
          storeOp.getValue(), plan);
  auto targetInterfaceBinding =
      getEquivalentOpOfType<IREE::HAL::InterfaceBindingSubspanOp>(
          storeOp.getTarget(), plan);
  if (!valueInterfaceBinding || !targetInterfaceBinding) {
    return true;
  }
  // If the binding and offsets are the same, make sure that the
  // !flow.dispatch.tensor is read-write.
  auto sourceType = llvm::dyn_cast<IREE::Flow::DispatchTensorType>(
      valueInterfaceBinding.getType());
  return sourceType &&
         sourceType.getAccess() == IREE::Flow::TensorAccess::ReadWrite;
}

/// Tries to add the `value` and `target` to the same equivalence class.
static LogicalResult
analyseInterfaceStoreTensorOp(IREE::Flow::DispatchTensorStoreOp storeOp,
                              BufferizationPlan &plan) {
  // The value and target can be union-ed if the set the value is part of does
  // not contain any hal.interface.binding.subspan from a different binding.
  Value value = storeOp.getValue();
  Value target = storeOp.getTarget();
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

static LogicalResult
analyseInterfaceBindingSubspanOp(IREE::HAL::InterfaceBindingSubspanOp subspanOp,
                                 BufferizationPlan &plan) {
  plan.insert(subspanOp.getResult());
  return success();
}

static LogicalResult analysePadTensorOp(tensor::PadOp padTensorOp,
                                        BufferizationPlan &plan) {
  plan.insert(padTensorOp.getSource());
  plan.insert(padTensorOp.getResult());
  return success();
}

/// For every result of the LinalgOp, gets the operands (`ins` or `outs`) whose
/// buffer can be reused for the result.
static SmallVector<Value>
getTiedOperandsForDPSOps(DestinationStyleOpInterface dpsOp,
                         const BufferizationPlan &plan) {
  SmallVector<Value> tiedOperands(dpsOp.getOperation()->getNumResults());
  auto outputOperands = dpsOp.getDpsInits();
  for (auto [index, outTensor] : llvm::enumerate(outputOperands)) {
    // If the `outs` tensor has a single use (this op) and is not from a
    // read-only buffer, the `outs` tensor can be tied to the result.
    if (outTensor.hasOneUse() && !isFromReadOnlyTensor(outTensor, plan)) {
      tiedOperands[index] = outTensor;
    }
  }
  return tiedOperands;
}

/// Adds the corresponding `outs` and result tensors of the linalg op into the
/// same equivalence class.
static LogicalResult analyseDPSOps(DestinationStyleOpInterface dpsOp,
                                   BufferizationPlan &plan) {
  if (!dpsOp.hasTensorSemantics())
    return success();
  auto results = dpsOp->getResults();
  auto tiedOperands = getTiedOperandsForDPSOps(dpsOp, plan);
  if (tiedOperands.empty())
    return failure();
  for (auto [index, resultTensor, tiedOperand] : llvm::zip_equal(
           llvm::seq<int64_t>(0, results.size()), results, tiedOperands)) {
    if (tiedOperand) {
      plan.unionSets(resultTensor, tiedOperand);
    }
    plan.insert(dpsOp.getDpsInitOperand(index)->get());
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
    plan.insert(subTensorOp.getSource());
    plan.insert(subTensorOp.getResult());
    return success();
  }
  return analyseSingleOperandResultOp(subTensorOp.getSource(),
                                      subTensorOp.getResult(), plan);
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
  if (!ifOp.getNumResults())
    return success();
  for (auto [result, thenOperand, elseOperand] :
       llvm::zip_equal(ifOp.getResults(), ifOp.thenYield().getOperands(),
                       ifOp.elseYield().getOperands())) {
    if (!llvm::isa<RankedTensorType>(result.getType()))
      continue;
    // All results and yields of the if-then-else are tied together.
    plan.unionSets(result, thenOperand);
    plan.unionSets(result, elseOperand);
  }
  return success();
}

static LogicalResult analyseScfForOp(scf::ForOp forOp,
                                     BufferizationPlan &plan) {
  if (forOp.getResults().empty())
    return success();
  if (!llvm::all_of(forOp->getResultTypes(), [](Type resultType) {
        return llvm::isa<RankedTensorType>(resultType);
      })) {
    return success();
  }

  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  auto regionArgs = forOp.getRegionIterArgs();
  auto initArgs = forOp.getInitArgs();
  for (int i = 0; i < yieldOp.getResults().size(); ++i) {
    Value yieldTensor = yieldOp.getResults()[i];
    Value resultTensor = forOp.getResults()[i];
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

/// Look for destructive update loop pattern involving `source` using these
/// constraints
/// - single tensor.insert_slice operation where `source` is the `dest` operand.
/// - all `tensor.extract_slice` operations dominate the `tensor.insert_slice`
/// op.
static void hasDestructiveUpdatePattern(Value source, BufferizationPlan &plan) {
  auto isUpdateOp = [](Operation *op) {
    return isa<tensor::InsertSliceOp, vector::TransferWriteOp>(op);
  };
  auto isReadOp = [](Operation *op) {
    return isa<tensor::ExtractSliceOp, vector::TransferReadOp>(op);
  };
  auto getDest = [](Operation *op) -> Value {
    if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op)) {
      return insertSliceOp.getDest();
    }
    if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op)) {
      return transferWriteOp.getSource();
    }
    return nullptr;
  };
  auto getSource = [](Operation *op) -> Value {
    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
      return extractSliceOp.getSource();
    }
    if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op)) {
      return transferReadOp.getSource();
    }
    return nullptr;
  };
  // Source should have only one use that is a tensor::InsertSliceOp as a `dest`
  // operand.
  Operation *updateOp = nullptr;
  for (OpOperand &use : source.getUses()) {
    auto user = use.getOwner();
    // Process only update ops uses here.
    if (!isUpdateOp(user))
      continue;
    // If this is not the first use in a tensor::InsertSliceOp abort.
    if (updateOp) {
      return;
    }
    // If the use is not the `dest` operand, abort.
    Value dest = getDest(user);
    assert(dest && "unable to get dest of update op");
    if (use.get() != dest) {
      return;
    }
    if (isFromReadOnlyTensor(dest, plan)) {
      return;
    }
    updateOp = user;
  }
  // Need to have one use of tensor::InsertSliceOp for destructive update
  // pattern.
  if (!updateOp) {
    return;
  }

  Block *updateOpBlock = updateOp->getBlock();
  for (OpOperand &use : source.getUses()) {
    Operation *user = use.getOwner();
    if (user == updateOp)
      continue;
    if (isReadOp(user)) {
      Value source = getSource(user);
      assert(source && "unable to find source from read op");
      if (source != use.get()) {
        return;
      }
      // The read must dominate the insert op. For now just check its in the
      // same block and before it.
      if (user->getBlock() != updateOpBlock ||
          !user->isBeforeInBlock(updateOp)) {
        return;
      }
      continue;
    } else if (isa<scf::YieldOp, tensor::DimOp>(user)) {
      continue;
    }
    // Unaccounted for use. Return without doing anything;
    return;
  }

  // Found destructive update pattern. Tie all the
  // - extract_slice source and result
  // - insert_slice value and dest
  for (Operation *user : source.getUsers()) {
    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
      plan.unionSets(extractSliceOp.getSource(), extractSliceOp.getResult());
      continue;
    }
    if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(user)) {
      if (!isFromReadOnlyTensor(insertSliceOp.getSource(), plan)) {
        plan.unionSets(insertSliceOp.getSource(), insertSliceOp.getDest());
      }
    }
  }
}

void BufferizationPlan::unionSets(Value v1, Value v2) {
  if (!canSetsBeMerged(v1, v2, *this)) {
    return;
  }
  // If one the sets was part of the store set, the store set
  // needs to be updated to drop the all leaders from the store set
  // and add the new leader to it.
  Value leader1 = getLeaderValue(v1);
  Value leader2 = getLeaderValue(v2);
  bool insertNewStoreLeader =
      storeLeaders.count(leader1) || storeLeaders.count(leader2);
  storeLeaders.erase(leader1);
  storeLeaders.erase(leader2);
  mappedTensors.unionSets(getPointer(v1), getPointer(v2));
  if (insertNewStoreLeader) {
    storeLeaders.insert(getLeaderValue(v1));
  }
}

void BufferizationPlan::dump() {
  llvm::dbgs() << "BufferMappings : \n";
  unsigned numSets = 0;
  for (auto it = mappedTensors.begin(), ie = mappedTensors.end(); it != ie;
       ++it) {
    if (!it->isLeader())
      continue;
    llvm::dbgs() << "\tSet " << numSets;
    if (storeLeaders.count(
            getLeaderValue(getValue(*mappedTensors.member_begin(it))))) {
      llvm::dbgs() << "(StoreSet) ";
    }
    llvm::dbgs() << ":\n";
    for (auto member : llvm::make_range(mappedTensors.member_begin(it),
                                        mappedTensors.member_end())) {
      llvm::dbgs() << "\t\t";
      getValue(member).print(llvm::dbgs());
      llvm::dbgs() << "\n";
    }
    numSets++;
  }
  llvm::dbgs() << "StoreLeaders : \n";
  for (auto storeLeader : storeLeaders) {
    storeLeader.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  }
}

LogicalResult createTensorEquivalenceClasses(func::FuncOp funcOp,
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
        .Case<tensor::PadOp>([&](tensor::PadOp padTensorOp) {
          return analysePadTensorOp(padTensorOp, plan);
        })
        .Case<DestinationStyleOpInterface>(
            [&](DestinationStyleOpInterface dpsOp) {
              return analyseDPSOps(dpsOp, plan);
            })
        .Case<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(
            [&](auto reshapeOp) {
              return analyseSingleOperandResultOp(reshapeOp.getSrc(),
                                                  reshapeOp.getResult(), plan);
            })
        .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp sliceOp) {
          return analyseSubTensorOp(sliceOp, plan);
        })
        .Case<tensor::InsertSliceOp>(
            [&](tensor::InsertSliceOp subTensorInsertOp) {
              return analyseDestructiveUpdateOp(
                  subTensorInsertOp, subTensorInsertOp.getSource(),
                  subTensorInsertOp.getDest(), subTensorInsertOp.getResult(),
                  plan);
            })
        .Case<tensor::CastOp>([&](tensor::CastOp castOp) {
          return analyseSingleOperandResultOp(castOp.getSource(),
                                              castOp.getDest(), plan);
        })
        .Case<tensor::InsertOp>([&](tensor::InsertOp insertOp) {
          return analyseDestructiveUpdateOp(insertOp, /*source =*/nullptr,
                                            insertOp.getDest(),
                                            insertOp.getResult(), plan);
        })
        .Case<vector::TransferReadOp>(
            [&](vector::TransferReadOp transferReadOp) {
              if (llvm::isa<RankedTensorType>(
                      transferReadOp.getSource().getType())) {
                plan.insert(transferReadOp.getSource());
              }
              return success();
            })
        .Case<vector::TransferWriteOp>(
            [&](vector::TransferWriteOp transferWriteOp) {
              if (!llvm::isa<RankedTensorType>(
                      transferWriteOp.getSource().getType())) {
                return success();
              }
              return analyseDestructiveUpdateOp(
                  transferWriteOp, nullptr, transferWriteOp.getSource(),
                  transferWriteOp.getResult(), plan);
            })
        .Case<scf::IfOp>(
            [&](scf::IfOp ifOp) { return analyseScfIfOp(ifOp, plan); })
        .Case<scf::ForOp>(
            [&](scf::ForOp forOp) { return analyseScfForOp(forOp, plan); })
        .Case<scf::YieldOp, tensor::EmptyOp, tensor::DimOp, tensor::ExtractOp,
              tensor::GenerateOp, tensor::PadOp, bufferization::ToMemrefOp,
              bufferization::AllocTensorOp>(
            [&](Operation *op) { return success(); })
        .Default([&](Operation *op) -> LogicalResult {
          if (llvm::any_of(op->getOperands(),
                           [](Value v) {
                             return llvm::isa<RankedTensorType>(v.getType());
                           }) ||
              llvm::any_of(op->getResultTypes(), [](Type t) {
                return llvm::isa<RankedTensorType>(t);
              })) {
            return op->emitOpError("unhandled tensor operation");
          }
          return success();
        });
  };
  if (funcOp.walk<WalkOrder::PreOrder>(bufferMappingFn).wasInterrupted()) {
    return failure();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "After First walk ";
    plan.dump();
  });

  funcOp.walk([&](Operation *updateOp) {
    if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(updateOp)) {
      hasDestructiveUpdatePattern(insertSliceOp.getDest(), plan);
      return;
    }
    if (auto vectorWriteOp = dyn_cast<vector::TransferWriteOp>(updateOp)) {
      if (llvm::isa<RankedTensorType>(vectorWriteOp.getSource().getType())) {
        hasDestructiveUpdatePattern(vectorWriteOp.getSource(), plan);
      }
    }
  });
  LLVM_DEBUG({
    llvm::dbgs() << "After Destructive update walk ";
    plan.dump();
  });

  if (funcOp
          .walk([&](IREE::Flow::DispatchTensorStoreOp storeOp) -> WalkResult {
            return analyseInterfaceStoreTensorOp(storeOp, plan);
          })
          .wasInterrupted()) {
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "After Store walk ";
    plan.dump();
  });

  return success();
}

} // namespace mlir::iree_compiler
