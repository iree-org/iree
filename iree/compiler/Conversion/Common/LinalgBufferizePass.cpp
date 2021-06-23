// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LinalgBufferizePass.cpp.cpp - Pass to bufferize Linalg on tensors --===//
//
// The overall bufferizarion algorithm is summarized here. Each of the
// individual steps are explained in detail later.
//
// Problem statement:
//
// The bufferization in this file is intended for converting tensor-operations
// into memref-operations for ops within a dispatch region. The goal is to reuse
// the buffers provided as inputs/outputs by the hal layer as memrefs for each
// of the operations. If the transformation cannot reuse input/output buffer to
// store an intermediate tensor, an allocation is done. This allocation is
// typically meant to be to target scratchspace memory.
//
// The algorithm has two phases an analysis phase and a tranformation phase.
//
// - The analysis phase walks the function and organizes relevant tensors
//   (tensors that need to be converted to memrefs) into equivalence clases. Two
//   tensors are part of the same equivalence class if they can eventually be
//   mapped to the same memref. This allows determining which operations can use
//   the buffer provided for the outputs to compute the results in place.
// - The transformation phase walks the function again and inserts corresponding
//   memref operations. The tensor operations are still kept around since the
//   analysis driving the transformation is based on the tensor values.
//   - Converting tensor operations to memref operations when all operands use
//     either buffers that are inputs to the dispatch or are allocated
//     temporarily within the dispatch region can be achieved by a
//     straight-forward walk.
//   - Reusing memref for the result of the dispatch for operations is more
//     involved and explained below.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/PassDetail.h"
#include "iree/compiler/Conversion/Passes.h"
#include "iree/compiler/Conversion/Transforms/Transforms.h"
#include "iree/compiler/Conversion/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-linalg-bufferize"

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
    if (isa<IREE::Flow::DispatchTensorStoreOp, linalg::TensorCollapseShapeOp,
            linalg::TensorExpandShapeOp>(user)) {
      return false;
    }
  }
  return true;
}

/// Class that tracks the equivalence relationship between tensors. Its a
/// light-weight wrapper around `llvm::EquivalenceClasses` to account for
/// `Value` not directly supported as a value type by this class.
class BufferizationPlan {
 public:
  llvm::EquivalenceClasses<void *>::iterator findValue(Value v) const {
    return mappedTensors.findValue(getPointer(v));
  }

  llvm::EquivalenceClasses<void *>::iterator end() const {
    return mappedTensors.end();
  }

  SmallVector<Value> getTensorsMappedToSameSet(Value v) const {
    SmallVector<Value> tensors;
    for (auto it = mappedTensors.findLeader(getPointer(v)),
              ie = mappedTensors.member_end();
         it != ie; ++it) {
      tensors.push_back(getValue(*it));
    }
    return tensors;
  }

  bool isEquivalent(Value v1, Value v2) const {
    return mappedTensors.isEquivalent(getPointer(v1), getPointer(v2));
  }

  void insert(Value v) { mappedTensors.insert(getPointer(v)); }

  void unionSets(Value v1, Value v2) {
    mappedTensors.unionSets(getPointer(v1), getPointer(v2));
  }

  /// Sets the equivalance class that contains `v` as the set that contains the
  /// result tensor of the dispatch region (i.e. a tensor that is the `value`
  /// operand of a flow.dispatch.tensor.store` op). All operations in this
  /// equivalence class can use the result buffer of the dispatch region to
  /// compute their values in place.
  void storeSet(Value v) { storeLeaders.insert(getLeaderValue(v)); }

  /// Queries if the value `v` is in the same equivalence class as the result of
  /// the dispatch region.
  bool isInStoreSet(Value v) { return storeLeaders.count(getLeaderValue(v)); }

  void dump() {
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

 private:
  Value getLeaderValue(Value v1) const {
    return getValue(mappedTensors.getLeaderValue(getPointer(v1)));
  }

  void *getPointer(Value v) const { return v.getAsOpaquePointer(); }

  Value getValue(void *v) const { return Value::getFromOpaquePointer(v); }

  llvm::EquivalenceClasses<void *> mappedTensors;

  /// Leaders of the sets that contain the result tensor of the dispatch
  /// region, i.e. a tensor that is the `value` operand of a
  /// flow.dispatch.tensor.store` op
  llvm::DenseSet<Value> storeLeaders;
};

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
      .Case<ConstantOp>([&](ConstantOp constantOp) { return true; })
      .Case<linalg::TensorCollapseShapeOp, linalg::TensorExpandShapeOp>(
          [&](auto op) { return isFromReadOnlyTensor(op.src(), plan); })
      .Case<SubTensorOp>([&](SubTensorOp subTensorOp) {
        return isFromReadOnlyTensor(subTensorOp.source(), plan);
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
static LogicalResult analyseConstantOp(ConstantOp constantOp,
                                       BufferizationPlan &plan) {
  if (!constantOp.getResult().getType().isa<ShapedType>()) return success();
  plan.insert(constantOp.getResult());
  return success();
}

/// Adds the result of the `flow.dispatch.tensor.load` op to the same
/// equivalence class as the source.
static LogicalResult analyseInterfaceLoadTensorOp(
    IREE::Flow::DispatchTensorLoadOp loadOp, BufferizationPlan &plan) {
  if (!(loadOp.getMixedOffsets().empty() && loadOp.getMixedSizes().empty() &&
        loadOp.getMixedStrides().empty()) &&
      !canUsersHandleSubviews(loadOp)) {
    plan.insert(loadOp.source());
    plan.insert(loadOp.result());
    return success();
  }
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
  if (!(storeOp.getMixedOffsets().empty() && storeOp.getMixedSizes().empty() &&
        storeOp.getMixedStrides().empty())) {
    SmallVector<Value> mappedTensors = plan.getTensorsMappedToSameSet(value);
    for (auto v : mappedTensors) {
      // TODO(ravishankarm): At this point it is not clear why the following
      // restriction exists. It might have something to do with subviews and
      // reshapes not working well together, but there is no comment about why
      // this was added with the change that added this.
      Operation *op = v.getDefiningOp();
      if (op && isa<linalg::TensorCollapseShapeOp, linalg::TensorExpandShapeOp>(
                    v.getDefiningOp())) {
        return false;
      }
    }
  }

  Value target = storeOp.target();
  auto targetInterfaceOp =
      getEquivalentOpOfType<IREE::HAL::InterfaceBindingSubspanOp>(target, plan);
  assert(targetInterfaceOp);
  if (auto valueConstantOp = getEquivalentOpOfType<ConstantOp>(value, plan)) {
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
    if (linalgOp.payloadUsesValueFromOperand(outTensor.value())) {
      // If the `outs` tensor has a single use (this op) and is not from a
      // read-only buffer, the `outs` tensor can be tied to the result.
      if (outTensor.value()->get().hasOneUse() &&
          !isFromReadOnlyTensor(outTensor.value()->get(), plan)) {
        tiedOperands[outTensor.index()] = outTensor.value()->get();
      }
    }
  }
  for (auto result : llvm::enumerate(outputOperands)) {
    // If the output tensor is not actually used (for initialization) by this
    // op, we can reuse the result tensor's buffer for some operands.
    // TODO(#5040): A better way to handle this case is to allocate a buffer and
    // then vectorization + load-store forwarding to remove the intermediate
    // buffer. This requires vectorization to handle all cases downstream. This
    // is a WAR for current use cases.
    if (linalgOp.payloadUsesValueFromOperand(result.value())) {
      continue;
    }
    for (auto input : linalgOp.getInputTensorOperands()) {
      auto producerOp = input->get().getDefiningOp<linalg::LinalgOp>();
      if (producerOp && input->get().hasOneUse() &&
          input->get().getType() == result.value()->get().getType() &&
          linalgOp.getTiedIndexingMap(input) ==
              linalgOp.getTiedIndexingMap(result.value())) {
        assert(!tiedOperands[result.index()]);
        tiedOperands[result.index()] = input->get();
        break;
      }
    }
  }
  return tiedOperands;
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
    if (!isa<memref::DimOp>(use.getOwner())) {
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

static LogicalResult analyseSubTensorOp(SubTensorOp subTensorOp,
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
///     %yieldVal = subtensor_insert %val, %arg1[...]
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
          .Case<SubTensorOp>([&](SubTensorOp subTensorOp) {
            return subTensorOp.source() == arg;
          })
          .Case<SubTensorInsertOp>([&](SubTensorInsertOp subTensorInsertOp) {
            return subTensorInsertOp.dest() == arg;
          })
          .Case<memref::DimOp, scf::YieldOp>([&](auto op) { return true; })
          .Default([&](Operation *op) { return false; });
    };
    if (llvm::all_of(arg.getUses(), isDestructiveUpdateUses)) {
      for (Operation *user : arg.getUsers()) {
        TypeSwitch<Operation *>(user)
            .Case<SubTensorOp>([&](SubTensorOp subTensorOp) {
              plan.unionSets(subTensorOp.source(), subTensorOp.result());
            })
            .Case<SubTensorInsertOp>([&](SubTensorInsertOp subTensorInsertOp) {
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

static LogicalResult analyseOperations(FuncOp funcOp, BufferizationPlan &plan) {
  auto bufferMappingFn = [&](Operation *op) -> WalkResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<ConstantOp>([&](ConstantOp constantOp) {
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
        .Case<IREE::Flow::DispatchTieShapeOp>(
            [&](IREE::Flow::DispatchTieShapeOp tieShapeOp) {
              return analyseSingleOperandResultOp(tieShapeOp.operand(),
                                                  tieShapeOp.result(), plan);
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
        .Case<linalg::TensorCollapseShapeOp, linalg::TensorExpandShapeOp>(
            [&](auto reshapeOp) {
              return analyseSingleOperandResultOp(reshapeOp.src(),
                                                  reshapeOp.result(), plan);
            })
        .Case<SubTensorOp>([&](SubTensorOp subTensorOp) {
          return analyseSubTensorOp(subTensorOp, plan);
        })
        .Case<SubTensorInsertOp>([&](SubTensorInsertOp subTensorInsertOp) {
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
  return success();
}

//===----------------------------------------------------------------------===//
// Bufferization helper functions using BlockAndValueMapping.
//===----------------------------------------------------------------------===//

/// Returns the dynamic dimensions of a Value `v` that is assumed to be
/// ShapedType.
static SmallVector<Value, 4> getDynamicDims(OpBuilder &b, Location loc,
                                            Value v) {
  SmallVector<Value, 4> dynamicDims;
  for (auto shape : enumerate(v.getType().cast<ShapedType>().getShape())) {
    if (shape.value() == ShapedType::kDynamicSize) {
      dynamicDims.push_back(
          b.createOrFold<memref::DimOp>(loc, v, shape.index()));
    }
  }
  return dynamicDims;
}

/// Allocates a memref for the results of an operation. Uses the
/// `InferShapedTypeOpInterface` where possible to get the shape of the output
/// in terms of the shapes of the operands.
static Value allocateBufferForResult(OpBuilder &b, Operation *op,
                                     unsigned resultNum,
                                     WorkgroupMemoryAllocationFn allocationFn) {
  assert(op->getNumResults() > resultNum);
  RankedTensorType resultType =
      op->getResult(resultNum).getType().cast<RankedTensorType>();
  SmallVector<Value, 4> dynamicDims;

  // Get the shape of the result
  Location loc = op->getLoc();
  if (auto shapedOp = dyn_cast<InferShapedTypeOpInterface>(op)) {
    SmallVector<SmallVector<Value>> resultShape;
    if (failed(shapedOp.reifyReturnTypeShapesPerResultDim(b, resultShape))) {
      return nullptr;
    }
    for (auto shape : enumerate(resultShape[resultNum])) {
      if (resultType.isDynamicDim(shape.index())) {
        dynamicDims.push_back(shape.value());
      }
    }
  } else if (auto loadOp = dyn_cast<IREE::Flow::DispatchTensorLoadOp>(op)) {
    dynamicDims = llvm::to_vector<4>(loadOp.sizes());
  } else if (auto subTensorOp = dyn_cast<SubTensorOp>(op)) {
    dynamicDims = llvm::to_vector<4>(subTensorOp.sizes());
  } else if (auto subTensorInsertOp = dyn_cast<SubTensorInsertOp>(op)) {
    dynamicDims = getDynamicDims(b, loc, subTensorInsertOp.dest());
  } else if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op)) {
    dynamicDims = getDynamicDims(b, loc, transferWriteOp.source());
  } else {
    dynamicDims = getDynamicDims(b, loc, op->getResult(resultNum));
  }

  // If its a static allocation hoist it all the way up at begining of the
  // function.
  if (dynamicDims.empty()) {
    auto funcOp = op->getParentOfType<FuncOp>();
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&funcOp.front());
    return allocationFn(b, loc, resultType.getShape(),
                        resultType.getElementType(), dynamicDims);
  }
  return allocationFn(b, loc, resultType.getShape(),
                      resultType.getElementType(), dynamicDims);
}

template <typename TensorType>
static MemRefType getMemrefTypeForTensor(TensorType tensorType,
                                         ArrayRef<AffineMap> layout = {},
                                         unsigned memorySpace = 0) {
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         layout, memorySpace);
}

/// Creates a subview operation given the `src`, `offsets`, `sizes` and
/// `strides`. Handles the corner case where the `offsets`, `sizes` and
/// `strides` are empty in which case just forward the `src` value.
/// TODO(ataei): Instead create memref.subview %v [][][] folder.
static Value createSubviewOp(OpBuilder &b, Location loc, Value src,
                             ArrayRef<OpFoldResult> offsets,
                             ArrayRef<OpFoldResult> sizes,
                             ArrayRef<OpFoldResult> strides,
                             MemRefType resultType = MemRefType()) {
  if (offsets.empty() && sizes.empty() && strides.empty()) return src;
  return b.create<memref::SubViewOp>(loc, resultType, src, offsets, sizes,
                                     strides);
}

//===----------------------------------------------------------------------===//
// There might be cases when the `value` stored into a
// `flow.dispatch.tensor.store` operation is obtained from operation that
// computes the value (say a `linalg` operation) through a series of `reshapes`,
// `cast` etc. When trying to reuse the buffer for the result passed in to the
// dispatch region for these operations, these operations need to be "replayed"
// in reverse so that the type of the buffer in the operation computing the
// value matches what is expected.
//
// For example,
// ```mlir
//   %buffer = hal.interface.binding.subspan .. : tensor<?xf32>
//   %result = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
//       outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
//   %value = linalg.tensor_reshape %result [affine_map<(d0, d1) -> (d0, d1)]
//       : tensor<?x?xf32> into tensor<?xf32>
//   flow.dispatch.tensor.store %value, %buffer[..] [..] [..]
// ```
//
// needs to be converted to
//
// ```mlir
//   %buffer = hal.interface.binding.subspan .. : memref<?xf32>
//   %result = subview %buffer[..] [..] [..] : memref<?xf32>
//   %value = linalg.reshape %result [affine_map<(d0, d1) -> (d0, d1)]
//       : memref<?xf32> into memref<?x?xf32>
//   linalg.matmul ins(%lhs, %rhs : memref<?x?xf32>, memref<?x?xf32>)
//       outs(%result : memref<?x?xf32>)
//   flow.dispatch.tensor.store %value, %buffer[..] [..] [..]
// ```
//
// ===----------------------------------------------------------------------===//

/// For a given store-like `op` that is to be replaced, find the insertion point
/// in the same block earliest possible when
/// - the replacement op uses values in `usedValues`, so has to be inserted
///   after the ops that define these.
/// - The op needs to be inserted before `insertBefore` (which is in the same
/// block). Return nullptr all other times.
static Operation *getInsertionPointForReplacementStoreOp(
    Operation *op, Operation *insertBefore, ArrayRef<Value> usedValues) {
  if (op->getBlock() != insertBefore->getBlock()) return nullptr;
  Operation *insertAfter = nullptr;
  for (auto value : usedValues) {
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp || definingOp->getBlock() != insertBefore->getBlock())
      continue;
    if (!insertAfter || insertAfter->isBeforeInBlock(definingOp))
      insertAfter = definingOp;
  }
  // All defining ops are outside of the block, so just insert at the start of
  // the block.
  if (!insertAfter) return &(op->getBlock()->front());
  if (insertAfter->isBeforeInBlock(insertBefore))
    return insertAfter->getNextNode();
  return nullptr;
}

/// Returns the subview into the buffer that is supposed to be populated with
/// the `value` of the `flow.dispatch.tensor.store` operation. This can be used
/// to compute the results in place.
static Value getSubviewOpForTensorStoreOp(OpBuilder &b, Operation *storeOp,
                                          const BlockAndValueMapping &bvm) {
  SmallVector<Value, 4> operandsOfSubviewOp;
  auto op = cast<OffsetSizeAndStrideOpInterface>(storeOp);
  Value target =
      TypeSwitch<Operation *, Value>(op)
          .Case<IREE::Flow::DispatchTensorStoreOp>(
              [&](auto storeOp) { return storeOp.target(); })
          .Case<SubTensorInsertOp>([&](auto storeOp) { return storeOp.dest(); })
          .Default([](Operation *) { return nullptr; });
  if (!target) return nullptr;
  operandsOfSubviewOp.push_back(bvm.lookup(target));
  operandsOfSubviewOp.append(op.offsets().begin(), op.offsets().end());
  operandsOfSubviewOp.append(op.sizes().begin(), op.sizes().end());
  operandsOfSubviewOp.append(op.strides().begin(), op.strides().end());
  Operation *insertBefore = &(*b.getInsertionPoint());
  Operation *insertionPoint = getInsertionPointForReplacementStoreOp(
      op.getOperation(), insertBefore, operandsOfSubviewOp);
  if (!insertionPoint) return nullptr;
  OpBuilder::InsertionGuard g(b);
  Value subview =
      createSubviewOp(b, op.getLoc(), bvm.lookup(target), op.getMixedOffsets(),
                      op.getMixedSizes(), op.getMixedStrides());
  return subview;
}

/// Gets the reverse of a
/// `linalg.tensor_expand_shape`/`linalg.tensor_collapse_shape` op to get a
/// memref type that can be used for in-place computation of the result of a
/// dispatch region.
template <typename TensorReshapeOpTy>
static Value getReverseOfReshapeOp(OpBuilder &b, TensorReshapeOpTy reshapeOp,
                                   Value resultBuffer) {
  auto memrefType = getMemrefTypeForTensor(
      reshapeOp.getSrcType(), {},
      resultBuffer.getType().cast<MemRefType>().getMemorySpaceAsInt());
  using ReverseReshapeOpTy = typename std::conditional<
      std::is_same<TensorReshapeOpTy, linalg::TensorCollapseShapeOp>::value,
      linalg::ExpandShapeOp, linalg::CollapseShapeOp>::type;
  return b.create<ReverseReshapeOpTy>(reshapeOp.getLoc(), memrefType,
                                      resultBuffer, reshapeOp.reassociation());
}

/// Gets the reverse of a `tensor.cast` op to get a memref type that
/// can be used for in-place computation of the result of a disaptch region.
static Value getReverseOfCastOp(OpBuilder &b, tensor::CastOp castOp,
                                Value resultBuffer) {
  auto memrefType = getMemrefTypeForTensor(
      castOp.source().getType().cast<RankedTensorType>(),
      resultBuffer.getType().cast<MemRefType>().getAffineMaps(),
      resultBuffer.getType().cast<MemRefType>().getMemorySpaceAsInt());
  return b.create<memref::CastOp>(castOp.getLoc(), memrefType, resultBuffer);
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
  return nullptr;
}

/// To perform updates directly into the result buffer, the uses need to be
/// walked to get to a value already mapped to a buffer or a
/// `flow.dispatch.tensor.store` operation. For each use, gets the tied result
/// and follow its uses. The traversed uses and thir tied results are returned
/// in `traversedUses`.
static Value walkUseToGetResultBuffer(
    OpBuilder &b, Value value, const BufferizationPlan &plan,
    const BlockAndValueMapping &bvm,
    SmallVectorImpl<std::pair<OpOperand *, Value>> &traversedUses) {
  Operation *user = nullptr;
  while (value.hasOneUse()) {
    OpOperand &use = *value.use_begin();
    user = use.getOwner();
    if (isa<IREE::Flow::DispatchTensorStoreOp>(user)) {
      return getSubviewOpForTensorStoreOp(b, user, bvm);
    }
    value = getTiedResultForOperand(use, plan);
    if (!value) return nullptr;
    traversedUses.push_back(std::make_pair(&use, value));
    if (auto resultBuffer = bvm.lookupOrNull(value)) return resultBuffer;
  }
  return nullptr;
}

/// For an operation whose `resultValue` is the result of the dispatch region,
/// gets the buffer to use to compute the value in-place.
static Value getInplaceResultBuffer(OpBuilder &b, OpResult resultValue,
                                    const BufferizationPlan &plan,
                                    BlockAndValueMapping &bvm) {
  // Traverse the use-def chains to get the `flow.dispatch.tensor.store`
  // operation keeping track of all the traversed operations. Note that the
  // equivalence set construction should ensure that all operations traversed
  // here have a single use.
  SmallVector<std::pair<OpOperand *, Value>> traversedUses;
  Value resultBuffer =
      walkUseToGetResultBuffer(b, resultValue, plan, bvm, traversedUses);
  if (!resultBuffer) return nullptr;

  DEBUG_WITH_TYPE(DEBUG_TYPE, {
    llvm::dbgs() << "Pair :\n\tTensor :";
    resultValue.getOwner()->print(llvm::dbgs());
    llvm::dbgs() << "\nt\tMemref :";
    resultBuffer.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Now replay the instructions that are essentially doing type-conversion, in
  // reverse, to get the type needed for the operation computing the value.
  for (auto &it : llvm::reverse(traversedUses)) {
    Operation *op = it.first->getOwner();
    resultBuffer =
        TypeSwitch<Operation *, Value>(op)
            .Case<scf::IfOp, scf::ForOp, linalg::LinalgOp, SubTensorInsertOp,
                  vector::TransferWriteOp>(
                [&](auto op) { return resultBuffer; })
            .Case<linalg::TensorCollapseShapeOp, linalg::TensorExpandShapeOp>(
                [&](auto reshapeOp) {
                  return getReverseOfReshapeOp(b, reshapeOp, resultBuffer);
                })
            .Case<tensor::CastOp>([&](tensor::CastOp castOp) {
              return getReverseOfCastOp(b, castOp, resultBuffer);
            })
            .Default([&](Operation *) { return nullptr; });
    // TODO(ravishankarm): Maybe this needs to be an error. This should not have
    // happened.
    if (!resultBuffer) return nullptr;
    Value tiedResult = it.second;
    bvm.map(tiedResult, resultBuffer);
    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "Pair :\n\tTensor result "
                   << tiedResult.cast<OpResult>().getResultNumber() << " :";
      op->print(llvm::dbgs());
      llvm::dbgs() << "\nt\tMemref :";
      resultBuffer.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
  }
  return resultBuffer;
}

/// Converts a `tensor.cast` operation into a `memref.cast` operation with the
/// result aliasing the buffer for the operand.
static Value getAliasingBufferForResult(OpBuilder &b, tensor::CastOp castOp,
                                        BlockAndValueMapping &bvm) {
  Value inputBuffer = bvm.lookup(castOp.source());
  Value resultTensor = castOp.dest();
  auto outputType = getMemrefTypeForTensor(
      resultTensor.getType().cast<RankedTensorType>(), {},
      inputBuffer.getType().cast<MemRefType>().getMemorySpaceAsInt());
  return b.create<memref::CastOp>(castOp.getLoc(), outputType, inputBuffer);
}

/// Returns the subview that indexes into the source of the interface buffer.
static Value getAliasingBufferForResult(OpBuilder &b,
                                        IREE::Flow::DispatchTensorLoadOp loadOp,
                                        BlockAndValueMapping &bvm) {
  Location loc = loadOp.getLoc();
  Value memref = bvm.lookup(loadOp.source());
  return createSubviewOp(b, loc, memref, loadOp.getMixedOffsets(),
                         loadOp.getMixedSizes(), loadOp.getMixedStrides());
}

/// Converts a `linalg.tensor_collapse/expand_shape` operation to a
/// `linalg.collapse/expand_shape` operation with the result aliasing the buffer
/// for the operand.
template <typename TensorReshapeOpTy>
static Value getAliasingBufferForReshapeResult(OpBuilder &b,
                                               TensorReshapeOpTy op,
                                               BlockAndValueMapping &bvm) {
  Location loc = op.getLoc();
  Value srcTensor = op.src();
  RankedTensorType resultTensorType = op.getResultType();
  Value inputBuffer = bvm.lookup(srcTensor);

  // Create the reshape op.
  MemRefType inputBufferType = inputBuffer.getType().cast<MemRefType>();
  auto reshapeResultType = getMemrefTypeForTensor(
      resultTensorType, {}, inputBufferType.getMemorySpaceAsInt());
  using ReshapeOpTy = typename std::conditional<
      std::is_same<TensorReshapeOpTy, linalg::TensorCollapseShapeOp>::value,
      linalg::CollapseShapeOp, linalg::ExpandShapeOp>::type;
  Value bufferReshape = b.create<ReshapeOpTy>(loc, reshapeResultType,
                                              inputBuffer, op.reassociation());
  return bufferReshape;
}

/// Converts a `subtensor` operation to a `subview` operation.
static Value getAliasingBufferForResult(OpBuilder &b, SubTensorOp op,
                                        BlockAndValueMapping &bvm) {
  Location loc = op.getLoc();
  Value srcTensor = op.source();
  Value inputBuffer = bvm.lookup(srcTensor);

  ShapedType sourceType = op.getSourceType();
  ShapedType resultType = op.getType();
  SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = op.getMixedSizes();
  SmallVector<OpFoldResult> strides = op.getMixedStrides();
  MemRefType subViewResultType =
      (resultType.getRank() < sourceType.getRank()
           ? memref::SubViewOp::inferRankReducedResultType(
                 resultType.getRank(), inputBuffer.getType().cast<MemRefType>(),
                 offsets, sizes, strides)
                 .cast<MemRefType>()
           : MemRefType());
  return b.create<memref::SubViewOp>(loc, subViewResultType, inputBuffer,
                                     offsets, sizes, strides);
}

/// Returns output buffers that aliases inputs.
static SmallVector<Value> getAliasingBuffersForResult(
    scf::ForOp scfFor, BlockAndValueMapping &bvm) {
  SmallVector<Value> aliasedBuffers(scfFor.results().size(), nullptr);
  for (int i = 0; i < scfFor.results().size(); ++i) {
    Value inputTensor = scfFor.initArgs()[i];
    if (!inputTensor.getType().isa<RankedTensorType>()) continue;
    Value inputBuffer = bvm.lookup(inputTensor);
    aliasedBuffers[i] = inputBuffer;
  }
  return aliasedBuffers;
}

/// Returns a `memref` for every result that aliases the buffer for one of its
/// operands. Returns the memref of the right shape/type based on the operation.
static SmallVector<Value, 4> getAliasingBuffersForResults(
    OpBuilder &b, Operation *op, BlockAndValueMapping &bvm) {
  return TypeSwitch<Operation *, SmallVector<Value, 4>>(op)
      .Case<IREE::Flow::DispatchTensorLoadOp, SubTensorOp, tensor::CastOp>(
          [&](auto singleResultOp) -> SmallVector<Value, 4> {
            return {getAliasingBufferForResult(b, singleResultOp, bvm)};
          })
      .Case<linalg::TensorCollapseShapeOp, linalg::TensorExpandShapeOp>(
          [&](auto reshapeOp) -> SmallVector<Value, 4> {
            return {getAliasingBufferForReshapeResult(b, reshapeOp, bvm)};
          })
      .Case<scf::ForOp>([&](auto scfFor) -> SmallVector<Value> {
        return getAliasingBuffersForResult(scfFor, bvm);
      })
      .Default([&](Operation *op) -> SmallVector<Value, 4> {
        return SmallVector<Value, 4>(op->getNumResults(), nullptr);
      });
}

/// For a result value, gets the operand that is tied with it. If no such
/// operand exists, returns `nullptr`.
static Value getTiedOperandForResult(OpResult result,
                                     const BufferizationPlan &plan) {
  for (Value operand : result.getOwner()->getOperands()) {
    if (plan.isEquivalent(operand, result)) {
      return operand;
    }
  }
  return nullptr;
}
static bool hasTiedOperandForResult(OpResult result,
                                    const BufferizationPlan &plan) {
  return static_cast<bool>(getTiedOperandForResult(result, plan));
}

/// Computes the `memrefs` to use for the result of an operation based on
/// - If the result has a tied operand reuse the buffer for the tied operand (or
///   an alias of it) as the buffer for the result. The `alaisingBuffer` vector
///   is expected to be as large as the number of results.
/// - If the result has no tied operands, the corresponding position in the
///   `aliasingBuffer` list must be `nullptr`.
/// - If the result is in the same equivalence set as the result of the dispatch
///   region (i.e. `value` operand of a `flow.dispatch.tensor.store`) then
///   return an alias/view of the buffer passed into the dispatch region to
///   store the results.
/// - Lastly, allocate a temporary buffer for the result using the passed
///   allocation function.
static LogicalResult getOrAllocateResultBuffers(
    OpBuilder &b, Operation *op, ArrayRef<Value> aliasingBuffers,
    BlockAndValueMapping &bvm, BufferizationPlan &plan,
    WorkgroupMemoryAllocationFn allocationFn) {
  assert(aliasingBuffers.size() == op->getNumResults());
  auto results = op->getResults();
  for (auto result : llvm::enumerate(results)) {
    if (!result.value().getType().isa<RankedTensorType>() ||
        bvm.contains(result.value())) {
      continue;
    }
    Value buffer;

    if (aliasingBuffers[result.index()] &&
        hasTiedOperandForResult(result.value(), plan)) {
      buffer = aliasingBuffers[result.index()];
    }
    if (!buffer && plan.isInStoreSet(result.value())) {
      buffer = getInplaceResultBuffer(b, result.value(), plan, bvm);
    }
    if (!buffer) {
      buffer = allocateBufferForResult(b, op, result.index(), allocationFn);
    }
    if (!buffer) {
      return op->emitError("unable to get result buffer for op");
    }
    bvm.map(result.value(), buffer);
    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "Pair :\n\tTensor result " << result.index() << ":";
      op->print(llvm::dbgs());
      llvm::dbgs() << "\nt\tMemref :";
      buffer.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
  }
  return success();
}

/// Convenience wrapper around core allocation function for the case where the
/// alias is the buffer for the result directly.
static LogicalResult getOrAllocateResultBuffers(
    OpBuilder &b, Operation *op, BlockAndValueMapping &bvm,
    BufferizationPlan &plan, WorkgroupMemoryAllocationFn allocationFn) {
  auto aliasingBuffers = llvm::to_vector<4>(
      llvm::map_range(op->getResults(), [&](OpResult result) {
        Value tiedOperand = getTiedOperandForResult(result, plan);
        return tiedOperand ? bvm.lookupOrNull(tiedOperand) : tiedOperand;
      }));
  return getOrAllocateResultBuffers(b, op, aliasingBuffers, bvm, plan,
                                    allocationFn);
}

/// Generic conversion pattern that matches any linalg::LinalgOp. This avoids
/// template instantiating one pattern for each linalg::LinalgOp. The method
/// expects all operands and results have already been mapped to memrefs.
static LogicalResult convertAnyLinalgOp(
    OpBuilder &b, linalg::LinalgOp op, BlockAndValueMapping &bvm,
    BufferizationPlan &plan, WorkgroupMemoryAllocationFn allocationFn) {
  // Skip linalg ops inserted by this pass.
  if (op.hasBufferSemantics()) return success();

  Location loc = op.getLoc();
  SmallVector<Value, 2> newInputBuffers;
  newInputBuffers.reserve(op.getNumInputs());
  for (OpOperand *opOperand : op.getInputOperands()) {
    if (op.isScalar(opOperand)) {
      newInputBuffers.push_back(opOperand->get());
      continue;
    }
    // For `linalg.poolin_*` ops, the input might be from a
    // `linalg.init_tensor`. In such cases, the `BlockAndValueMapping` wont have
    // a mapping for the buffer. Allocate a buffer for these.
    Value inputBuffer = bvm.lookupOrNull(opOperand->get());
    if (!inputBuffer) {
      OpResult definingOpResult = opOperand->get().dyn_cast<OpResult>();
      if (!definingOpResult) return failure();
      inputBuffer = allocateBufferForResult(b, definingOpResult.getOwner(),
                                            definingOpResult.getResultNumber(),
                                            allocationFn);
    }
    newInputBuffers.push_back(inputBuffer);
  }
  SmallVector<Value, 2> newOutputBuffers;
  auto results = op.getOperation()->getResults();
  auto outputs = op.getOutputOperands();
  for (auto it : llvm::zip(results, outputs)) {
    Value resultTensor = std::get<0>(it);
    Value resultBuffer = bvm.lookup(resultTensor);

    OpOperand *outOperand = std::get<1>(it);
    Value outTensor = outOperand->get();
    Value outBuffer = bvm.lookupOrNull(outTensor);
    if (outBuffer && !plan.isEquivalent(outTensor, resultTensor) &&
        op.payloadUsesValueFromOperand(outOperand)) {
      b.create<linalg::CopyOp>(loc, outBuffer, resultBuffer);
    }
    newOutputBuffers.push_back(resultBuffer);
  }

  SmallVector<Value, 8> newOperands(newInputBuffers.begin(),
                                    newInputBuffers.end());
  newOperands.append(newOutputBuffers.begin(), newOutputBuffers.end());
  auto otherOperands =
      llvm::map_range(op.getAssumedNonShapedOperands(),
                      [&bvm](Value v) { return bvm.lookupOrDefault(v); });
  newOperands.append(otherOperands.begin(), otherOperands.end());
  op.clone(b, loc, {}, newOperands);
  return success();
}

/// Constants that return tensor types can be handled natively by the
/// backends. Here just provide a cast to memref to bridge the gap from tensors
/// to memrefs.
static LogicalResult convertConstantOp(OpBuilder &b, ConstantOp constantOp,
                                       BlockAndValueMapping &bvm) {
  Value result = constantOp.getResult();
  assert(!bvm.lookupOrNull(result));
  RankedTensorType tensorType = result.getType().dyn_cast<RankedTensorType>();
  if (!tensorType) return success();
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointAfter(constantOp);
  auto memrefType = getMemrefTypeForTensor(tensorType);
  Value memref =
      b.create<memref::BufferCastOp>(constantOp.getLoc(), memrefType, result);
  bvm.map(result, memref);
  return success();
}

static LogicalResult convertDispatchTieShapeOp(
    OpBuilder &b, IREE::Flow::DispatchTieShapeOp shapeOp,
    BlockAndValueMapping &bvm) {
  if (Value v = bvm.lookupOrNull(shapeOp.operand())) {
    auto tieShapeOp = b.create<Shape::TieShapeOp>(shapeOp.getLoc(), v.getType(),
                                                  v, shapeOp.shape());
    bvm.map(shapeOp.getResult(), tieShapeOp.getResult());
  }
  return success();
}

/// Converts a `tensor.extract` operation into a `load`.
static LogicalResult convertTensorExtractOp(OpBuilder &b, tensor::ExtractOp op,
                                            const BlockAndValueMapping &bvm) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  Value inputBuffer = bvm.lookup(op.tensor());
  Value load =
      b.createOrFold<memref::LoadOp>(op.getLoc(), inputBuffer, op.indices());
  // Since the value is the scalar, and `bvm` is used to only track tensor ->
  // memref mappings, just replace the uses directly.
  op.result().replaceAllUsesWith(load);
  return success();
}

/// Converts a `flow.dispatch.tensor.store` operation to memrefs. If the `value`
/// and `target` are in the same equivalent set, then there is nothing to do. If
/// no create a subview into the result buffer and copy the `value`.
static LogicalResult convertInterfaceStoreTensorOp(
    OpBuilder &b, IREE::Flow::DispatchTensorStoreOp storeOp,
    BlockAndValueMapping &bvm, BufferizationPlan &plan) {
  if (plan.isEquivalent(storeOp.target(), storeOp.value())) {
    return success();
  }
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(storeOp);
  Value storeTo = bvm.lookup(storeOp.target());
  Value storeFrom = bvm.lookup(storeOp.value());
  Value subview =
      createSubviewOp(b, storeOp.getLoc(), storeTo, storeOp.getMixedOffsets(),
                      storeOp.getMixedSizes(), storeOp.getMixedStrides());

  b.create<linalg::CopyOp>(storeOp->getLoc(), storeFrom, subview);
  return success();
}

/// Converts a `subtensor_insert` operation to buffers by
/// - Allocating a buffer for the result (if needed), and copying the
///   destination value into this buffer.
/// - Copying the source values into a subview of the result buffer.
static LogicalResult convertSubTensorInsertOp(OpBuilder &b,
                                              SubTensorInsertOp op,
                                              BlockAndValueMapping &bvm,
                                              BufferizationPlan &plan) {
  Location loc = op.getLoc();
  Value result = op.getResult();
  ShapedType resultType = op.getType();
  Value resultBuffer = bvm.lookup(result);

  // If `dest` and `result` are not equivalent, need a copy for that.
  Value dest = op.dest();
  if (!plan.isEquivalent(dest, result)) {
    Value destBuffer = bvm.lookup(dest);
    b.create<linalg::CopyOp>(loc, destBuffer, resultBuffer);
  }

  Value source = op.source();
  if (plan.isEquivalent(source, dest)) {
    return success();
  }

  // Copy from the source to the result subview.
  ShapedType sourceType = op.getSourceType();
  Value sourceBuffer = bvm.lookup(source);
  SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = op.getMixedSizes();
  SmallVector<OpFoldResult> strides = op.getMixedStrides();
  MemRefType subViewResultType =
      (sourceType.getRank() < resultType.getRank()
           ? memref::SubViewOp::inferRankReducedResultType(
                 sourceType.getRank(),
                 resultBuffer.getType().cast<MemRefType>(), offsets, sizes,
                 strides)
                 .cast<MemRefType>()
           : MemRefType());
  Value subViewOp = createSubviewOp(b, loc, resultBuffer, offsets, sizes,
                                    strides, subViewResultType);
  b.create<linalg::CopyOp>(loc, sourceBuffer, subViewOp);
  return success();
}

/// Converts a `tensor.insert` operations into a `memref.store`.
static LogicalResult convertTensorInsertOp(OpBuilder &b, tensor::InsertOp op,
                                           BlockAndValueMapping &bvm,
                                           BufferizationPlan &plan) {
  Location loc = op.getLoc();
  Value result = op.result();
  Value resultBuffer = bvm.lookup(result);
  if (!plan.isEquivalent(op.dest(), result)) {
    Value destBuffer = bvm.lookup(op.dest());
    b.create<linalg::CopyOp>(loc, destBuffer, resultBuffer);
  }

  b.create<memref::StoreOp>(loc, op.scalar(), resultBuffer, op.indices());
  return success();
}

/// Converts a vector.transfer_write op to use memref operands for source.
static LogicalResult convertVectorTransferWriteOp(OpBuilder &b,
                                                  vector::TransferWriteOp op,
                                                  BlockAndValueMapping &bvm,
                                                  BufferizationPlan &plan) {
  Location loc = op.getLoc();
  Value result = op.result();
  RankedTensorType resultType = result.getType().dyn_cast<RankedTensorType>();
  if (!resultType) return success();
  Value resultBuffer = bvm.lookup(result);

  if (!plan.isEquivalent(op.source(), result)) {
    Value destBuffer = bvm.lookup(op.source());
    b.create<linalg::CopyOp>(loc, destBuffer, resultBuffer);
  }

  // Create a new vector.transfer_write operation without a result value.
  b.create<vector::TransferWriteOp>(
      loc, op.vector(), resultBuffer, op.indices(), op.permutation_map(),
      op.mask(), op.in_bounds() ? *op.in_bounds() : ArrayAttr());
  return success();
}

static LogicalResult convertScfForOp(OpBuilder &b, scf::ForOp forOp,
                                     BlockAndValueMapping &bvm,
                                     BufferizationPlan &plan) {
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  Location loc = forOp.getLoc();
  for (auto arg : llvm::enumerate(forOp.getRegionIterArgs())) {
    if (!arg.value().getType().isa<RankedTensorType>()) continue;
    Value resultTensor = forOp.getResult(arg.index());
    Value resultBuffer = bvm.lookup(resultTensor);
    OpOperand &initOperand = forOp.getOpOperandForRegionIterArg(arg.value());
    Value yieldOperand = yieldOp.getOperand(arg.index());
    bvm.map(arg.value(), resultBuffer);
    bvm.map(yieldOperand, resultBuffer);
    if (!plan.isEquivalent(arg.value(), initOperand.get())) {
      Value initBuffer = bvm.lookup(initOperand.get());
      b.create<linalg::CopyOp>(loc, initBuffer, resultBuffer);
    }
  }
  return success();
}

static LogicalResult convertScfIfOp(OpBuilder &b, scf::IfOp ifOp,
                                    BlockAndValueMapping &bvm,
                                    BufferizationPlan &plan) {
  auto thenYieldOp = ifOp.thenYield();
  auto elseYieldOp = ifOp.elseYield();
  for (auto result : llvm::enumerate(ifOp.getResults())) {
    Value resultValue = result.value();
    if (!resultValue.getType().isa<RankedTensorType>()) continue;
    Value resultBuffer = bvm.lookup(resultValue);
    Value thenYield = thenYieldOp->getOperand(result.index());
    Value elseYield = elseYieldOp->getOperand(result.index());
    bvm.map(thenYield, resultBuffer);
    bvm.map(elseYield, resultBuffer);
  }
  return success();
}

/// If the alias of the buffer for an input oeprand cannot be used for the
/// "tied" results, need to do an explicit copy of the memory pointed to by the
/// aliased buffer into the buffer assigned to the result.
static void copyFromAliasingBufferToResultBuffer(
    OpBuilder &b, Location loc, ArrayRef<Value> tiedOperands,
    ArrayRef<Value> tiedResults, ArrayRef<Value> aliasingBuffers,
    BlockAndValueMapping &bvm, BufferizationPlan &plan) {
  for (auto result : enumerate(tiedResults)) {
    Value operand = tiedOperands[result.index()];
    if (!plan.isEquivalent(result.value(), operand)) {
      b.create<linalg::CopyOp>(loc, aliasingBuffers[result.index()],
                               bvm.lookup(result.value()));
    }
  }
}

/// Returns the static/dynamic mixed sizes of the memref.
static SmallVector<OpFoldResult> getMemrefSizes(OpBuilder &b, Location loc,
                                                Value memref) {
  auto inputShape = memref.getType().cast<ShapedType>().getShape();
  SmallVector<OpFoldResult> sizeMixedValues;
  for (int64_t i = 0; i < inputShape.size(); ++i) {
    if (inputShape[i] == ShapedType::kDynamicSize) {
      Value dim = b.create<memref::DimOp>(loc, memref, i);
      sizeMixedValues.push_back(dim);
    } else {
      sizeMixedValues.push_back(b.getI64IntegerAttr(inputShape[i]));
    }
  }
  return sizeMixedValues;
}

static LogicalResult convertPadTensorOp(OpBuilder &b,
                                        linalg::PadTensorOp padTensorOp,
                                        BlockAndValueMapping &bvm) {
  auto inputTensor = padTensorOp.source();
  auto inputMemref = bvm.lookup(inputTensor);

  auto loc = padTensorOp.getLoc();

  auto resultPaddedBuffer = bvm.lookup(padTensorOp.result());

  // Get padding value and fill the result buffer.
  linalg::YieldOp yeildOp =
      *padTensorOp.region().getOps<linalg::YieldOp>().begin();
  Value paddingValue = yeildOp.values()[0];

  auto constOp = paddingValue.getDefiningOp<ConstantOp>();
  if (!constOp) {
    return padTensorOp.emitError(
        "Converting linalg.pad_tensor with non-constant padding value");
  }
  if (constOp.getValue().isa<DenseElementsAttr>()) {
    return padTensorOp.emitError(
        "Converting linalg.pad_tensor with non-scalar constant padding "
        "value");
  }

  b.create<linalg::FillOp>(loc, resultPaddedBuffer, paddingValue);

  // Get the interior region.
  SmallVector<OpFoldResult> sizeMixedValues =
      getMemrefSizes(b, loc, inputMemref);
  SmallVector<OpFoldResult> strides(
      inputMemref.getType().cast<ShapedType>().getRank(),
      b.getI64IntegerAttr(1));

  auto resultSubView = b.create<memref::SubViewOp>(loc, resultPaddedBuffer,
                                                   padTensorOp.getMixedLowPad(),
                                                   sizeMixedValues, strides);
  // Copy to the interior region.
  b.create<linalg::CopyOp>(loc, inputMemref, resultSubView);
  return success();
}

namespace {
/// Pass to convert from tensor based ops to memref based ops.
class LinalgBufferizePass : public LinalgBufferizeBase<LinalgBufferizePass> {
 public:
  LinalgBufferizePass(WorkgroupMemoryAllocationFn fn) : allocationFn(fn) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREEDialect, linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, StandardOpsDialect>();
  }
  void runOnOperation() override;

 private:
  WorkgroupMemoryAllocationFn allocationFn;
};
}  // namespace

void LinalgBufferizePass::runOnOperation() {
  BufferizationPlan plan;
  FuncOp funcOp = getOperation();
  if (failed(analyseOperations(funcOp, plan))) {
    return signalPassFailure();
  }
  if (funcOp
          .walk([&](IREE::Flow::DispatchTensorStoreOp storeOp) -> WalkResult {
            return analyseInterfaceStoreTensorOp(storeOp, plan);
          })
          .wasInterrupted()) {
    return signalPassFailure();
  }

  MLIRContext *context = &getContext();
  OpBuilder b(context);

  BlockAndValueMapping bvm;

  // First go over all hal.interface.binding.subspan ops and create counterparts
  // working with memrefs.
  funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp op) {
    auto shapedType =
        op.getResult().getType().dyn_cast<IREE::Flow::DispatchTensorType>();
    if (!shapedType || !shapedType.hasRank()) return;
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(op);
    // Just change the result type of the InterfaceBindingSubspanOp to form
    // the base buffer.
    auto tensorType =
        op.result().getType().cast<IREE::Flow::DispatchTensorType>();
    auto memRefType = getMemrefTypeForTensor(tensorType);
    auto baseBuffer = b.create<IREE::HAL::InterfaceBindingSubspanOp>(
        op->getLoc(), memRefType, op.binding(), op.byte_offset(),
        op.byte_length());
    bvm.map(op, baseBuffer);
  });

  // Visit all the operations that return `tensor`s and convert them to using
  // `memref`s.
  auto convertTensorProducingOps = [&](Operation *op) -> WalkResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<ConstantOp>([&](ConstantOp constantOp) {
          return convertConstantOp(b, constantOp, bvm);
        })
        .Case<IREE::Flow::DispatchTensorStoreOp>(
            [&](IREE::Flow::DispatchTensorStoreOp storeOp) {
              return convertInterfaceStoreTensorOp(b, storeOp, bvm, plan);
            })
        .Case<IREE::Flow::DispatchTieShapeOp>(
            [&](IREE::Flow::DispatchTieShapeOp shapeOp) {
              return convertDispatchTieShapeOp(b, shapeOp, bvm);
            })
        .Case<scf::ForOp>([&](scf::ForOp forOp) {
          if (failed(getOrAllocateResultBuffers(b, forOp, bvm, plan,
                                                allocationFn))) {
            return failure();
          }
          return convertScfForOp(b, forOp, bvm, plan);
        })
        .Case<scf::IfOp>([&](scf::IfOp ifOp) {
          if (failed(getOrAllocateResultBuffers(b, ifOp, bvm, plan,
                                                allocationFn))) {
            return failure();
          }
          return convertScfIfOp(b, ifOp, bvm, plan);
        })
        .Case<IREE::Flow::DispatchTensorLoadOp, linalg::TensorCollapseShapeOp,
              linalg::TensorExpandShapeOp, SubTensorOp, tensor::CastOp>(
            [&](auto aliasingOp) {
              auto aliasingBuffers =
                  getAliasingBuffersForResults(b, aliasingOp, bvm);
              if (failed(getOrAllocateResultBuffers(b, aliasingOp,
                                                    aliasingBuffers, bvm, plan,
                                                    allocationFn))) {
                return failure();
              }
              copyFromAliasingBufferToResultBuffer(
                  b, aliasingOp->getLoc(), aliasingOp->getOperand(0),
                  aliasingOp->getResult(0), aliasingBuffers, bvm, plan);
              return success();
            })
        .Case<linalg::PadTensorOp>([&](linalg::PadTensorOp padTensorOp) {
          if (failed(getOrAllocateResultBuffers(b, padTensorOp, bvm, plan,
                                                allocationFn))) {
            return failure();
          }
          return convertPadTensorOp(b, padTensorOp, bvm);
        })
        .Case<linalg::LinalgOp>([&](linalg::LinalgOp linalgOp) {
          if (failed(getOrAllocateResultBuffers(b, linalgOp.getOperation(), bvm,
                                                plan, allocationFn))) {
            return failure();
          }
          return convertAnyLinalgOp(b, linalgOp, bvm, plan, allocationFn);
        })
        .Case<SubTensorInsertOp>([&](SubTensorInsertOp subTensorInsertOp) {
          if (failed(getOrAllocateResultBuffers(b, subTensorInsertOp, bvm, plan,
                                                allocationFn))) {
            return failure();
          }
          return convertSubTensorInsertOp(b, subTensorInsertOp, bvm, plan);
        })
        .Case<tensor::InsertOp>([&](tensor::InsertOp tensorInsertOp) {
          if (failed(getOrAllocateResultBuffers(b, tensorInsertOp, bvm, plan,
                                                allocationFn))) {
            return failure();
          }
          return convertTensorInsertOp(b, tensorInsertOp, bvm, plan);
        })
        .Case<vector::TransferWriteOp>(
            [&](vector::TransferWriteOp transferWriteOp) {
              if (failed(getOrAllocateResultBuffers(b, transferWriteOp, bvm,
                                                    plan, allocationFn))) {
                return failure();
              }
              return convertVectorTransferWriteOp(b, transferWriteOp, bvm,
                                                  plan);
            })
        .Default([&](Operation *op) { return success(); });
  };
  auto walkResult =
      funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        b.setInsertionPoint(op);
        return convertTensorProducingOps(op);
      });
  if (walkResult.wasInterrupted()) {
    return signalPassFailure();
  }

  // Lastly visit the non-tensor return operations that still use `tensor`
  // values. These need to be updated to use the corresponding `memref` values,
  // but dont need to update the block-and-value mapping.
  auto convertNonTensorProducingOps = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<tensor::ExtractOp>([&](tensor::ExtractOp op) {
          return convertTensorExtractOp(b, op, bvm);
        })
        .Case<memref::DimOp, vector::TransferReadOp>([&](auto op) {
          for (unsigned i : llvm::seq<unsigned>(0, op->getNumOperands())) {
            Value operand = op->getOperand(i);
            if (operand.getType().isa<RankedTensorType>()) {
              Value remappedVal = bvm.lookupOrNull(operand);
              if (remappedVal) op->setOperand(i, remappedVal);
            }
          }
          return success();
        })
        .Case<scf::ForOp>([&](scf::ForOp forOp) {
          // To canonicalize the `scf.for` tensor result/operand/yield value
          // away, forward the init argument to the yeild of the loop.
          auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
          for (auto arg : llvm::enumerate(forOp.getIterOperands())) {
            if (!arg.value().getType().isa<RankedTensorType>()) continue;
            yieldOp.setOperand(arg.index(), arg.value());
          }
          return success();
        })
        .Case<IREE::Flow::DispatchTensorStoreOp>([&](auto op) {
          op.erase();
          return success();
        })
        .Default([&](Operation *op) { return success(); });
  };

  walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
    b.setInsertionPoint(op);
    return convertNonTensorProducingOps(op);
  });
  if (walkResult.wasInterrupted()) {
    return signalPassFailure();
  }
}

static Value defaultAllocationFn(OpBuilder &builder, Location loc,
                                 ArrayRef<int64_t> staticShape,
                                 Type elementType,
                                 ArrayRef<Value> dynamicSizes) {
  auto allocationType = MemRefType::get(staticShape, elementType);
  return builder.create<memref::AllocOp>(loc, allocationType, dynamicSizes);
}

std::unique_ptr<OperationPass<FuncOp>> createLinalgBufferizePass(
    WorkgroupMemoryAllocationFn allocationFn) {
  return std::make_unique<LinalgBufferizePass>(
      allocationFn ? allocationFn : defaultAllocationFn);
}

void addLinalgBufferizePasses(OpPassManager &passManager,
                              WorkgroupMemoryAllocationFn allocationFn) {
  passManager.addNestedPass<FuncOp>(createLinalgBufferizePass(allocationFn));
  passManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCleanupBufferAllocViewPass());
  // passManager.addPass(createBufferHoistingPass());
  // TODO(nicolasvasilache): bug in buffer loop hoisting with
  // dynamic_linalg_matmul_on_tensors_fuse_0.mlir
  // passManager.addPass(createBufferLoopHoistingPass());
}

}  // namespace iree_compiler
}  // namespace mlir
