// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/Analysis/GlobalTable.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::iree_compiler::IREE::Flow {

#define DEBUG_TYPE "iree-flow-replicate-globals-per-affinity"

#define GEN_PASS_DEF_REPLICATEGLOBALSPERAFFINITYPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {
// Helper class to manage the creation of operations per affinity.
// Each global op is cloned per affinity it is used with. The new global op
// will be initialized in the initializer region to hold a copy of the original
// global op's value, but transferred to the target affinity.
// When an op operand is requested for a specific affinity it will be cloned,
// unless the operand already matches the affinity. Operations will be cloned
// as needed to wrap the new operands. Global load ops will be cloned to load
// from the new global op for the requested affinity.
class ValuePerAffinityHelper {
public:
  explicit ValuePerAffinityHelper(mlir::ModuleOp moduleOp, bool useTransfers);
  ~ValuePerAffinityHelper() = default;

  using OpAffinityPair = std::tuple<Operation *, IREE::Stream::AffinityAttr>;
  using ValueAffinityPair = std::tuple<Value, IREE::Stream::AffinityAttr>;

  // Returns new or cached value for the given affinity.
  Value getOrCreateValueForAffinity(OpOperand *opOperand,
                                    IREE::Stream::AffinityAttr affinityAttr);

  // Returns new or existing global op for the given affinity. It loads the data
  // from the original global op and transfers it to the target affinity in a
  // new initializer op.
  IREE::Util::GlobalOpInterface
  getOrCreateGlobalForAffinity(StringRef globalName,
                               IREE::Stream::AffinityAttr affinityAttr);

private:
  // Returns new global load op for the given affinity. If the source global op
  // does not match the affinity, a new global op will be created.
  Value
  getOrCreateGlobalLoadForAffinity(IREE::Util::GlobalLoadOpInterface loadOp,
                                   IREE::Stream::AffinityAttr affinityAttr);

  // Returns new or cached affinity op for the given affinity.
  Value
  getOrCreateAffinityOpForAffinity(IREE::Stream::AffinityOpInterface affinityOp,
                                   OpResult opResult,
                                   IREE::Stream::AffinityAttr affinityAttr);

  OpBuilder builder;
  SymbolTable symbolTable;
  bool useTransfers;
  DenseMap<OpAffinityPair, IREE::Util::GlobalOpInterface> cachedGlobals;
  DenseMap<ValueAffinityPair, Value> cachedValuePerAffinity;

  // Cache the insertion point (last initializer or the global itself) for each
  // global for performance.
  DenseMap<Operation *, Operation *> cachedInsertionPointForGlobal;
};

ValuePerAffinityHelper::ValuePerAffinityHelper(mlir::ModuleOp moduleOp,
                                               bool useTransfers)
    : builder(moduleOp), symbolTable(moduleOp), useTransfers(useTransfers) {
  // If we are using transfers, we don't need to pre-compute the insertion
  // points to replicate globals.
  if (useTransfers) {
    return;
  }
  // Pre-compute the insertion point for each global for performance.
  // This avoids scanning all initializers multiple times during transformation.
  IREE::Util::GlobalTable globalTable(moduleOp);
  globalTable.rebuild();
  globalTable.forEach([&](IREE::Util::Global &global) {
    // Initialize with the global op itself as the default insertion point.
    Operation *insertionPoint = global.op.getOperation();

    // Iterate through store operations to find initializers.
    for (auto storeOp : global.storeOps) {
      // Get the parent initializer op if the store is within one.
      auto initOp = storeOp->getParentOfType<IREE::Util::InitializerOp>();
      if (!initOp) {
        continue;
      }

      // Update the insertion point if this is the latest initializer.
      if (insertionPoint->isBeforeInBlock(initOp)) {
        insertionPoint = initOp;
      }
    }

    cachedInsertionPointForGlobal[global.op.getOperation()] = insertionPoint;
    return IREE::Util::GlobalAction::PRESERVE;
  });
}

Value ValuePerAffinityHelper::getOrCreateValueForAffinity(
    OpOperand *opOperand, IREE::Stream::AffinityAttr affinityAttr) {
  ValueAffinityPair key = {opOperand->get(), affinityAttr};
  if (cachedValuePerAffinity.contains(key)) {
    return cachedValuePerAffinity.lookup(key);
  }

  return TypeSwitch<Operation *, Value>(opOperand->get().getDefiningOp())
      .Case<IREE::Util::GlobalLoadOpInterface>([&](auto loadOp) {
        return getOrCreateGlobalLoadForAffinity(loadOp, affinityAttr);
      })
      .Case<IREE::Stream::AffinityOpInterface>([&](auto affinityOp) {
        return getOrCreateAffinityOpForAffinity(
            affinityOp, cast<OpResult>(opOperand->get()), affinityAttr);
      })
      .Case<IREE::Flow::TensorReshapeOp>([&](auto reshapeOp) {
        builder.setInsertionPoint(reshapeOp);
        Value source = getOrCreateValueForAffinity(
            &reshapeOp.getSourceMutable(), affinityAttr);
        cachedValuePerAffinity[key] = IREE::Flow::TensorReshapeOp::create(
            builder, reshapeOp.getLoc(), reshapeOp.getResult().getType(),
            source, reshapeOp.getSourceDims(), reshapeOp.getResultDims());
        return cachedValuePerAffinity[key];
      })
      .Default([&](Operation *op) {
        LDBG() << "unsupported op in GlobalPerAffinityHelper: " << *op;
        assert(false && "unsupported op");
        return Value();
      });
}

static std::string getNewGlobalName(StringRef originalName,
                                    IREE::Stream::AffinityAttr affinityAttr) {
  std::string resul = originalName.str();
  llvm::raw_string_ostream sstream(resul);
  sstream << "_" << affinityAttr;
  return resul;
}

IREE::Util::GlobalOpInterface
ValuePerAffinityHelper::getOrCreateGlobalForAffinity(
    StringRef globalName, IREE::Stream::AffinityAttr affinityAttr) {
  auto globalOp = symbolTable.lookup<IREE::Util::GlobalOpInterface>(globalName);
  auto affinityOp = dyn_cast_if_present<IREE::Stream::AffinityOpInterface>(
      globalOp.getOperation());
  if (affinityOp.getAffinityAttr() == affinityAttr) {
    return globalOp;
  }

  // If we are using transfers, we don't need to replicate the global.
  if (useTransfers) {
    return globalOp;
  }

  OpAffinityPair key = {globalOp, affinityAttr};
  if (cachedGlobals.contains(key)) {
    return cachedGlobals.lookup(key);
  }

  // Find the insertion point: after the last initializer that references this
  // global, or after the global itself if no initializers exist.
  // The cache was already populated in the constructor.
  Operation *insertionPoint =
      cachedInsertionPointForGlobal.lookup(globalOp.getOperation());
  assert(insertionPoint && "failed to find insertion point for global");

  // Create the new global and initializer after the insertion point.
  Location loc = globalOp.getLoc();
  builder.setInsertionPointAfter(insertionPoint);
  std::string newGlobalName = getNewGlobalName(globalName, affinityAttr);
  auto newGlobalOp = IREE::Util::GlobalOp::create(builder, loc, newGlobalName,
                                                  /*isMutable=*/false,
                                                  globalOp.getGlobalType());
  newGlobalOp.setPrivate();
  symbolTable.insert(newGlobalOp);
  auto initializerOp = IREE::Util::InitializerOp::create(builder, loc);

  // Create transfer op and store op in the initializer.
  builder.setInsertionPointToStart(initializerOp.addEntryBlock());
  Value loadedValue =
      globalOp.createLoadOp(loc, builder).getLoadedGlobalValue();
  Value transferOp = IREE::Flow::TensorTransferOp::create(
      builder, loc, loadedValue, affinityAttr);
  IREE::Util::GlobalStoreOp::create(builder, loc, transferOp, newGlobalOp);
  IREE::Util::ReturnOp::create(builder, loc);

  cachedGlobals[key] = newGlobalOp;
  return newGlobalOp;
}

Value ValuePerAffinityHelper::getOrCreateGlobalLoadForAffinity(
    IREE::Util::GlobalLoadOpInterface loadOp,
    IREE::Stream::AffinityAttr affinityAttr) {
  IREE::Util::GlobalOpInterface globalOp =
      getOrCreateGlobalForAffinity(loadOp.getGlobalName(), affinityAttr);
  ValueAffinityPair key = {loadOp.getLoadedGlobalValue(), affinityAttr};
  if (useTransfers) {
    builder.setInsertionPointAfter(loadOp);
    Value loadedValue = loadOp.getLoadedGlobalValue();
    Value transferOp = IREE::Flow::TensorTransferOp::create(
        builder, loadOp->getLoc(), loadedValue, affinityAttr);
    cachedValuePerAffinity[key] = transferOp;
    return transferOp;
  }
  builder.setInsertionPoint(loadOp);
  auto newLoadOp =
      IREE::Util::GlobalLoadOp::create(builder, loadOp.getLoc(), globalOp);
  newLoadOp.setIsImmutable(true);
  cachedValuePerAffinity[key] = newLoadOp.getLoadedGlobalValue();
  return newLoadOp.getLoadedGlobalValue();
}

Value ValuePerAffinityHelper::getOrCreateAffinityOpForAffinity(
    IREE::Stream::AffinityOpInterface affinityOp, OpResult opResult,
    IREE::Stream::AffinityAttr affinityAttr) {
  ValueAffinityPair key = {opResult, affinityAttr};
  if (affinityOp.getAffinityAttr() == affinityAttr) {
    return opResult;
  }
  SmallVector<Value> newOperands;
  for (OpOperand &operand : affinityOp->getOpOperands()) {
    if (!isa<IREE::Stream::AffinityTypeInterface>(operand.get().getType())) {
      newOperands.push_back(operand.get());
      continue;
    }
    newOperands.push_back(getOrCreateValueForAffinity(&operand, affinityAttr));
  }
  builder.setInsertionPoint(affinityOp);
  Operation *newAffinityOp =
      clone(builder, affinityOp, affinityOp->getResultTypes(), newOperands);
  newAffinityOp->setDiscardableAttrs(
      affinityOp->getDiscardableAttrDictionary());
  // Cache all the results, so it won't create new operations when querying for
  // other results that have the same affinity.
  for (auto [oldResult, newResult] :
       llvm::zip_equal(affinityOp->getResults(), newAffinityOp->getResults())) {
    ValueAffinityPair resultKey = {oldResult, affinityAttr};
    cachedValuePerAffinity[resultKey] = newResult;
  }
  return cachedValuePerAffinity[key];
}

// Tracks which operands of an operation are avaialble for affinity analysis.
// Some operands may come from globals that do not have specified affinity. It
// can be used to prioritize the operand affinities for the operation.
class OpOperandAffinityState {
public:
  OpOperandAffinityState() {}
  explicit OpOperandAffinityState(Operation *op) {
    availableOperands.resize(op->getNumOperands(), true);
    for (auto &operand : op->getOpOperands()) {
      if (!isa<IREE::Stream::AffinityTypeInterface>(operand.get().getType())) {
        availableOperands.reset(operand.getOperandNumber());
      }
    }
  }
  ~OpOperandAffinityState() = default;

  void setUnavailableOperand(int idx) { availableOperands.reset(idx); }
  bool isAvailableOperand(int idx) const { return availableOperands.test(idx); }
  bool hasAnyAvailableOperand() const { return availableOperands.any(); }

private:
  llvm::BitVector availableOperands;
};

// Manages the operand affinity state of operations. The op state is initialized
// when it is first accessed.
class OpOperandAffinityStateMap {
public:
  OpOperandAffinityStateMap() = default;
  ~OpOperandAffinityStateMap() = default;

  OpOperandAffinityState &operator[](Operation *op) {
    return operandAvailableStateMap[op];
  }

  // Marks the operand of the given op as not having affinity information.
  void setUnavailableOperand(Operation *op, int idx) {
    if (!operandAvailableStateMap.contains(op)) {
      operandAvailableStateMap[op] = OpOperandAffinityState(op);
    }
    operandAvailableStateMap[op].setUnavailableOperand(idx);
  }

  // Returns true if the operand of the given op is known to have affinity.
  bool isAvailableOperand(Operation *op, int idx) {
    // Still initialize the state because the op might not have any
    // IREE::Stream::AffinityTypeInterface operand type.
    if (!operandAvailableStateMap.contains(op)) {
      operandAvailableStateMap.insert({op, OpOperandAffinityState(op)});
    }
    return operandAvailableStateMap[op].isAvailableOperand(idx);
  }

  // Returns true if the op can get affinity information from its operands.
  bool hasAnyAvailableOperand(Operation *op) {
    if (!operandAvailableStateMap.contains(op)) {
      operandAvailableStateMap.insert({op, OpOperandAffinityState(op)});
    }
    return operandAvailableStateMap[op].hasAnyAvailableOperand();
  }

  // Dumps the operand affinity state of the given op.
  void dump(Operation *op, llvm::raw_ostream &os) {
    if (!operandAvailableStateMap.contains(op)) {
      os << "no state\n";
      return;
    }
    auto &state = operandAvailableStateMap[op];
    os << "known operand affinity: ";
    for (int i = 0; i < op->getNumOperands(); ++i) {
      if (state.isAvailableOperand(i)) {
        os << i << " ";
      }
    }
    os << "\n";
  }

private:
  DenseMap<Operation *, OpOperandAffinityState> operandAvailableStateMap;
};

struct ReplicateGlobalsPerAffinityPass
    : public impl::ReplicateGlobalsPerAffinityPassBase<
          ReplicateGlobalsPerAffinityPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

// Dumps the operand affinities and result usage affinities of the given op.
static void
dumpOpAffinityStatus(IREE::Stream::AffinityOpInterface affinityOp,
                     ArrayRef<IREE::Stream::AffinityAttr> affinityAttrs,
                     IREE::Stream::AffinityAnalysis &affinityAnalysis,
                     OpOperandAffinityStateMap &opOperandAffinityStateMap) {
  LDBG_OS([&](raw_ostream &os) {
    os << "multiple/empty affinities: [";
    llvm::interleaveComma(affinityAttrs, os);
    os << "]";
  });
  LDBG() << " where the corresponding globals are:";
  for (OpOperand &opOperand : affinityOp->getOpOperands()) {
    int idx = opOperand.getOperandNumber();
    Value value = opOperand.get();
    if (!isa<IREE::Stream::AffinityTypeInterface>(value.getType())) {
      LDBG() << "\toperand #" << idx << " type: " << value.getType();
      continue;
    }
    if (!opOperandAffinityStateMap.isAvailableOperand(affinityOp, idx)) {
      LDBG() << "\toperand #" << idx << " affinity: ignored";
      continue;
    }
    auto attr = affinityAnalysis.lookupResourceAffinity(value);
    LDBG() << "\toperand #" << idx << " affinity: " << attr;
  }
  for (auto result : affinityOp->getResults()) {
    SmallVector<IREE::Stream::AffinityAttr> usageAffinities;
    if (affinityAnalysis.tryLookupResourceUsageAffinity(result,
                                                        usageAffinities)) {
      LDBG_OS([&](raw_ostream &os) {
        os << "\tresult type: " << result.getType() << " affinities: [";
        llvm::interleaveComma(usageAffinities, os);
        os << "]";
      });
    } else {
      LDBG_OS([&](raw_ostream &os) {
        os << "\tresult type: " << result.getType() << " affinities: failed";
      });
    }
  }
}

void ReplicateGlobalsPerAffinityPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  IREE::Stream::AffinityAnalysis affinityAnalysis(moduleOp);
  if (failed(affinityAnalysis.run())) {
    LDBG() << "failed on running affinity analysis";
    return;
  }

  Explorer explorer(moduleOp.getOperation(), TraversalAction::RECURSE);
  explorer.setOpInterfaceAction<mlir::FunctionOpInterface>(
      TraversalAction::RECURSE);
  explorer.setDialectAction<mlir::scf::SCFDialect>(TraversalAction::RECURSE);
  explorer.setDialectAction<IREE::Flow::FlowDialect>(TraversalAction::RECURSE);
  explorer.setDialectAction<IREE::Util::UtilDialect>(TraversalAction::RECURSE);
  explorer.initialize();

  SetVector<Operation *> worklist;
  explorer.forEachGlobal([&](const Explorer::GlobalInfo *globalInfo) {
    if (globalInfo->isIndirect || globalInfo->op.isGlobalMutable()) {
      return;
    }
    // Scalars are expected to be used on host, so we do not need to replicate
    // them.
    if (globalInfo->op.getGlobalType().isIntOrIndexOrFloat()) {
      return;
    }
    worklist.insert(globalInfo->op);
  });

  // List of operands that the source is from global ops.
  //   [Operation, [(operand number, maybe global op)]].
  // If the global op is null, it means that the operand is not directly from a
  // load op. It is important to track the global op, so we can support cross
  // functions/branches cases.
  DenseMap<Operation *,
           SmallVector<std::tuple<OpOperand *, IREE::Util::GlobalOpInterface>>>
      opToGlobalUseMap;
  OpOperandAffinityStateMap opOperandAffinityStateMap;
  SymbolTable symbolTable(moduleOp);
  while (!worklist.empty()) {
    Operation *currentOp = *worklist.begin();
    LDBG() << "processing op: " << *currentOp;
    worklist.erase(worklist.begin());
    TypeSwitch<Operation *>(currentOp)
        .Case<IREE::Util::GlobalOpInterface>([&](auto globalOp) {
          const Explorer::GlobalInfo *globalInfo =
              explorer.getGlobalInfo(globalOp);
          for (auto loadOp : globalInfo->getLoads()) {
            worklist.insert(loadOp);
          }
        })
        .Case<IREE::Util::GlobalLoadOpInterface>([&](auto loadOp) {
          explorer.walkTransitiveUses(
              loadOp.getLoadedGlobalValue(), [&](OpOperand &operand) {
                if (isa<IREE::Util::InitializerOp>(
                        operand.getOwner()->getParentOp())) {
                  return WalkResult::advance();
                }
                LDBG() << "\tuse: " << *operand.getOwner();
                Operation *consumerOp = operand.getOwner();
                auto globalOp =
                    symbolTable.lookup<IREE::Util::GlobalOpInterface>(
                        loadOp.getGlobalName());
                opToGlobalUseMap[consumerOp].push_back({&operand, globalOp});

                // Do not query from affinityAnalysis here because it could
                // select the default affinity, while we want to prioritize
                // the use and specialize the global op per affinity.
                auto affinityAttr = cast<IREE::Stream::AffinityOpInterface>(
                                        globalOp.getOperation())
                                        .getAffinityAttr();
                if (!affinityAttr) {
                  opOperandAffinityStateMap.setUnavailableOperand(
                      consumerOp, operand.getOperandNumber());
                }
                if (!opOperandAffinityStateMap.hasAnyAvailableOperand(
                        consumerOp)) {
                  worklist.insert(consumerOp);
                }
                return WalkResult::advance();
              });
        })
        .Case<IREE::Stream::AffinityOpInterface, IREE::Flow::TensorReshapeOp>(
            [&](auto affinityOp) {
              for (OpResult result : affinityOp->getResults()) {
                if (!isa<IREE::Stream::AffinityTypeInterface>(
                        result.getType())) {
                  continue;
                }
                explorer.walkTransitiveUses(result, [&](OpOperand &operand) {
                  LDBG() << "\tuse: " << *operand.getOwner();
                  Operation *consumerOp = operand.getOwner();
                  opOperandAffinityStateMap.setUnavailableOperand(
                      consumerOp, operand.getOperandNumber());
                  opToGlobalUseMap[consumerOp].push_back({&operand, {}});
                  if (!opOperandAffinityStateMap.hasAnyAvailableOperand(
                          consumerOp)) {
                    worklist.insert(consumerOp);
                  }
                  return WalkResult::advance();
                });
              }
            })
        .Default([&](Operation *) {});
  }

  // We have to update the operands in topological order, because they may
  // depend on each other. It can lead to ambigious affinity if an op queries
  // the affinity from unresolved operations. In this context, it can return
  // multiple affinities and fail to replicate the global for its uses.
  SetVector<Operation *> sortedAffinityOps(opToGlobalUseMap.keys().begin(),
                                           opToGlobalUseMap.keys().end());
  sortedAffinityOps = mlir::topologicalSort(sortedAffinityOps);
  ValuePerAffinityHelper globalPerAffinityHelper(moduleOp, useTransfers);

  IRRewriter rewriter(&getContext());
  for (auto operation : sortedAffinityOps) {
    auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(operation);
    if (!affinityOp) {
      LDBG() << "skipping non-affinity op: " << *operation;
      continue;
    }
    LDBG() << "updating: " << affinityOp;
    LDBG_OS([&](raw_ostream &os) {
      os << "\tavailable operand state: ";
      opOperandAffinityStateMap.dump(affinityOp, os);
    });

    // All the operands are from globals. It needs to be updated by consumers.
    if (!opOperandAffinityStateMap.hasAnyAvailableOperand(affinityOp)) {
      continue;
    }

    llvm::SmallSetVector<Stream::AffinityAttr, 4> affinityAttrs;
    for (auto [idx, operand] : llvm::enumerate(affinityOp->getOperands())) {
      if (!opOperandAffinityStateMap.isAvailableOperand(affinityOp, idx)) {
        continue;
      }
      if (auto attr = affinityAnalysis.lookupResourceAffinity(operand)) {
        affinityAttrs.insert(attr);
      } else {
        LDBG() << "failed to get affinity from operand #" << idx
               << " of op: " << operand;
      }
    }

    // It is very bad if it happens. Something might be wrong in the input
    // program or something is missing in the implementation.
    if (affinityAttrs.size() != 1) {
      LDBG() << "skipping affinity op: " << affinityOp;
      dumpOpAffinityStatus(affinityOp, affinityAttrs.getArrayRef(),
                           affinityAnalysis, opOperandAffinityStateMap);
      continue;
    }

    IREE::Stream::AffinityAttr executionAffinityAttr = *affinityAttrs.begin();
    LDBG() << "updating operands with the affinity: " << executionAffinityAttr;

    // Order matters here because it is wrong if an operand is already updated
    // with a new value. E.g.,
    //   %0 = flow.dispatch @dispatch0(%global)
    //   %1 = flow.reshape %0 ...
    // If we update %0 first, then when updating %1 we will see a new global
    // from %0 which uses the new global, that is not cached in
    // `globalPerAffinityHelper`.
    // Thus, we collect all the updates first, then apply them.
    SmallVector<std::pair<OpOperand *, Value>> updateList;
    for (auto [operand, maybeGlobalOp] : opToGlobalUseMap[affinityOp]) {
      if (maybeGlobalOp) {
        rewriter.setInsertionPoint(affinityOp);
        IREE::Util::GlobalOpInterface newGlobalOp =
            globalPerAffinityHelper.getOrCreateGlobalForAffinity(
                maybeGlobalOp.getGlobalName(), executionAffinityAttr);
        auto newLoadOp = IREE::Util::GlobalLoadOp::create(
            rewriter, affinityOp.getLoc(), newGlobalOp);
        newLoadOp.setIsImmutable(true);
        if (useTransfers) {
          // If transfers are used instead of replicating globals, the global
          // operation returned from 'getOrCreateGlobalForAffinity' above will
          // be the same as the original global and we need to transfer to the
          // right device after loading.
          Value transferOp = IREE::Flow::TensorTransferOp::create(
              rewriter, affinityOp->getLoc(), newLoadOp.getLoadedGlobalValue(),
              executionAffinityAttr);
          updateList.push_back({operand, transferOp});
        } else {
          updateList.push_back({operand, newLoadOp.getLoadedGlobalValue()});
        }
      } else {
        updateList.push_back(
            {operand, globalPerAffinityHelper.getOrCreateValueForAffinity(
                          operand, executionAffinityAttr)});
      }
    }
    for (auto [opOperand, newValue] : updateList) {
      opOperand->assign(newValue);
    }
  }
}

} // namespace mlir::iree_compiler::IREE::Flow
