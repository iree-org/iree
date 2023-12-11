// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-annotate-dispatch-arguments"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ANNOTATEDISPATCHARGUMENTSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Analysis state
//===----------------------------------------------------------------------===//

// TODO(benvanik): move to Util/Analysis/ as this would be useful in other
// passes as well and only depends on util.align and upstream ops.

static std::string
getPVSAsStr(const DFX::PotentialConstantIntValuesState &pvs) {
  std::string str;
  llvm::raw_string_ostream sstream(str);
  sstream << "pvs: ";
  if (pvs.isValidState()) {
    sstream << "[";
    if (pvs.isUndefContained()) {
      sstream << "undef, ";
    }
    llvm::interleaveComma(pvs.getAssumedSet(), sstream,
                          [&](APInt value) { value.print(sstream, false); });
    sstream << "]";
  } else {
    sstream << "(invalid)";
  }
  sstream.flush();
  return str;
}

static llvm::MaybeAlign commonAlignment(llvm::MaybeAlign A,
                                        llvm::MaybeAlign B) {
  return A && B ? std::min(*A, *B) : A ? A : B;
}

class GlobalPVS : public DFX::StateWrapper<
                      DFX::PotentialConstantIntValuesState,
                      DFX::TypedOperationElement<IREE::Util::GlobalOp>> {
public:
  using BaseType =
      DFX::StateWrapper<DFX::PotentialConstantIntValuesState,
                        DFX::TypedOperationElement<IREE::Util::GlobalOp>>;

  static GlobalPVS &createForPosition(const Position &pos,
                                      DFX::Solver &solver) {
    return *(new (solver.getAllocator()) GlobalPVS(pos));
  }

  const std::string getName() const override { return "GlobalPVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return getPVSAsStr(getState());
  }

private:
  explicit GlobalPVS(const Position &pos) : BaseType(pos) {}

  void initializeOperation(IREE::Util::GlobalOp globalOp,
                           DFX::Solver &solver) override;
  ChangeStatus updateOperation(IREE::Util::GlobalOp globalOp,
                               DFX::Solver &solver) override;

  friend class DFX::Solver;
};
const char GlobalPVS::ID = 0;

class ValuePVS : public DFX::StateWrapper<DFX::PotentialConstantIntValuesState,
                                          DFX::ValueElement> {
public:
  using BaseType = DFX::StateWrapper<DFX::PotentialConstantIntValuesState,
                                     DFX::ValueElement>;

  static ValuePVS &createForPosition(const Position &pos, DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ValuePVS(pos));
  }

  const std::string getName() const override { return "ValuePVS"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return getPVSAsStr(getState());
  }

private:
  explicit ValuePVS(const Position &pos) : BaseType(pos) {}

  void initializeValue(Value value, DFX::Solver &solver) override {
    APInt staticValue;
    if (matchPattern(value, m_ConstantInt(&staticValue))) {
      unionAssumed(staticValue);
      indicateOptimisticFixpoint();
    }
  }

  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    StateType newState;
    if (solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
          APInt staticValue;
          if (matchPattern(result, m_ConstantInt(&staticValue))) {
            newState.unionAssumed(staticValue);
            return WalkResult::advance();
          }

          if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadOpInterface>(
                  result.getDefiningOp())) {
            auto *globalInfo = solver.getExplorer().queryGlobalInfoFrom(
                loadOp.getGlobalName(), loadOp);
            auto global = solver.getElementFor<GlobalPVS>(
                *this, Position::forOperation(globalInfo->op),
                DFX::Resolution::REQUIRED);
            if (global.isValidState()) {
              newState.unionAssumed(global);
              return WalkResult::advance();
            }
          }

          // TODO(benvanik): more ops supported for joining. We could for
          // example walk the lhs/rhs of elementwise ops and perform the set
          // operations (so addi %lhs, %rhs could produce a PVS of all of %lhs
          // summed to all of %rhs). May not be worth it, though.
          // TODO(benvanik): move select op walking to the explorer.
          if (auto selectOp =
                  dyn_cast<mlir::arith::SelectOp>(result.getDefiningOp())) {
            auto lhs = solver.getElementFor<ValuePVS>(
                *this, Position::forValue(selectOp.getTrueValue()),
                DFX::Resolution::REQUIRED);
            auto rhs = solver.getElementFor<ValuePVS>(
                *this, Position::forValue(selectOp.getFalseValue()),
                DFX::Resolution::REQUIRED);
            if (!lhs.isValidState() || !rhs.isValidState()) {
              newState.unionAssumedWithUndef();
              newState.indicatePessimisticFixpoint();
            } else {
              newState.unionAssumed(lhs);
              newState.unionAssumed(rhs);
            }
            return WalkResult::advance();
          }

          // Some other dynamic value we can't analyze (yet).
          newState.unionAssumedWithUndef();
          newState.indicatePessimisticFixpoint();
          return WalkResult::advance();
        }) == TraversalResult::INCOMPLETE) {
      newState.unionAssumedWithUndef();
      newState.indicatePessimisticFixpoint();
    }
    return DFX::clampStateAndIndicateChange(getState(), newState);
  }

  friend class DFX::Solver;
};
const char ValuePVS::ID = 0;

void GlobalPVS::initializeOperation(IREE::Util::GlobalOp globalOp,
                                    DFX::Solver &solver) {
  auto *globalInfo = solver.getExplorer().getGlobalInfo(globalOp);
  if (!globalInfo || globalInfo->isIndirect) {
    // Cannot perform analysis.
    indicatePessimisticFixpoint();
  } else if (globalInfo) {
    if (auto initialValue = llvm::dyn_cast_if_present<IntegerAttr>(
            globalOp.getInitialValueAttr())) {
      // Initial value is available for use; stored values from the rest of the
      // program will come during iteration.
      unionAssumed(initialValue.getValue());
    }
  }
}

ChangeStatus GlobalPVS::updateOperation(IREE::Util::GlobalOp globalOp,
                                        DFX::Solver &solver) {
  StateType newState;
  auto *globalInfo = solver.getExplorer().getGlobalInfo(globalOp);
  for (auto use : globalInfo->uses) {
    auto storeOp = dyn_cast<IREE::Util::GlobalStoreOpInterface>(use);
    if (!storeOp)
      continue;
    auto value = solver.getElementFor<ValuePVS>(
        *this, Position::forValue(storeOp.getStoredGlobalValue()),
        DFX::Resolution::REQUIRED);
    if (value.isValidState()) {
      newState.unionAssumed(value);
    } else {
      newState.unionAssumedWithUndef();
      newState.indicatePessimisticFixpoint();
    }
  }
  return DFX::clampStateAndIndicateChange(getState(), newState);
}

static constexpr uint64_t kMaximumAlignment = 1ull << 32;

using AlignmentStateType = DFX::IncIntegerState<uint64_t, kMaximumAlignment, 1>;
class ValueAlignment
    : public DFX::StateWrapper<AlignmentStateType, DFX::ValueElement> {
public:
  using BaseType = DFX::StateWrapper<AlignmentStateType, DFX::ValueElement>;

  static ValueAlignment &createForPosition(const Position &pos,
                                           DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ValueAlignment(pos));
  }

  llvm::MaybeAlign getAssumedAlignment() const {
    return llvm::MaybeAlign(getAssumed());
  }

  llvm::MaybeAlign getKnownAlignment() const {
    return llvm::MaybeAlign(getKnown());
  }

  const std::string getName() const override { return "ValueAlignment"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }
  static const char ID;

  const std::string getAsStr(AsmState &asmState) const override {
    return std::string("alignment: ") +
           std::to_string(getAssumedAlignment().valueOrOne().value());
  }

private:
  explicit ValueAlignment(const Position &pos) : BaseType(pos) {}

  void initializeValue(Value value, DFX::Solver &solver) override {
    if (!value.getType().isIndex()) {
      indicatePessimisticFixpoint();
      return;
    }
  }

  static llvm::MaybeAlign computeAlignment(const ValuePVS::SetTy &set) {
    if (set.empty())
      return llvm::MaybeAlign();
    llvm::MaybeAlign alignment;
    for (auto value : set) {
      APInt valueDivisor = (value & (~(value - 1)));
      alignment = commonAlignment(
          alignment, llvm::MaybeAlign(valueDivisor.getZExtValue()));
    }
    return alignment;
  }

  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    StateType newState = getState();

    // If we can get a full potential value set then we can derive an alignment
    // from that.
    auto pvs = solver.getElementFor<ValuePVS>(*this, Position::forValue(value),
                                              DFX::Resolution::OPTIONAL);
    if (pvs.isValidState() && !pvs.isUndefContained()) {
      auto alignment = computeAlignment(pvs.getAssumedSet());
      if (alignment.has_value()) {
        newState.takeAssumedMinimum(alignment.valueOrOne().value());
        newState.indicateOptimisticFixpoint();
      }
    }

    if (!newState.isAtFixpoint()) {
      // Scan IR to see if we can infer the alignment.
      // TODO(benvanik): walk math ops (like muli) to peek through to alignments
      // of inputs. For now we just look for util.align only. We should also
      // be able to look through casts/exts/etc and affine.apply.
      if (solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
            if (auto alignOp =
                    dyn_cast<IREE::Util::AlignOp>(result.getDefiningOp())) {
              auto alignment = solver.getElementFor<ValueAlignment>(
                  *this, Position::forValue(alignOp.getAlignment()),
                  DFX::Resolution::REQUIRED);
              newState ^= alignment;
            }
            return WalkResult::advance();
          }) == TraversalResult::INCOMPLETE) {
        newState.indicatePessimisticFixpoint();
      }
    }

    return DFX::clampStateAndIndicateChange(getState(), newState);
  }

  friend class DFX::Solver;
};
const char ValueAlignment::ID = 0;

class ArgumentAnalysis {
public:
  explicit ArgumentAnalysis(Operation *rootOp)
      : explorer(rootOp, TraversalAction::SHALLOW),
        solver(explorer, allocator) {
    explorer.setOpAction<IREE::Util::InitializerOp>(TraversalAction::RECURSE);
    explorer.setOpAction<mlir::func::FuncOp>(TraversalAction::RECURSE);
    explorer.setDialectAction<IREE::Stream::StreamDialect>(
        TraversalAction::RECURSE);
    // Ignore the contents of executables (linalg goo, etc).
    explorer.setOpAction<IREE::Stream::ExecutableOp>(TraversalAction::IGNORE);
    explorer.initialize();

    // Find all dispatches and bucket by their target entry point.
    rootOp->walk([&](IREE::Stream::CmdDispatchOp dispatchOp) {
      dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPointAttr) {
        auto exportOp = explorer.getSymbolTables().lookupNearestSymbolFrom(
            dispatchOp, entryPointAttr);
        entryDispatchMap[exportOp].push_back(dispatchOp);
      });
    });
  }

  // Runs analysis and populates the state cache.
  // May fail if analysis cannot be completed due to unsupported or unknown IR.
  LogicalResult run() {
    // Seed all dispatch arguments we want to analyze.
    for (auto it : entryDispatchMap) {
      for (auto dispatchOp : it.second) {
        for (auto operand : dispatchOp.getUniformOperands()) {
          solver.getOrCreateElementFor<ValuePVS>(Position::forValue(operand));
          solver.getOrCreateElementFor<ValueAlignment>(
              Position::forValue(operand));
        }
        for (auto resourceOffset : dispatchOp.getResourceOffsets()) {
          solver.getOrCreateElementFor<ValueAlignment>(
              Position::forValue(resourceOffset));
        }
      }
    }

    // Run solver to completion.
    return solver.run();
  }

  // Returns a list of dispatch sites in arbitrary order to the given
  // |exportOp|.
  ArrayRef<IREE::Stream::CmdDispatchOp>
  getDispatchSites(IREE::Stream::ExecutableExportOp exportOp) {
    auto it = entryDispatchMap.find(exportOp);
    if (it == entryDispatchMap.end())
      return {};
    return it->second;
  }

  // Returns the minimum alignment for the given |value| or None if it could not
  // be analyzed and natural alignment should be assumed.
  llvm::MaybeAlign getAlignmentFor(Value value) {
    auto element =
        solver.lookupElementFor<ValueAlignment>(Position::forValue(value));
    if (!element)
      return llvm::MaybeAlign();
    return element->getAssumedAlignment();
  }

  // TODO(benvanik): replace these with dedicated
  // ArgumentAlignment/ResourceOffsetAlignment state that does this unioning as
  // part of the solver. It's not strictly required as this is unidirectional
  // (the alignment of the export arguments is dictated by the dispatch sites
  // and not the other way around) but would be cleaner.

  // Returns the potential constant values across all dispatch sites to
  // |exportOp| for the operand at |operandIdx|.
  DFX::PotentialConstantIntValuesState
  getOperandPVS(IREE::Stream::ExecutableExportOp exportOp,
                unsigned operandIdx) {
    DFX::PotentialConstantIntValuesState state;
    for (auto dispatchOp : getDispatchSites(exportOp)) {
      auto element = solver.lookupElementFor<ValuePVS>(
          Position::forValue(dispatchOp.getUniformOperands()[operandIdx]));
      if (!element) {
        state.unionAssumedWithUndef();
        state.indicatePessimisticFixpoint();
        break;
      }
      state ^= element->getState();
    }
    return state;
  }

  // Returns the minimum alignment across all dispatch sites to |exportOp| for
  // the operand at |operandIdx|.
  llvm::MaybeAlign
  getOperandAlignment(IREE::Stream::ExecutableExportOp exportOp,
                      unsigned operandIdx) {
    llvm::MaybeAlign alignment;
    for (auto dispatchOp : getDispatchSites(exportOp)) {
      auto element = solver.lookupElementFor<ValueAlignment>(
          Position::forValue(dispatchOp.getUniformOperands()[operandIdx]));
      if (!element || !element->isValidState())
        return llvm::MaybeAlign();
      alignment = commonAlignment(alignment, element->getAssumedAlignment());
    }
    if (alignment.valueOrOne().value() == kMaximumAlignment) {
      return llvm::MaybeAlign();
    }
    return alignment;
  }

  // Returns the minimum alignment across all dispatch sites to |exportOp| for
  // the resource offset at |resourceIdx|.
  llvm::MaybeAlign
  getResourceOffsetAlignment(IREE::Stream::ExecutableExportOp exportOp,
                             unsigned resourceIdx) {
    llvm::MaybeAlign alignment;
    for (auto dispatchOp : getDispatchSites(exportOp)) {
      auto element = solver.lookupElementFor<ValueAlignment>(
          Position::forValue(dispatchOp.getResourceOffsets()[resourceIdx]));
      if (!element || !element->isValidState())
        return llvm::MaybeAlign();
      alignment = commonAlignment(alignment, element->getAssumedAlignment());
    }
    if (alignment.valueOrOne().value() == kMaximumAlignment) {
      // Alignment is natural, which for resources means the base resource
      // alignment.
      auto configAttr = IREE::Stream::ResourceConfigAttr::lookup(exportOp);
      return llvm::MaybeAlign(configAttr.getMinBufferOffsetAlignment());
    }
    return alignment;
  }

private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;

  DenseMap<Operation *, SmallVector<IREE::Stream::CmdDispatchOp>>
      entryDispatchMap;
};

//===----------------------------------------------------------------------===//
// Per-dispatchable export argument annotation
//===----------------------------------------------------------------------===//

// Annotates |exportOp| (and its target function) with information derived from
// all dispatch sites of that export.
static void annotateExport(IREE::Stream::ExecutableOp executableOp,
                           IREE::Stream::ExecutableExportOp exportOp,
                           ArgumentAnalysis &analysis) {
  auto *context = executableOp.getContext();

  // Operands/resources on the func are in an arbitrary order; get maps that
  // lets us go from dispatch site operand/resource to function argument.
  auto funcOp = exportOp.lookupFunctionRef();
  if (!funcOp)
    return;
  auto operandToArgMap =
      IREE::Stream::CmdDispatchOp::makeOperandToArgMap(funcOp);
  auto resourceToArgMap =
      IREE::Stream::CmdDispatchOp::makeResourceToArgMap(funcOp);

  auto indexType = IndexType::get(context);

  // Annotate operand arguments with their potential values and alignment.
  for (unsigned operandIdx = 0; operandIdx < operandToArgMap.size();
       ++operandIdx) {
    unsigned argIdx = operandToArgMap[operandIdx];
    auto argType = funcOp.getArgument(argIdx).getType();

    auto pvs = analysis.getOperandPVS(exportOp, operandIdx);
    if (pvs.isValidState() && !pvs.isUndefContained()) {
      SmallVector<Attribute> potentialValues;
      potentialValues.reserve(pvs.getAssumedSet().size());
      for (auto value : pvs.getAssumedSet()) {
        potentialValues.push_back(IntegerAttr::get(argType, value));
      }
      llvm::sort(potentialValues, [](Attribute lhs, Attribute rhs) {
        auto lhsInt = llvm::dyn_cast<IntegerAttr>(lhs);
        auto rhsInt = llvm::dyn_cast<IntegerAttr>(rhs);
        if (!lhsInt || !rhsInt)
          return false;
        return lhsInt.getValue().ult(rhsInt.getValue());
      });
      auto potentialValuesAttr = ArrayAttr::get(context, potentialValues);
      funcOp.setArgAttr(argIdx, "stream.values", potentialValuesAttr);
    }

    if (argType.isIndex()) {
      auto alignment = analysis.getOperandAlignment(exportOp, operandIdx);
      if (alignment.has_value()) {
        uint64_t alignmentOrOne = alignment.valueOrOne().value();
        funcOp.setArgAttr(argIdx, "stream.alignment",
                          IntegerAttr::get(indexType, alignmentOrOne));
      }
    }
  }

  // Annotate binding arguments with their base alignment.
  for (unsigned resourceIdx = 0; resourceIdx < resourceToArgMap.size();
       ++resourceIdx) {
    unsigned argIdx = resourceToArgMap[resourceIdx];
    auto alignment = analysis.getResourceOffsetAlignment(exportOp, resourceIdx);
    if (alignment.has_value()) {
      uint64_t alignmentOrOne = alignment.valueOrOne().value();
      funcOp.setArgAttr(argIdx, "stream.alignment",
                        IntegerAttr::get(indexType, alignmentOrOne));
    }
  }
}

//===----------------------------------------------------------------------===//
// --iree-stream-specialize-dispatches
//===----------------------------------------------------------------------===//

struct AnnotateDispatchArgumentsPass
    : public IREE::Stream::impl::AnnotateDispatchArgumentsPassBase<
          AnnotateDispatchArgumentsPass> {
  void runOnOperation() override {
    // Perform argument value analysis.
    ArgumentAnalysis analysis(getOperation());
    if (failed(analysis.run())) {
      return signalPassFailure();
    }

    // Annotate the exported dispatch functions.
    for (auto executableOp :
         getOperation().getBodyRegion().getOps<IREE::Stream::ExecutableOp>()) {
      for (auto exportOp :
           executableOp.getOps<IREE::Stream::ExecutableExportOp>()) {
        annotateExport(executableOp, exportOp, analysis);
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
