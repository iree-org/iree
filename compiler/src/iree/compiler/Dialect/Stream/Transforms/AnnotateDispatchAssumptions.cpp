// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/IntegerDivisibilityAnalysis.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-stream-annotate-dispatch-assumptions"

using llvm::dbgs;

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ANNOTATEDISPATCHASSUMPTIONSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Per-dispatchable export argument annotation
//===----------------------------------------------------------------------===//

class ArgumentAnalysis {
public:
  explicit ArgumentAnalysis(Operation *rootOp) {
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<mlir::dataflow::IntegerRangeAnalysis>();
    solver.load<IREE::Util::IntegerDivisibilityAnalysis>();

    // Find all dispatches and bucket by their target entry point.
    rootOp->walk([&](IREE::Stream::CmdDispatchOp dispatchOp) {
      dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPointAttr) {
        auto exportOp =
            symbolTables.lookupNearestSymbolFrom(dispatchOp, entryPointAttr);
        entryDispatchMap[exportOp].push_back(dispatchOp);
      });

      auto parentFunc = dispatchOp->getParentOfType<FunctionOpInterface>();
      assert(parentFunc && "dispatch not contained by function");
      if (parentFunc) {
        analysisRoots.insert(parentFunc);
      }
    });
  }

  DataFlowSolver &getSolver() { return solver; }

  LogicalResult run() {
    for (Operation *analysisRoot : analysisRoots) {
      if (failed(solver.initializeAndRun(analysisRoot)))
        return failure();
    }
    return success();
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

  std::pair<ArrayAttr, bool>
  getOperandAssumptions(IREE::Stream::ExecutableExportOp exportOp,
                        unsigned operandIdx) {
    auto *context = exportOp.getContext();
    bool hasNonEmptyAssumption = false;
    SmallVector<IREE::Util::IntAssumptionAttr> assumptions;
    for (auto dispatchOp : getDispatchSites(exportOp)) {
      std::optional<uint64_t> umin;
      std::optional<uint64_t> umax;
      std::optional<uint64_t> udiv;

      Value operand = dispatchOp.getUniformOperands()[operandIdx];
      auto *rangeState =
          solver.lookupState<dataflow::IntegerValueRangeLattice>(operand);
      if (rangeState && !rangeState->getValue().isUninitialized()) {
        // Only add an assumption if the range has any meaning.
        APInt minValue = rangeState->getValue().getValue().umin();
        APInt maxValue = rangeState->getValue().getValue().umax();
        if (!minValue.isMinValue() || !maxValue.isMaxValue()) {
          umin = minValue.getZExtValue();
          umax = maxValue.getZExtValue();
        }
      }
      auto *divState =
          solver.lookupState<IREE::Util::IntegerDivisibilityLattice>(operand);
      if (divState && !divState->getValue().isUninitialized()) {
        uint64_t divValue = divState->getValue().getValue().udiv();
        if (divValue > 1) {
          udiv = divValue;
        }
      }

      if (umin || umax || udiv) {
        hasNonEmptyAssumption = true;
      }
      assumptions.push_back(
          IREE::Util::IntAssumptionAttr::get(context, umin, umax, udiv));
    }

    if (assumptions.empty())
      return {};

    return std::make_pair(
        ArrayAttr::get(context, ArrayRef<Attribute>(assumptions.begin(),
                                                    assumptions.end())),
        hasNonEmptyAssumption);
  }

private:
  DataFlowSolver solver;
  SymbolTableCollection symbolTables;

  // Root functions that we are performing analysis on.
  DenseSet<Operation *> analysisRoots;

  // Map of export entry point to inbound list of dispatches.
  DenseMap<Operation *, SmallVector<IREE::Stream::CmdDispatchOp>>
      entryDispatchMap;
};

// Annotates |exportOp| (and its target function) with information derived from
// all dispatch sites of that export.
static void annotateExport(IREE::Stream::ExecutableOp executableOp,
                           IREE::Stream::ExecutableExportOp exportOp,
                           ArgumentAnalysis &analysis) {
  // Operands/resources on the func are in an arbitrary order; get maps that
  // lets us go from dispatch site operand/resource to function argument.
  auto funcOp = exportOp.lookupFunctionRef();
  if (!funcOp || funcOp.empty())
    return;
  auto operandToArgMap =
      IREE::Stream::CmdDispatchOp::makeOperandToArgMap(funcOp);
  auto resourceToArgMap =
      IREE::Stream::CmdDispatchOp::makeResourceToArgMap(funcOp);

  // List of arguments to annotate and the corresponding list of assumptions.
  SmallVector<Value> arguments;
  SmallVector<ArrayAttr> argumentAssumptions;

  // Annotate operand arguments with their potential values and alignment.
  int nonEmptyCount = 0;
  for (unsigned operandIdx = 0; operandIdx < operandToArgMap.size();
       ++operandIdx) {
    unsigned argIdx = operandToArgMap[operandIdx];
    Value argValue = funcOp.getArgument(argIdx);
    Type argType = argValue.getType();
    if (!argType.isIndex() && !argType.isInteger())
      continue;

    auto [assumptions, hasNonEmpty] =
        analysis.getOperandAssumptions(exportOp, operandIdx);
    if (hasNonEmpty) {
      nonEmptyCount += 1;
      arguments.push_back(argValue);
      argumentAssumptions.push_back(assumptions);
    }
  }

  if (nonEmptyCount == 0)
    return;

  // Do the rewrite.
  OpBuilder builder = OpBuilder::atBlockBegin(&funcOp.front());
  auto assumeOp = builder.create<IREE::Util::AssumeIntOp>(
      funcOp.getLoc(), arguments, argumentAssumptions);
  for (unsigned argIndex = 0; argIndex < arguments.size(); ++argIndex) {
    arguments[argIndex].replaceAllUsesExcept(assumeOp.getResult(argIndex),
                                             assumeOp);
  }
}

class AnnotateDispatchAssumptionsPass
    : public IREE::Stream::impl::AnnotateDispatchAssumptionsPassBase<
          AnnotateDispatchAssumptionsPass> {
  void runOnOperation() override {
    ArgumentAnalysis analysis(getOperation());
    if (failed(analysis.run()))
      return signalPassFailure();
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
