// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <iterator>

#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTraits.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#define DEBUG_TYPE "iree-util-ipo"

namespace mlir::iree_compiler::IREE::Util {
namespace {

struct LocAttr {
  std::optional<Location> loc;
  Type type;
  Attribute attr;
  operator bool() const { return attr != nullptr; }
};

// Sentinel indicating an unused/invalid slot.
static const int kUnassigned = -1;

// TODO(benvanik): track global loads/stores - we should move those across
// calls so that global folding works better. We could make an op interface
// for allowing ops to control this maybe? Timepoint joins should be sunk into
// callees for example.
struct FuncAnalysis {
  // Function under analysis.
  func::FuncOp funcOp;
  // All call sites across the whole program.
  SmallVector<func::CallOp> callOps;

  // Whether this function may be accessed indirectly or used externally.
  // This generally disables optimizations.
  bool isIncomplete = false;

  // Which args are uniform from all call sites.
  BitVector callerUniformArgs;
  // Values for each arg if they are uniformly constant at all call sites.
  SmallVector<LocAttr> callerUniformArgValues;
  // Uniform call operand index -> deduplicated index.
  // Base/non-duplicated values will be identity.
  // Example: func.call @foo(%a, %b, %a, %b) -> (0, 1, 0, 1)
  SmallVector<int> callerUniformArgDupeMap;
  // Which results are used by any caller of the function.
  BitVector callerUsedResults;

  // Which args are used by the function.
  BitVector calleeUsedArgs;
  // Which results are uniform from all return sites in the function.
  BitVector calleeUniformResults;
  // Values for each result if they are uniformly constant at all return sites.
  SmallVector<LocAttr> calleeUniformResultValues;
  // Uniform callee return operand index -> deduplicated index.
  // Base/non-duplicated values will be identity.
  // Example: func.return %a, %b, %a, %b -> (0, 1, 0, 1)
  SmallVector<int> calleeUniformResultDupeMap;

  // Result values mapping to argument indices if they are pass-through or -1.
  // Example:
  //   func.func @foo(%arg0: i32, %arg1: i32) -> i32 { return %arg1 : i32 }
  //   = [kUnassigned, 1]
  SmallVector<int> passthroughResultArgs;

  void print(llvm::raw_ostream &os, AsmState &asmState) {
    os << "FuncAnalysis: " << (isIncomplete ? "INCOMPLETE! " : "") << "@"
       << funcOp.getName() << funcOp.getFunctionType() << " "
       << "\n";
    auto argTypes = funcOp.getArgumentTypes();
    os << "  args: " << argTypes.size() << "\n";
    for (unsigned i = 0; i < argTypes.size(); ++i) {
      os << "    %arg" << i << ": ";
      os << (callerUniformArgs.test(i) ? "uniform" : "non-uniform") << " ";
      os << (calleeUsedArgs.test(i) ? "used" : "unused") << " ";
      if (callerUniformArgDupeMap[i] != i) {
        os << "dupe(%arg" << callerUniformArgDupeMap[i] << ") ";
      }
      os << argTypes[i] << " ";
      if (callerUniformArgValues[i]) {
        os << "constant = ";
        callerUniformArgValues[i].attr.print(os);
      }
      os << "\n";
    }
    auto resultTypes = funcOp.getResultTypes();
    os << "  results: " << resultTypes.size() << "\n";
    for (unsigned i = 0; i < resultTypes.size(); ++i) {
      os << "    %result#" << i << ": ";
      os << (calleeUniformResults.test(i) ? "uniform" : "non-uniform") << " ";
      os << (callerUsedResults.test(i) ? "used" : "unused") << " ";
      if (calleeUniformResultDupeMap[i] != i) {
        os << "dupe(%result#" << calleeUniformResultDupeMap[i] << ") ";
      }
      if (passthroughResultArgs[i] != kUnassigned) {
        os << "pass(%arg" << passthroughResultArgs[i] << ") ";
      }
      os << resultTypes[i] << " ";
      if (calleeUniformResultValues[i]) {
        os << "constant = ";
        calleeUniformResultValues[i].attr.print(os);
      }
      os << "\n";
    }
    os << "  callOps: " << callOps.size() << "\n";
    for (auto [i, callOp] : llvm::enumerate(callOps)) {
      os << "    [" << i << "]: ";
      callOp.print(os, asmState);
      os << "\n";
    }
  }
};

// Note that the analysis results may be incomplete.
static FuncAnalysis analyzeFuncOp(func::FuncOp funcOp, Explorer &explorer) {
  // Gather callers from across the program.
  FuncAnalysis analysis;
  analysis.funcOp = funcOp;
  analysis.isIncomplete = funcOp.isPublic() || funcOp.isExternal();
  if (explorer.walkIncomingCalls(funcOp, [&](mlir::CallOpInterface callOp) {
        if (auto funcCallOp = dyn_cast<func::CallOp>((Operation *)callOp)) {
          analysis.callOps.push_back(funcCallOp);
        } else {
          analysis.isIncomplete = true;
        }
        return WalkResult::advance();
      }) == TraversalResult::INCOMPLETE) {
    analysis.isIncomplete = true;
  }

  // Presize data types so we can index them freely below.
  unsigned argCount = funcOp.getNumArguments();
  unsigned resultCount = funcOp.getNumResults();
  analysis.callerUniformArgs.resize(argCount, true);
  analysis.callerUniformArgValues.resize(argCount);
  analysis.callerUniformArgDupeMap.resize(argCount, kUnassigned);
  analysis.callerUsedResults.resize(resultCount, true);
  analysis.calleeUsedArgs.resize(argCount, true);
  analysis.calleeUniformResults.resize(resultCount, true);
  analysis.calleeUniformResultValues.resize(resultCount);
  analysis.calleeUniformResultDupeMap.resize(resultCount, kUnassigned);
  analysis.passthroughResultArgs.resize(resultCount, kUnassigned);

  // Walk callee arguments.
  for (auto [i, value] : llvm::enumerate(funcOp.getArguments())) {
    if (value.use_empty())
      analysis.calleeUsedArgs.reset(i);
  }

  // Walk all return sites in the function.
  SmallVector<Value> seenResultValues(resultCount);
  funcOp.walk([&](func::ReturnOp returnOp) {
    for (auto [i, value] : llvm::enumerate(returnOp.getOperands())) {
      // Check to see if the value returned is a constant and stash.
      // We'll only use this value if all return sites are uniform.
      Attribute constantValue;
      if (matchPattern(value, m_Constant(&constantValue))) {
        analysis.calleeUniformResultValues[i] = {
            value.getLoc(),
            value.getType(),
            constantValue,
        };
      }

      // Check to see if the value returned is the same as previously seen.
      if (!seenResultValues[i]) {
        // First return site: take the value directly.
        seenResultValues[i] = value;
      } else if (seenResultValues[i] != value) {
        // Value has changed: mark non-uniform.
        analysis.calleeUniformResults.reset(i);
      }

      // Scan for duplication. nlogn.
      int dupeIndex = kUnassigned;
      for (int j = 0; j < i; ++j) {
        if (returnOp.getOperand(j) == value) {
          dupeIndex = j;
          break;
        }
      }
      if (analysis.calleeUniformResultDupeMap[i] == kUnassigned ||
          analysis.calleeUniformResultDupeMap[i] == dupeIndex) {
        analysis.calleeUniformResultDupeMap[i] = dupeIndex;
      } else {
        analysis.calleeUniformResultDupeMap[i] = i;
      }

      // If the result value is an argument track that here.
      // We'll only use this value if all return sites are uniform.
      if (auto arg = llvm::dyn_cast<BlockArgument>(value)) {
        if (arg.getParentBlock()->isEntryBlock()) {
          analysis.passthroughResultArgs[i] =
              static_cast<int>(arg.getArgNumber());
        }
      }
    }
  });

  // Walk all callers of the function.
  SmallVector<Value> seenArgValues(argCount);
  SmallVector<Attribute> seenArgAttrs(argCount);
  BitVector callerUnusedResults(resultCount, true);
  for (auto callOp : analysis.callOps) {
    // Handle call operands -> func arguments.
    for (auto [i, value] : llvm::enumerate(callOp.getArgOperands())) {
      // Check to see if the value is a constant and stash.
      // We'll only use this value if all call sites are uniform.
      Attribute constantValue;
      if (matchPattern(value, m_Constant(&constantValue))) {
        analysis.callerUniformArgValues[i] = {
            value.getLoc(),
            value.getType(),
            constantValue,
        };
      } else {
        // Check to see if the value is the same as previously seen.
        // This will ensure that across calling functions we set non-uniform
        // _unless_ it's a constant value.
        if (!seenArgValues[i]) {
          // First call site: take the value directly.
          seenArgValues[i] = value;
        } else if (seenArgValues[i] != value) {
          // Value has changed and is not constant: mark non-uniform.
          analysis.callerUniformArgs.reset(i);
        }
      }

      // Check to see if the constant value is the same as previously seen.
      // NOTE: unlike callee results we only check constant values.
      if (!seenArgAttrs[i]) {
        // First call site: take the value directly.
        seenArgAttrs[i] = constantValue;
      } else if (seenArgAttrs[i] != constantValue) {
        // Value has changed: mark non-uniform.
        analysis.callerUniformArgs.reset(i);
      }

      // Scan for duplication. nlogn.
      int dupeIndex = kUnassigned;
      for (int j = 0; j < i; ++j) {
        if (callOp.getOperand(j) == value) {
          dupeIndex = j;
          break;
        }
      }
      if (analysis.callerUniformArgDupeMap[i] == kUnassigned ||
          analysis.callerUniformArgDupeMap[i] == dupeIndex) {
        analysis.callerUniformArgDupeMap[i] = dupeIndex;
      } else {
        analysis.callerUniformArgDupeMap[i] = i;
      }
    }

    // Handle func results -> call results.
    // Note that we need to track unused results as an AND such that all callers
    // need to not use them. We'll flip the bits below so that `used = true`.
    for (auto [i, value] : llvm::enumerate(callOp.getResults())) {
      if (!value.use_empty())
        callerUnusedResults.reset(i);
    }
  }
  if (!analysis.callOps.empty()) {
    callerUnusedResults.flip();
    analysis.callerUsedResults = callerUnusedResults;
  }

  // Derive validity of fields that require uniformity.
  // Users of the analysis should check anyway but this makes debugging
  // easier.
  for (unsigned i = 0; i < argCount; ++i) {
    if (!analysis.callerUniformArgs.test(i)) {
      analysis.callerUniformArgValues[i] = {};
    }
    if (analysis.callerUniformArgDupeMap[i] == kUnassigned) {
      analysis.callerUniformArgDupeMap[i] = i;
    }
  }
  for (unsigned i = 0; i < resultCount; ++i) {
    if (!analysis.calleeUniformResults.test(i)) {
      analysis.calleeUniformResultValues[i] = {};
      analysis.passthroughResultArgs[i] = kUnassigned;
    }
    if (analysis.calleeUniformResultDupeMap[i] == kUnassigned) {
      analysis.calleeUniformResultDupeMap[i] = i;
    }
  }

  // If analysis was incomplete we reset things to safe values.
  if (analysis.isIncomplete) {
    for (unsigned i = 0; i < argCount; ++i) {
      analysis.callerUniformArgs.reset();
      analysis.callerUniformArgValues[i] = {};
      analysis.callerUniformArgDupeMap[i] = i;
    }
    for (unsigned i = 0; i < resultCount; ++i) {
      analysis.calleeUniformResults.reset();
      analysis.calleeUniformResultValues[i] = {};
      analysis.calleeUniformResultDupeMap[i] = i;
      analysis.callerUsedResults.set();
    }
  }

  // We can drop any pass-through args that are exclusively used by returns as
  // we know all callers will stop passing them.
  for (unsigned i = 0; i < resultCount; ++i) {
    int argIndex = analysis.passthroughResultArgs[i];
    if (argIndex == kUnassigned)
      continue;
    auto arg = funcOp.getArgument(argIndex);
    bool onlyReturnUsers = true;
    for (auto user : arg.getUsers()) {
      if (!isa<func::ReturnOp>(user)) {
        onlyReturnUsers = false;
        break;
      }
    }
    if (onlyReturnUsers) {
      analysis.calleeUsedArgs.reset(argIndex);
    }
  }

  // Any argument that is the base of a duplicate needs to inherit the usage
  // of all pointing at it.
  // For example, %arg0 unused + %arg1 used dupe(%arg0) needs to ensure that
  // %arg0 is preserved.
  for (unsigned i = 0; i < argCount; ++i) {
    int dupeIndex = analysis.callerUniformArgDupeMap[i];
    if (dupeIndex != i && analysis.calleeUsedArgs.test(i)) {
      analysis.calleeUsedArgs.set(dupeIndex);
    }
  }

  return analysis;
}

// Replaces all uses of |value| with the result of a new |constantValue| op.
// Assumes that it's possible to materialize the constant op.
static void replaceValueWithConstant(Value value, LocAttr constantValue,
                                     OpBuilder &builder) {
  Operation *op = nullptr;

  // Handle special builtin types that for some reason can't materialize
  // themselves.
  if (arith::ConstantOp::isBuildableWith(constantValue.attr,
                                         constantValue.type)) {
    op = builder.create<arith::ConstantOp>(constantValue.loc.value(),
                                           constantValue.type,
                                           cast<TypedAttr>(constantValue.attr));
  }

  // Try the attr and type dialects to see if they can materialize.
  if (!op) {
    op = constantValue.attr.getDialect().materializeConstant(
        builder, constantValue.attr, constantValue.type,
        constantValue.loc.value());
  }
  if (!op) {
    op = constantValue.type.getDialect().materializeConstant(
        builder, constantValue.attr, constantValue.type,
        constantValue.loc.value());
  }

  // If we hit errors at this point then we need to rethink how this analysis
  // is performed - we may need to do something silly like materializing
  // constants off in a throw-away region as there's no direct way to query if a
  // constant is materializable. Ideally nothing that matches m_Constant should
  // be impossible to materialize but here we are.
  if (!op) {
    llvm::report_fatal_error("can't materialize constant; unsupported type");
    return;
  }

  // NOTE: we're assuming constant ops return their value at index 0. There's
  // no constant interface (just constant trait) so this is convention instead
  // of contract.
  value.replaceAllUsesWith(op->getResult(0));
}

// Returns true if any changes were made.
static bool applyFuncChanges(FuncAnalysis &analysis, func::FuncOp funcOp) {
  // Build the new set of function arguments and inline uniform constants.
  auto builder = OpBuilder::atBlockBegin(&funcOp.getBlocks().front());
  auto oldArgTypes = llvm::to_vector(funcOp.getArgumentTypes());
  SmallVector<Type> newArgTypes;
  newArgTypes.reserve(oldArgTypes.size());
  BitVector deadArgs(oldArgTypes.size(), false);
  for (auto [i, arg] : llvm::enumerate(funcOp.getArguments())) {
    // If unused by the function then drop.
    if (!analysis.calleeUsedArgs.test(i)) {
      deadArgs.set(i);
      continue;
    }
    // If uniformly constant at all call sites then replace with that value.
    if (auto constantValue = analysis.callerUniformArgValues[i]) {
      replaceValueWithConstant(arg, constantValue, builder);
      deadArgs.set(i);
      continue;
    }
    // If a duplicate then we replace uses with the base value.
    int dupeIndex = analysis.callerUniformArgDupeMap[i];
    if (dupeIndex != i) {
      arg.replaceAllUsesWith(funcOp.getArgument(dupeIndex));
      deadArgs.set(i);
      continue;
    }
    newArgTypes.push_back(arg.getType());
  }

  // Build the new set of result types.
  auto oldResultTypes = llvm::to_vector(funcOp.getResultTypes());
  SmallVector<Type> newResultTypes;
  newResultTypes.reserve(oldResultTypes.size());
  BitVector deadResults(oldResultTypes.size(), false);
  for (auto [i, type] : llvm::enumerate(oldResultTypes)) {
    // If unused by all callers then drop.
    if (!analysis.callerUsedResults.test(i)) {
      deadResults.set(i);
      continue;
    }
    // If uniformly constant then inline at call sites and drop here.
    if (analysis.calleeUniformResultValues[i]) {
      deadResults.set(i);
      continue;
    }
    // If a duplicate then we drop here and fix up at call sites.
    if (analysis.calleeUniformResultDupeMap[i] != i) {
      deadResults.set(i);
      continue;
    }
    // If pass-through we drop here as the callers won't use the result.
    // This prevents the need for another IPO pass to clean them up.
    if (analysis.passthroughResultArgs[i] != kUnassigned) {
      deadResults.set(i);
      continue;
    }
    newResultTypes.push_back(type);
  }

  // Early out if no changes.
  if (deadArgs.none() && deadResults.none())
    return false;

  // Erase dead results from all return sites.
  funcOp.walk([&](func::ReturnOp returnOp) {
    for (int i = deadResults.size() - 1; i >= 0; --i) {
      if (deadResults.test(i))
        returnOp.getOperandsMutable().erase(i);
    }
  });

  // Erase dead args/results - args uses should have either been unused or
  // replaced with constants above. Note that because results may be using args
  // we need to drop those first above.
  funcOp.eraseArguments(deadArgs);
  funcOp.eraseResults(deadResults);

  return true;
}

// Returns true if any changes were made.
static bool applyCallChanges(FuncAnalysis &analysis, func::CallOp callOp) {
  // Build the new set of call operands.
  SmallVector<Value> oldOperands = callOp.getOperands();
  SmallVector<Value> newOperands;
  newOperands.reserve(oldOperands.size());
  BitVector deadOperands(oldOperands.size(), false);
  for (auto [i, value] : llvm::enumerate(oldOperands)) {
    // If the arg isn't used by the callee then we drop from all.
    if (!analysis.calleeUsedArgs.test(i)) {
      deadOperands.set(i);
      continue;
    }
    // If the arg is uniformly constant then we inline it and drop from all.
    if (analysis.callerUniformArgValues[i]) {
      deadOperands.set(i);
      continue;
    }
    // If the arg is duplicated then we drop all but the base value.
    if (analysis.callerUniformArgDupeMap[i] != i) {
      deadOperands.set(i);
      continue;
    }
    newOperands.push_back(value);
  }

  // Build the new set of return values and inline constant results.
  OpBuilder builder(callOp);
  builder.setInsertionPointAfter(callOp);
  SmallVector<Value> oldResults = callOp.getResults();
  SmallVector<Value> newResults;
  newResults.reserve(oldResults.size());
  SmallVector<Type> newResultTypes;
  newResultTypes.reserve(oldResults.size());
  BitVector deadResults(oldResults.size(), false);
  for (auto [i, value] : llvm::enumerate(oldResults)) {
    // If the result isn't used by any caller then we drop from all.
    if (!analysis.callerUsedResults.test(i)) {
      assert(value.use_empty() && "analysis said no uses");
      deadResults.set(i);
      continue;
    }
    // If the result is uniformly constant then we can replace with that.
    if (auto constantValue = analysis.calleeUniformResultValues[i]) {
      replaceValueWithConstant(value, constantValue, builder);
      deadResults.set(i);
      continue;
    }
    // If the result is a duplicate then we replace uses with the base value.
    int dupeIndex = analysis.calleeUniformResultDupeMap[i];
    if (dupeIndex != i) {
      value.replaceAllUsesWith(oldResults[dupeIndex]);
      deadResults.set(i);
      continue;
    }
    // If pass-through then just use the arg we were passing in as the result.
    int passthroughIndex = analysis.passthroughResultArgs[i];
    if (passthroughIndex != kUnassigned) {
      value.replaceAllUsesWith(oldOperands[passthroughIndex]);
      deadResults.set(i);
      continue;
    }
    newResults.push_back(value);
    newResultTypes.push_back(value.getType());
  }

  // Early out if no changes.
  if (deadOperands.none() && deadResults.none())
    return false;

  // Fully replace call op because we may have changed result count.
  auto newCallOp = OpBuilder(callOp).create<func::CallOp>(
      callOp.getLoc(), callOp.getCalleeAttr(), newResultTypes, newOperands);
  newCallOp->setDialectAttrs(callOp->getDialectAttrs());

  // Remap live old results -> new results.
  for (auto [oldValue, newValue] :
       llvm::zip_equal(newResults, newCallOp.getResults())) {
    oldValue.replaceAllUsesWith(newValue);
  }

  // Erase old op now that all uses are (or should be) replaced.
  callOp.erase();

  return true;
}

class IPOPass : public IPOBase<IPOPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // TODO(benvanik): find a nice way of skipping embedded executables. Maybe
    // an op interface like the inliner control interface. For now we recurse
    // into executables but since they (usually) only have a single call it's
    // relatively cheap as nothing changes.
    Explorer explorer(moduleOp, TraversalAction::RECURSE);
    explorer.initialize();

    // Analyze all top-level functions. We do some invasive surgery that can't
    // happen through callable interfaces today. Since we're joining data from
    // across the whole program we can't perform any mutations during this
    // analysis.
    std::vector<FuncAnalysis> analysisResults;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      analysisResults.push_back(analyzeFuncOp(funcOp, explorer));
    }

    LLVM_DEBUG({
      AsmState asmState(moduleOp);
      for (auto &analysis : analysisResults) {
        analysis.print(llvm::dbgs(), asmState);
      }
    });

    // Use analysis results to mutate functions.
    bool anyChanges = false;
    for (auto &analysis : analysisResults) {
      if (analysis.isIncomplete)
        continue;
      anyChanges = applyFuncChanges(analysis, analysis.funcOp) || anyChanges;
      for (auto callOp : analysis.callOps) {
        anyChanges = applyCallChanges(analysis, callOp) || anyChanges;
      }
    }

    // When running under the FixedPointIterator pass we need to signal when we
    // made a change.
    if (anyChanges) {
      signalFixedPointModified(moduleOp);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createIPOPass() {
  return std::make_unique<IPOPass>();
}

static PassRegistration<IPOPass> pass;

} // namespace mlir::iree_compiler::IREE::Util
