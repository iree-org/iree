// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/PassDetail.h"
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_INLINESHAPEASSERTIONHELPERS
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h.inc"

namespace {

// Placed functions and calls keep their boundary.
constexpr StringLiteral kAffinityName = "stream.affinity";

bool isShapeScalar(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return tensorType.getRank() == 0 &&
           isa<IntegerType>(tensorType.getElementType());
  }
  return isa<IntegerType>(type);
}

bool isMarkedNoInline(Operation *op) {
  return op->hasAttr("no_inline") || op->hasAttr("noinline") ||
         op->hasAttr(kAffinityName);
}

// Borrowed call/function handles remain valid through analysis. Application
// consumes each call once and never reads its handle after erasure.
struct ShapeHelperInfo {
  enum class State { Visiting, Accepted, Rejected };
  State state = State::Visiting;
  SmallVector<func::CallOp> calls;
  SmallVector<func::FuncOp> callees;
  uint64_t expandedOperations = 0;
};

func::FuncOp resolveShapeHelper(func::CallOp call,
                                SymbolTableCollection &symbolTables) {
  if (call.getNoInline() || isMarkedNoInline(call)) {
    return nullptr;
  }
  return symbolTables.lookupNearestSymbolFrom<func::FuncOp>(
      call, call.getCalleeAttr());
}

// Classifies the complete dependency closure without changing any helper body.
// Explicit DFS frames bound the C++ stack independently of call-graph depth.
class ShapeHelperAnalysis {
public:
  explicit ShapeHelperAnalysis(SymbolTableCollection &symbolTables)
      : symbolTables(symbolTables) {}

  void analyze(func::FuncOp root) {
    if (helpers.contains(root)) {
      return;
    }
    inspectBody(root);
    struct Frame {
      func::FuncOp function;
      size_t nextCall = 0;
    };
    SmallVector<Frame> stack{{root}};
    while (!stack.empty()) {
      func::FuncOp function = stack.back().function;
      auto &info = helpers.at(function);
      if (info.state == ShapeHelperInfo::State::Rejected ||
          stack.back().nextCall == info.callees.size()) {
        if (info.state == ShapeHelperInfo::State::Visiting) {
          info.state = ShapeHelperInfo::State::Accepted;
          postorder.push_back(function);
        }
        stack.pop_back();
        continue;
      }
      func::FuncOp child = info.callees[stack.back().nextCall];
      if (!helpers.contains(child)) {
        // Insertion can invalidate DenseMap references; reacquire next time.
        inspectBody(child);
        stack.push_back({child});
        continue;
      }
      const auto &childInfo = helpers.at(child);
      if (childInfo.state != ShapeHelperInfo::State::Accepted) {
        // The child is rejected, or still visiting because this call is a
        // back edge. Reject this helper and let rejection propagate to every
        // caller on the explicit stack.
        info.state = ShapeHelperInfo::State::Rejected;
        continue;
      }
      // Count every call occurrence, including repeated children in a diamond.
      // Retaining the original call/return costs is a conservative
      // overestimate.
      info.expandedOperations = llvm::SaturatingAdd(
          info.expandedOperations, childInfo.expandedOperations);
      ++stack.back().nextCall;
    }
  }

  const ShapeHelperInfo &get(func::FuncOp function) const {
    return helpers.at(function);
  }
  ArrayRef<func::FuncOp> getPostorder() const { return postorder; }

private:
  // Records `function` as a candidate helper, or as rejected unless it is a
  // private, single-block, inlinable function of shape scalars whose body only
  // holds pure operations and calls to other candidate helpers.
  void inspectBody(func::FuncOp function) {
    ShapeHelperInfo info;
    bool eligible =
        hasShapeHelperSignature(function) && recordBody(function, info);
    if (!eligible) {
      info.state = ShapeHelperInfo::State::Rejected;
    }
    helpers.try_emplace(function, std::move(info));
  }

  static bool hasShapeHelperSignature(func::FuncOp function) {
    return function.isPrivate() && !function.isExternal() &&
           function.getBody().hasOneBlock() && !isMarkedNoInline(function) &&
           llvm::all_of(function.getArgumentTypes(), isShapeScalar) &&
           llvm::all_of(function.getResultTypes(), isShapeScalar);
  }

  // Counts the operations of `function` into `info` and records its helper
  // calls. Returns false at the first operation a helper may not contain.
  bool recordBody(func::FuncOp function, ShapeHelperInfo &info) {
    WalkResult result = function.walk([&](Operation *op) {
      if (op == function) {
        return WalkResult::advance();
      }
      ++info.expandedOperations;
      return recordBodyOperation(op, info) ? WalkResult::advance()
                                           : WalkResult::interrupt();
    });
    return !result.wasInterrupted();
  }

  // Returns whether a helper may contain `op`, recording it in `info` if it is
  // a call to another candidate helper.
  bool recordBodyOperation(Operation *op, ShapeHelperInfo &info) {
    if (isa<func::ReturnOp>(op)) {
      return true;
    }
    auto call = dyn_cast<func::CallOp>(op);
    if (!call) {
      return isPure(op);
    }
    func::FuncOp callee = resolveShapeHelper(call, symbolTables);
    if (!callee) {
      return false;
    }
    info.calls.push_back(call);
    info.callees.push_back(callee);
    return true;
  }

  SymbolTableCollection &symbolTables;
  llvm::DenseMap<Operation *, ShapeHelperInfo> helpers;
  SmallVector<func::FuncOp> postorder;
};

// Capture producer calls before mutation. Keep slice order and nearest-symbol
// lookup, and deduplicate calls shared by multiple assertion operands.
SmallVector<func::CallOp> collectAssertionProducerCalls(ModuleOp module) {
  llvm::SetVector<func::CallOp> calls;
  module.walk([&](mlir::stablehlo::CustomCallOp assertion) {
    if (assertion.getCallTargetName() != "shape_assertion") {
      return;
    }
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.inclusive = false;
    SetVector<Operation *> slice;
    (void)getBackwardSlice(assertion, &slice, options);
    for (Operation *producer : slice) {
      if (auto call = dyn_cast<func::CallOp>(producer)) {
        calls.insert(call);
      }
    }
  });
  return calls.takeVector();
}

// Inlines accepted shape helpers while a pass-wide cloning allowance lasts.
// A declined or failed inline leaves its call and runtime assertion intact.
class BudgetedShapeHelperInliner {
public:
  BudgetedShapeHelperInliner(ModuleOp module,
                             const ShapeHelperAnalysis &analysis)
      : analysis(analysis), interface(module.getContext()) {
    uint64_t originalOperations = 0;
    module.walk([&](Operation *) { ++originalOperations; });
    // Permit ordinary small helpers while bounding repeated-call DAG
    // expansion. This is an optimization allowance, not a semantic limit on
    // valid programs.
    cloningAllowance = llvm::SaturatingMultiplyAdd(originalOperations,
                                                   uint64_t{8}, uint64_t{4096});
    remainingOperations = cloningAllowance;
  }

  // Inlines `callee` at `call` if the helper is accepted and its body fits
  // the remaining allowance, and erases the call on success.
  void inlineCallWithinBudget(func::CallOp call, func::FuncOp callee) {
    if (remainingOperations == 0) {
      return;
    }
    const auto &info = analysis.get(callee);
    if (info.state != ShapeHelperInfo::State::Accepted ||
        info.expandedOperations > cloningAllowance) {
      return;
    }
    // Reserve the current body, including nested regions, before cloning.
    // Counting the body avoids repeatedly charging already-erased calls along
    // long helper chains. The expansion estimate separately limits admission.
    uint64_t cloneCost = getCloneCost(callee);
    if (cloneCost > remainingOperations) {
      return;
    }
    remainingOperations -= cloneCost;
    if (succeeded(inlineCall(interface, config.getCloneCallback(), call, callee,
                             &callee.getBody()))) {
      call.erase();
    }
  }

private:
  // Returns the number of operations in the body of `callee`. The postorder
  // inlines into a helper before any call to it, so a callee's body is final
  // when it is first charged and its count can be reused.
  uint64_t getCloneCost(func::FuncOp callee) {
    auto [it, inserted] = bodyOperationCounts.try_emplace(callee, 0);
    if (inserted) {
      callee.getBody().walk([&](Operation *) { ++it->second; });
    }
    return it->second;
  }

  const ShapeHelperAnalysis &analysis;
  InlinerInterface interface;
  InlinerConfig config;
  uint64_t cloningAllowance = 0;
  uint64_t remainingOperations = 0;
  DenseMap<Operation *, uint64_t> bodyOperationCounts;
};

// Apply the immutable postorder plan: inline into each helper before any call
// to it, then inline the assertion producers' calls.
void inlineShapeHelperPlan(ModuleOp module,
                           ArrayRef<func::CallOp> producerCalls,
                           SymbolTableCollection &symbolTables,
                           const ShapeHelperAnalysis &analysis) {
  BudgetedShapeHelperInliner inliner(module, analysis);
  for (func::FuncOp function : analysis.getPostorder()) {
    const auto &info = analysis.get(function);
    for (auto [call, callee] : llvm::zip_equal(info.calls, info.callees)) {
      inliner.inlineCallWithinBudget(call, callee);
    }
  }
  for (func::CallOp call : producerCalls) {
    if (func::FuncOp callee = resolveShapeHelper(call, symbolTables)) {
      inliner.inlineCallWithinBudget(call, callee);
    }
  }
}

struct InlineShapeAssertionHelpers final
    : impl::InlineShapeAssertionHelpersBase<InlineShapeAssertionHelpers> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::CallOp> producerCalls =
        collectAssertionProducerCalls(module);
    SymbolTableCollection symbolTables;
    ShapeHelperAnalysis analysis(symbolTables);
    for (func::CallOp call : producerCalls) {
      if (func::FuncOp callee = resolveShapeHelper(call, symbolTables)) {
        analysis.analyze(callee);
      }
    }
    inlineShapeHelperPlan(module, producerCalls, symbolTables, analysis);
  }
};

} // namespace
} // namespace mlir::iree_compiler::stablehlo
