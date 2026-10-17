// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/PassDetail.h"
#include "compiler/plugins/input/StableHLO/Conversion/Passes.h"
#include "llvm/ADT/DenseMap.h"
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

// Inlines the private shape helpers that compute shape_assertion operands, so
// the assertion facts see the helpers' arithmetic.
class Inliner {
public:
  explicit Inliner(ModuleOp module) : interface(module.getContext()) {}

  // Inlines the qualifying calls among the producers of |op|'s operands.
  void inlineProducers(Operation *op) {
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.inclusive = false;
    SetVector<Operation *> slice;
    (void)getBackwardSlice(op, &slice, options);
    for (Operation *producer : slice) {
      if (auto call = dyn_cast<func::CallOp>(producer)) {
        inlineIfHelper(call);
      }
    }
  }

private:
  enum class State { Visiting, Helper, NotHelper };

  // Returns the callee if |call| may be inlined.
  func::FuncOp getHelper(func::CallOp call) {
    if (call.getNoInline() || isMarkedNoInline(call)) {
      return nullptr;
    }
    auto callee = symbolTables.lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    return callee && isHelper(callee) ? callee : nullptr;
  }

  // A helper is a private, pure, non-recursive single-block function over
  // scalar integers whose own calls are helpers. Its calls are inlined first.
  bool isHelper(func::FuncOp callee) {
    auto [it, inserted] = states.try_emplace(callee, State::Visiting);
    if (!inserted) {
      return it->second == State::Helper;
    }
    bool helper = callee.isPrivate() && !callee.isExternal() &&
                  callee.getBody().hasOneBlock() && !isMarkedNoInline(callee) &&
                  llvm::all_of(callee.getArgumentTypes(), isShapeScalar) &&
                  llvm::all_of(callee.getResultTypes(), isShapeScalar);
    SmallVector<func::CallOp> calls;
    if (helper) {
      callee.walk([&](Operation *op) {
        if (op == callee || isa<func::ReturnOp>(op)) {
          return WalkResult::advance();
        }
        if (auto call = dyn_cast<func::CallOp>(op)) {
          if (getHelper(call)) {
            calls.push_back(call);
            return WalkResult::advance();
          }
        } else if (isPure(op)) {
          return WalkResult::advance();
        }
        helper = false;
        return WalkResult::interrupt();
      });
    }
    states[callee] = helper ? State::Helper : State::NotHelper;
    if (helper) {
      for (func::CallOp call : calls) {
        inlineIfHelper(call);
      }
    }
    return helper;
  }

  void inlineIfHelper(func::CallOp call) {
    func::FuncOp callee = getHelper(call);
    if (!callee) {
      return;
    }
    InlinerConfig config;
    if (succeeded(inlineCall(interface, config.getCloneCallback(), call, callee,
                             &callee.getBody()))) {
      call.erase();
    }
  }

  SymbolTableCollection symbolTables;
  InlinerInterface interface;
  llvm::DenseMap<Operation *, State> states;
};

struct InlineShapeAssertionHelpers final
    : impl::InlineShapeAssertionHelpersBase<InlineShapeAssertionHelpers> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<mlir::stablehlo::CustomCallOp> assertions;
    module.walk([&](mlir::stablehlo::CustomCallOp op) {
      if (op.getCallTargetName() == "shape_assertion") {
        assertions.push_back(op);
      }
    });
    Inliner inliner(module);
    for (mlir::stablehlo::CustomCallOp assertion : assertions) {
      inliner.inlineProducers(assertion);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::stablehlo
