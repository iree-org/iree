// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUPROPAGATEDISPATCHSTATUSPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

constexpr StringLiteral kReturnsStatusAttr = "iree.returns_status";

} // namespace

// Thread failures through private calls to the dispatch ABI's status result.
static LogicalResult propagateExecutableStatus(ModuleOp moduleOp) {
  llvm::SetVector<Operation *> failingFunctions;
  WalkResult walk = moduleOp.walk([&](IREE::Util::StatusCheckOkOp check) {
    auto function = check->getParentOfType<func::FuncOp>();
    if (!function) {
      check.emitError("expected semantic failure inside a func.func");
      return WalkResult::interrupt();
    }
    failingFunctions.insert(function);
    return WalkResult::advance();
  });
  if (walk.wasInterrupted()) {
    return failure();
  }
  if (failingFunctions.empty()) {
    return success();
  }

  SymbolTableCollection symbols;
  bool changed;
  do {
    changed = false;
    moduleOp.walk([&](func::CallOp call) {
      auto callee = symbols.lookupNearestSymbolFrom<func::FuncOp>(
          call, call.getCalleeAttr());
      if (callee && failingFunctions.contains(callee)) {
        auto caller = call->getParentOfType<func::FuncOp>();
        if (caller) {
          changed |= failingFunctions.insert(caller);
        }
      }
    });
  } while (changed);

  for (Operation *operation : failingFunctions) {
    auto function = cast<func::FuncOp>(operation);
    if (function.isPublic() &&
        (function.getNumArguments() || function.getNumResults())) {
      return function.emitError("dispatch entry point must have type () -> ()");
    }
    auto uses = SymbolTable::getSymbolUses(function, moduleOp);
    if (!uses) {
      return function.emitError("cannot resolve failure propagation uses");
    }
    for (const auto &use : *uses) {
      // Dispatch metadata names the entry point without calling its ABI.
      if (auto config =
              dyn_cast<IREE::Codegen::DispatchConfigOp>(use.getUser());
          config && function.isPublic() &&
          use.getSymbolRef() == config.getFunctionRefAttr()) {
        continue;
      }
      auto call = dyn_cast<func::CallOp>(use.getUser());
      if (!call || function.isPublic() ||
          !failingFunctions.contains(call->getParentOfType<func::FuncOp>())) {
        return function.emitError(
            "semantic failure requires direct calls within the executable");
      }
    }
  }

  // A check must be able to exit the function from inside a structured region.
  RewritePatternSet patterns(moduleOp.getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  ConversionTarget target(*moduleOp.getContext());
  target.addIllegalDialect<affine::AffineDialect, scf::SCFDialect>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  for (Operation *function : failingFunctions) {
    if (failed(applyPartialConversion(function, target, frozenPatterns))) {
      return failure();
    }
  }

  walk = moduleOp.walk([&](IREE::Util::StatusCheckOkOp check) {
    auto function = check->getParentOfType<func::FuncOp>();
    if (check->getParentRegion() != &function.getBody()) {
      check.emitError("semantic failure requires function-level control flow");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walk.wasInterrupted()) {
    return failure();
  }

  OpBuilder builder(moduleOp.getContext());
  Type statusType = builder.getI32Type();
  for (Operation *operation : failingFunctions) {
    auto function = cast<func::FuncOp>(operation);
    SmallVector<Type> resultTypes(function.getResultTypes());
    resultTypes.push_back(statusType);
    function.setType(
        builder.getFunctionType(function.getArgumentTypes(), resultTypes));
    // Failed calls return unused poison values, so result guarantees no longer
    // apply unconditionally. Argument contracts are unchanged.
    function.removeResAttrsAttr();
    if (function.isPublic()) {
      function->setAttr(kReturnsStatusAttr, builder.getUnitAttr());
    }
    function.walk([&](func::ReturnOp returnOp) {
      builder.setInsertionPoint(returnOp);
      Value ok =
          arith::ConstantIntOp::create(builder, returnOp.getLoc(), 0, 32);
      returnOp->insertOperands(returnOp.getNumOperands(), ok);
    });
  }
  SmallVector<func::CallOp> calls;
  moduleOp.walk([&](func::CallOp call) {
    auto callee = symbols.lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    if (callee && failingFunctions.contains(callee)) {
      calls.push_back(call);
    }
  });
  for (func::CallOp call : calls) {
    builder.setInsertionPoint(call);
    SmallVector<Type> resultTypes(call.getResultTypes());
    resultTypes.push_back(statusType);
    auto replacement =
        func::CallOp::create(builder, call.getLoc(), call.getCallee(),
                             resultTypes, call.getOperands());
    replacement->setAttrs(call->getAttrs());
    call.replaceAllUsesWith(replacement.getResults().drop_back());
    IREE::Util::StatusCheckOkOp::create(builder, call.getLoc(),
                                        replacement.getResults().back());
    call.erase();
  }
  SmallVector<IREE::Util::StatusCheckOkOp> checks;
  moduleOp.walk(
      [&](IREE::Util::StatusCheckOkOp check) { checks.push_back(check); });
  for (auto check : checks) {
    auto function = check->getParentOfType<func::FuncOp>();
    Block *block = check->getBlock();
    Block *continuation = block->splitBlock(check->getIterator());
    builder.setInsertionPointToEnd(block);
    Value zero = arith::ConstantIntOp::create(builder, check.getLoc(), 0, 32);
    Value failed =
        arith::CmpIOp::create(builder, check.getLoc(), arith::CmpIPredicate::ne,
                              check.getStatus(), zero);
    Block *failureBlock = builder.createBlock(&function.getBody());
    SmallVector<Value> results;
    for (Type type : function.getResultTypes().drop_back()) {
      results.push_back(ub::PoisonOp::create(builder, check.getLoc(), type));
    }
    results.push_back(check.getStatus());
    func::ReturnOp::create(builder, check.getLoc(), results);
    builder.setInsertionPointToEnd(block);
    cf::CondBranchOp::create(builder, check.getLoc(), failed, failureBlock,
                             ValueRange{}, continuation, ValueRange{});
    check.erase();
  }
  return success();
}

namespace {

struct LLVMCPUPropagateDispatchStatusPass final
    : impl::LLVMCPUPropagateDispatchStatusPassBase<
          LLVMCPUPropagateDispatchStatusPass> {
  void runOnOperation() override {
    if (failed(propagateExecutableStatus(getOperation()))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
