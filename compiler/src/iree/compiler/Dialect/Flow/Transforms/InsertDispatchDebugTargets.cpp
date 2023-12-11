// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Regex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Flow {

// Filters out non-tensor values for tracing.
static SmallVector<Value> filterNonTensorValues(ValueRange &&range) {
  SmallVector<Value> result;
  for (auto value : range) {
    if (llvm::isa<TensorType>(value.getType()))
      result.push_back(value);
  }
  return result;
}

// Attempts to interpret a pass arg as @<function_name>:<ordinal>, else returns
// a negative ordinal indicating no match.
static std::tuple<std::string, int>
getOrdinalFromDebugTarget(std::string marker) {
  if (marker.empty() || marker[0] != '@')
    return std::make_tuple("", -1);

  SmallVector<StringRef, 2> parts;
  auto cropped = marker.substr(1);
  llvm::SplitString(llvm::StringRef(cropped), parts, ":");
  if (parts.size() != 2)
    return std::make_tuple("", -1);

  int ordinal;
  if (parts[1].getAsInteger(10, ordinal))
    return std::make_tuple("", -1);

  return std::make_tuple(parts[0].str(), ordinal);
}

// Inserts flow.tensor.trace ops around the specified dispatch op.
static void traceOpWithName(IREE::Flow::DispatchOp dispatchOp,
                            std::string name) {
  OpBuilder builder(dispatchOp);
  // Input tensors:
  builder.create<IREE::Flow::TensorTraceOp>(
      dispatchOp.getLoc(), builder.getStringAttr(name + " inputs"),
      filterNonTensorValues(dispatchOp.getArguments()));

  // Output tensors:
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(dispatchOp);
  builder.create<IREE::Flow::TensorTraceOp>(
      dispatchOp.getLoc(), builder.getStringAttr(name + " outputs"),
      filterNonTensorValues(dispatchOp.getResults()));
}

// Breaks the given function on the specified op by simply returning immediately
// after the op. Updates the function signature to match the return type of the
// target operation.
static LogicalResult replaceReturnWithOpResults(mlir::ModuleOp moduleOp,
                                                mlir::func::FuncOp funcOp,
                                                Operation *op) {
  if (!funcOp->isProperAncestor(op))
    return failure();

  // TODO: Handle nested function calls.
  if (!SymbolTable::symbolKnownUseEmpty(funcOp, moduleOp))
    return failure();

  // TODO: Handle (nested) control flow.
  auto funcBlock = op->getBlock();
  if (funcBlock->getParentOp() != funcOp ||
      &funcOp.getBody().front() != funcBlock)
    return failure();

  // Collect the op results and create export ops for any tensor results.
  OpBuilder builder(funcOp);
  auto context = op->getContext();
  auto loc = op->getLoc();
  auto oldTerminator = funcBlock->getTerminator();
  builder.setInsertionPoint(oldTerminator);
  SmallVector<Value> exports;
  SmallVector<Type> newTypes;
  for (auto retVal : op->getResults()) {
    if (llvm::isa<TensorType>(retVal.getType())) {
      auto type = IREE::HAL::BufferViewType::get(context);
      auto exportOp = builder.create<IREE::HAL::TensorExportOp>(
          loc, type, retVal, TypeAttr::get(retVal.getType()), /*name=*/nullptr);
      exports.push_back(exportOp.getResult());
      newTypes.push_back(type);
    } else {
      exports.push_back(retVal);
      newTypes.push_back(retVal.getType());
    }
  }

  // Create the new return and update the function type.
  IRRewriter rewriter(builder);
  rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(oldTerminator, exports);

  SmallVector<Type> argTypes;
  for (const auto &arg : llvm::enumerate(funcOp.getArguments()))
    argTypes.push_back(arg.value().getType());

  funcOp.setType(FunctionType::get(context,
                                   /*inputs=*/argTypes, /*results=*/newTypes));
  return success();
}

// Insert break/tracing by ordinal for the specified function.
struct InsertDebugTargetAtOrdinalPass
    : public InsertDebugTargetAtOrdinalBase<InsertDebugTargetAtOrdinalPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, IREE::HAL::HALDialect>();
  }
  InsertDebugTargetAtOrdinalPass(std::string breakStr, std::string traceStr) {
    this->breakDebugTarget = breakStr;
    this->traceDebugTarget = traceStr;
  };
  InsertDebugTargetAtOrdinalPass(const InsertDebugTargetAtOrdinalPass &pass)
      : InsertDebugTargetAtOrdinalPass(pass.breakDebugTarget,
                                       pass.traceDebugTarget) {}

  void runOnOperation() override {
    auto [breakFname, breakOrdinal] =
        getOrdinalFromDebugTarget(breakDebugTarget);
    auto [traceFname, traceOrdinal] =
        getOrdinalFromDebugTarget(traceDebugTarget);

    for (auto it :
         llvm::enumerate(getOperation().getOps<FunctionOpInterface>())) {
      FunctionOpInterface op = it.value();
      Operation *operation = op;

      // Only look for dispatches in upstream func ops.
      auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(operation);
      if (!funcOp)
        continue;

      std::string fName = funcOp.getName().str();
      int localBreakOrdinal = -1;
      if (fName == breakFname)
        localBreakOrdinal = breakOrdinal;
      int localTraceOrdinal = -1;
      if (fName == traceFname)
        localTraceOrdinal = traceOrdinal;

      auto &bodyRegion = op.getFunctionBody();
      auto dispatchOps =
          llvm::to_vector<8>(bodyRegion.getOps<IREE::Flow::DispatchOp>());

      // Trace on a valid ordinal.
      if (localTraceOrdinal >= 0 && localTraceOrdinal < dispatchOps.size()) {
        auto traceTarget = dispatchOps[localTraceOrdinal];
        // Append the ordinal to the trace name.
        std::string entryPointName = traceTarget.getEntryPointName();
        traceOpWithName(traceTarget, entryPointName + std::string("::") +
                                         std::to_string(localTraceOrdinal));
      }

      // Break on a valid ordinal, updating the function signature in the
      // process. Currently only a single ordinal is supported so no need to
      // check for overlapping breaks.
      if (localBreakOrdinal >= 0 && localBreakOrdinal < dispatchOps.size()) {
        auto breakTarget = dispatchOps[localBreakOrdinal];
        if (failed(replaceReturnWithOpResults(getOperation(), funcOp,
                                              breakTarget)))
          return signalPassFailure();
      }
    }
  }
};

// Break/trace by symbol, after outlining dispatch regions and
// deduplication.
struct InsertDebugTargetAtSymbolPass
    : public InsertDebugTargetAtSymbolBase<InsertDebugTargetAtSymbolPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, IREE::HAL::HALDialect>();
  }
  InsertDebugTargetAtSymbolPass(std::string breakStr, std::string traceStr) {
    this->breakDebugTarget = breakStr;
    this->traceDebugTarget = traceStr;
  };
  InsertDebugTargetAtSymbolPass(const InsertDebugTargetAtSymbolPass &pass)
      : InsertDebugTargetAtSymbolPass(pass.breakDebugTarget,
                                      pass.traceDebugTarget) {}

  void runOnOperation() override {
    // Setup regex for matching symbol names.
    llvm::Regex traceMatcher;
    if (!traceDebugTarget.empty())
      traceMatcher = llvm::Regex(traceDebugTarget);

    llvm::Regex breakMatcher;
    if (!breakDebugTarget.empty())
      breakMatcher = llvm::Regex(breakDebugTarget);

    for (auto it :
         llvm::enumerate(getOperation().getOps<FunctionOpInterface>())) {
      FunctionOpInterface funcOp = it.value();

      // Find the target dispatch to break on and trace on all matching
      // dispatches.
      IREE::Flow::DispatchOp breakTarget;
      funcOp.walk([&](IREE::Flow::DispatchOp dispatchOp) {
        std::string entryPointName = dispatchOp.getEntryPointName();
        if (traceMatcher.match(entryPointName))
          traceOpWithName(dispatchOp, entryPointName);
        if (!breakTarget && breakMatcher.match(entryPointName))
          breakTarget = dispatchOp;
      });

      // Break on the selected operation (dispatch). Currently this breaks on
      // the first occurance of a dispatch that matches the symbol by assuming
      // no control flow within the function. This will fail if the target
      // dispatch is not found within the entry block of the function.
      if (breakTarget) {
        Operation *operation = funcOp;
        auto mlirFuncOp = dyn_cast<mlir::func::FuncOp>(operation);
        if (!mlirFuncOp || failed(replaceReturnWithOpResults(
                               getOperation(), mlirFuncOp, breakTarget)))
          return signalPassFailure();
      }
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createInsertDebugTargetAtOrdinalPass(std::string breakDebugTarget,
                                     std::string traceDebugTarget) {
  return std::make_unique<InsertDebugTargetAtOrdinalPass>(breakDebugTarget,
                                                          traceDebugTarget);
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createInsertDebugTargetAtSymbolPass(std::string breakDebugTarget,
                                    std::string traceDebugTarget) {
  return std::make_unique<InsertDebugTargetAtSymbolPass>(breakDebugTarget,
                                                         traceDebugTarget);
}

} // namespace mlir::iree_compiler::IREE::Flow
