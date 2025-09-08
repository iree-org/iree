// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Regex.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_INSERTDEBUGTARGETATORDINALPASS
#define GEN_PASS_DEF_INSERTDEBUGTARGETATSYMBOLPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

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
  IREE::Flow::TensorTraceOp::create(
      builder, dispatchOp.getLoc(), builder.getStringAttr(name + " inputs"),
      filterNonTensorValues(dispatchOp.getArguments()));

  // Output tensors:
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(dispatchOp);
  IREE::Flow::TensorTraceOp::create(
      builder, dispatchOp.getLoc(), builder.getStringAttr(name + " outputs"),
      filterNonTensorValues(dispatchOp.getResults()));
}

// Breaks the given function on the specified op by simply returning immediately
// after the op. Updates the function signature to match the return type of the
// target operation.
static LogicalResult replaceReturnWithOpResults(mlir::ModuleOp moduleOp,
                                                IREE::Util::FuncOp funcOp,
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
      auto exportOp = IREE::HAL::TensorExportOp::create(
          builder, loc, type, retVal, TypeAttr::get(retVal.getType()),
          /*name=*/nullptr,
          /*affinity=*/nullptr);
      exports.push_back(exportOp.getResult());
      newTypes.push_back(type);
    } else {
      exports.push_back(retVal);
      newTypes.push_back(retVal.getType());
    }
  }

  // Create the new return and update the function type.
  IRRewriter rewriter(builder);
  rewriter.replaceOpWithNewOp<IREE::Util::ReturnOp>(oldTerminator, exports);

  SmallVector<Type> argTypes;
  for (const auto &arg : llvm::enumerate(funcOp.getArguments()))
    argTypes.push_back(arg.value().getType());

  funcOp.setType(FunctionType::get(context,
                                   /*inputs=*/argTypes, /*results=*/newTypes));
  funcOp.removeTiedOperandsAttr();
  return success();
}

namespace {

// Insert break/tracing by ordinal for the specified function.
struct InsertDebugTargetAtOrdinalPass
    : public IREE::Flow::impl::InsertDebugTargetAtOrdinalPassBase<
          InsertDebugTargetAtOrdinalPass> {
  using IREE::Flow::impl::InsertDebugTargetAtOrdinalPassBase<
      InsertDebugTargetAtOrdinalPass>::InsertDebugTargetAtOrdinalPassBase;
  void runOnOperation() override {
    auto [breakFname, breakOrdinal] =
        getOrdinalFromDebugTarget(breakDebugTarget);
    auto [traceFname, traceOrdinal] =
        getOrdinalFromDebugTarget(traceDebugTarget);

    bool foundBreakFunc = breakFname.empty();
    bool foundTraceFunc = traceFname.empty();
    for (auto it :
         llvm::enumerate(getOperation().getOps<mlir::FunctionOpInterface>())) {
      mlir::FunctionOpInterface op = it.value();
      Operation *operation = op;

      // Only look for dispatches in util func ops.
      auto funcOp = llvm::dyn_cast<IREE::Util::FuncOp>(operation);
      if (!funcOp)
        continue;

      std::string fName = funcOp.getName().str();
      if (fName != breakFname && fName != traceFname)
        continue;

      int localBreakOrdinal = -1;
      if (fName == breakFname) {
        localBreakOrdinal = breakOrdinal;
        foundBreakFunc = true;
      }
      int localTraceOrdinal = -1;
      if (fName == traceFname) {
        foundTraceFunc = true;
        localTraceOrdinal = traceOrdinal;
      }

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

    if (!foundBreakFunc || !foundTraceFunc) {
      getOperation()->emitError()
          << "Failed to find breaking or tracing target function";
      return signalPassFailure();
    }
  }
};

// Break/trace by symbol, after outlining dispatch regions and
// deduplication.
struct InsertDebugTargetAtSymbolPass
    : public IREE::Flow::impl::InsertDebugTargetAtSymbolPassBase<
          InsertDebugTargetAtSymbolPass> {
  using IREE::Flow::impl::InsertDebugTargetAtSymbolPassBase<
      InsertDebugTargetAtSymbolPass>::InsertDebugTargetAtSymbolPassBase;
  void runOnOperation() override {
    bool foundBreakFunc = true;
    bool foundTraceFunc = true;

    // Setup regex for matching symbol names.
    llvm::Regex traceMatcher;
    if (!traceDebugTarget.empty()) {
      foundTraceFunc = false;
      traceMatcher = llvm::Regex(traceDebugTarget);
    }

    llvm::Regex breakMatcher;
    if (!breakDebugTarget.empty()) {
      foundBreakFunc = false;
      breakMatcher = llvm::Regex(breakDebugTarget);
    }

    for (auto it :
         llvm::enumerate(getOperation().getOps<mlir::FunctionOpInterface>())) {
      mlir::FunctionOpInterface funcOp = it.value();

      // Find the target dispatch to break on and trace on all matching
      // dispatches.
      IREE::Flow::DispatchOp breakTarget;
      funcOp.walk([&](IREE::Flow::DispatchOp dispatchOp) {
        std::string entryPointName = dispatchOp.getEntryPointName();
        if (traceMatcher.match(entryPointName)) {
          foundTraceFunc = true;
          traceOpWithName(dispatchOp, entryPointName);
        }
        if (!breakTarget && breakMatcher.match(entryPointName)) {
          foundBreakFunc = true;
          breakTarget = dispatchOp;
        }
      });

      // Break on the selected operation (dispatch). Currently this breaks on
      // the first occurance of a dispatch that matches the symbol by assuming
      // no control flow within the function. This will fail if the target
      // dispatch is not found within the entry block of the function.
      if (breakTarget) {
        Operation *operation = funcOp;
        auto mlirFuncOp = dyn_cast<IREE::Util::FuncOp>(operation);
        if (!mlirFuncOp || failed(replaceReturnWithOpResults(
                               getOperation(), mlirFuncOp, breakTarget)))
          return signalPassFailure();
      }
    }

    if (!foundBreakFunc || !foundTraceFunc) {
      getOperation()->emitError()
          << "Failed to find any breaking or tracing target dispatch";
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
