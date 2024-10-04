// Copyright 2024 The IREE Authors
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
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-annotate-dispatch-signatures"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ANNOTATEDISPATCHSIGNATURESPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

class ArgumentAnalysis {
public:
  explicit ArgumentAnalysis(Operation *rootOp)
      : explorer(rootOp, TraversalAction::SHALLOW),
        solver(explorer, allocator) {
    explorer.setOpInterfaceAction<mlir::FunctionOpInterface>(
        TraversalAction::RECURSE);
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

  LogicalResult run() {
    // Seed all dispatch arguments we want to analyze.
    for (auto it : entryDispatchMap) {
      for (auto dispatchOp : it.second) {
        for (auto operand : dispatchOp.getUniformOperands()) {
          (void)operand;
          // solver.getOrCreateElementFor<ValuePVS>(Position::forValue(operand));
          // solver.getOrCreateElementFor<ValueAlignment>(
          //     Position::forValue(operand));
        }
        // for (auto resourceOffset : dispatchOp.getResourceOffsets()) {
        //   solver.getOrCreateElementFor<ValueAlignment>(
        //       Position::forValue(resourceOffset));
        // }
      }
    }

    // Run solver to completion.
    return solver.run();
  }

private:
  Explorer explorer;
  llvm::BumpPtrAllocator allocator;
  DFX::Solver solver;

  DenseMap<Operation *, SmallVector<IREE::Stream::CmdDispatchOp>>
      entryDispatchMap;
};

} // namespace

struct AnnotateDispatchSignaturesPass
    : public IREE::Stream::impl::AnnotateDispatchSignaturesPassBase<
          AnnotateDispatchSignaturesPass> {
  void runOnOperation() override {
    ArgumentAnalysis analysis(getOperation());
    if (failed(analysis.run())) {
      return signalPassFailure();
    }
  }
};

} // namespace mlir::iree_compiler::IREE::Stream
