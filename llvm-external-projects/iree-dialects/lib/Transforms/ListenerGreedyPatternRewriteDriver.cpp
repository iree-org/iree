// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Transforms/ListenerGreedyPatternRewriteDriver.h"

#include "iree-dialects/Transforms/Listener.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "listener-greedy-rewriter"

//===----------------------------------------------------------------------===//
// GreedyPatternRewriteDriver
//===----------------------------------------------------------------------===//

namespace {
/// This is a worklist-driven driver for the PatternMatcher, which repeatedly
/// applies the locally optimal patterns in a roughly "bottom up" way.
class GreedyPatternRewriteDriver : public RewriteListener {
public:
  explicit GreedyPatternRewriteDriver(Operation *rootOp,
                                      const FrozenRewritePatternSet &patterns,
                                      const GreedyRewriteConfig &config,
                                      RewriteListener *listener);
  //===--------------------------------------------------------------------===//
  // BEGIN copied from mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
  //===--------------------------------------------------------------------===//

  /// Simplify the operations within the given regions.
  bool simplify(MutableArrayRef<Region> regions);

  /// Add the given operation to the worklist.
  void addToWorklist(Operation *op);

  /// Pop the next operation from the worklist.
  Operation *popFromWorklist();

  /// If the specified operation is in the worklist, remove it.
  void removeFromWorklist(Operation *op);

protected:
  // Implement the hook for inserting operations, and make sure that newly
  // inserted ops are added to the worklist for processing.
  void notifyOperationInserted(Operation *op) override;

  void notifyOperationModified(Operation *op) override;

  // Look over the provided operands for any defining operations that should
  // be re-added to the worklist. This function should be called when an
  // operation is modified or removed, as it may trigger further
  // simplifications.
  void addOperandsToWorklist(ValueRange operands);

  // If an operation is about to be removed, make sure it is not in our
  // worklist anymore because we'd get dangling references to it.
  void notifyOperationRemoved(Operation *op) override;

  // When the root of a pattern is about to be replaced, it can trigger
  // simplifications to its users - make sure to add them to the worklist
  // before the root is changed.
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override;

  //===--------------------------------------------------------------------===//
  // END copied from mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
  //===--------------------------------------------------------------------===//
  // This seems unused
  /// PatternRewriter hook for erasing a dead operation.
  // void eraseOp(Operation *op) override;
  // //===-----------------------------------------------------------------===//
  // BEGIN copied from mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
  //===--------------------------------------------------------------------===//

  /// PatternRewriter hook for notifying match failure reasons.
  LogicalResult
  notifyMatchFailure(Location loc,
                     function_ref<void(Diagnostic &)> reasonCallback) override;

  /// The low-level pattern applicator.
  PatternApplicator matcher;

  /// The worklist for this transformation keeps track of the operations that
  /// need to be revisited, plus their index in the worklist.  This allows us to
  /// efficiently remove operations from the worklist when they are erased, even
  /// if they aren't the root of a pattern.
  std::vector<Operation *> worklist;
  DenseMap<Operation *, unsigned> worklistMap;

  /// Non-pattern based folder for operations.
  OperationFolder folder;

private:
  /// Configuration information for how to simplify.
  GreedyRewriteConfig config;

  //===--------------------------------------------------------------------===//
  // END copied from mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
  //===--------------------------------------------------------------------===//
  /// The pattern rewriter to use.
  PatternRewriterListener rewriter;
  /// The operation under which all processed ops must be nested.
  Operation *rootOp;
  //===--------------------------------------------------------------------===//
  // BEGIN copied from mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
  //===--------------------------------------------------------------------===//

#ifndef NDEBUG
  /// A logger used to emit information during the application process.
  llvm::ScopedPrinter logger{llvm::dbgs()};
#endif
};
} // namespace

//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
//===----------------------------------------------------------------------===//
GreedyPatternRewriteDriver::GreedyPatternRewriteDriver(
    Operation *rootOp, const FrozenRewritePatternSet &patterns,
    const GreedyRewriteConfig &config, RewriteListener *listener)
    : matcher(patterns), folder(rootOp->getContext()), config(config),
      rewriter(rootOp->getContext()), rootOp(rootOp) {
  // Add self as a listener and the user-provided listener.
  rewriter.addListener(this);
  if (listener)
    rewriter.addListener(listener);
  //===--------------------------------------------------------------------===//
  // BEGIN copied from mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
  //===--------------------------------------------------------------------===//

  worklist.reserve(64);

  // Apply a simple cost model based solely on pattern benefit.
  matcher.applyDefaultCostModel();
}

bool GreedyPatternRewriteDriver::simplify(MutableArrayRef<Region> regions) {
#ifndef NDEBUG
  const char *logLineComment =
      "//===-------------------------------------------===//\n";

  /// A utility function to log a process result for the given reason.
  auto logResult = [&](StringRef result, const llvm::Twine &msg = {}) {
    logger.unindent();
    logger.startLine() << "} -> " << result;
    if (!msg.isTriviallyEmpty())
      logger.getOStream() << " : " << msg;
    logger.getOStream() << "\n";
  };
  auto logResultWithLine = [&](StringRef result, const llvm::Twine &msg = {}) {
    logResult(result, msg);
    logger.startLine() << logLineComment;
  };
#endif

  auto insertKnownConstant = [&](Operation *op) {
    // Check for existing constants when populating the worklist. This avoids
    // accidentally reversing the constant order during processing.
    Attribute constValue;
    if (matchPattern(op, m_Constant(&constValue)))
      if (!folder.insertKnownConstant(op, constValue))
        return true;
    return false;
  };

  bool changed = false;
  unsigned iteration = 0;
  do {
    worklist.clear();
    worklistMap.clear();

    if (!config.useTopDownTraversal) {
      // Add operations to the worklist in postorder.
      for (auto &region : regions) {
        region.walk([&](Operation *op) {
          if (!insertKnownConstant(op))
            addToWorklist(op);
        });
      }
    } else {
      // Add all nested operations to the worklist in preorder.
      for (auto &region : regions) {
        region.walk<WalkOrder::PreOrder>([&](Operation *op) {
          if (!insertKnownConstant(op)) {
            worklist.push_back(op);
            return WalkResult::advance();
          }
          return WalkResult::skip();
        });
      }

      // Reverse the list so our pop-back loop processes them in-order.
      std::reverse(worklist.begin(), worklist.end());
      // Remember the reverse index.
      for (size_t i = 0, e = worklist.size(); i != e; ++i)
        worklistMap[worklist[i]] = i;
    }

    // These are scratch vectors used in the folding loop below.
    SmallVector<Value, 8> originalOperands, resultValues;

    changed = false;
    while (!worklist.empty()) {
      auto *op = popFromWorklist();

      // Nulls get added to the worklist when operations are removed, ignore
      // them.
      if (op == nullptr)
        continue;

      LLVM_DEBUG({
        logger.getOStream() << "\n";
        logger.startLine() << logLineComment;
        logger.startLine() << "Processing operation : '" << op->getName()
                           << "'(" << op << ") {\n";
        logger.indent();

        // If the operation has no regions, just print it here.
        if (op->getNumRegions() == 0) {
          op->print(
              logger.startLine(),
              OpPrintingFlags().printGenericOpForm().elideLargeElementsAttrs());
          logger.getOStream() << "\n\n";
        }
      });

      // If the operation is trivially dead - remove it.
      if (isOpTriviallyDead(op)) {
        notifyOperationRemoved(op);
        op->erase();
        changed = true;

        LLVM_DEBUG(logResultWithLine("success", "operation is trivially dead"));
        continue;
      }

      // Collects all the operands and result uses of the given `op` into work
      // list. Also remove `op` and nested ops from worklist.
      originalOperands.assign(op->operand_begin(), op->operand_end());
      auto preReplaceAction = [&](Operation *op) {
        // Add the operands to the worklist for visitation.
        addOperandsToWorklist(originalOperands);

        // Add all the users of the result to the worklist so we make sure
        // to revisit them.
        for (auto result : op->getResults())
          for (auto *userOp : result.getUsers())
            addToWorklist(userOp);

        notifyOperationRemoved(op);
      };

      // Add the given operation to the worklist.
      auto collectOps = [this](Operation *op) { addToWorklist(op); };

      // Try to fold this op.
      bool inPlaceUpdate;
      if ((succeeded(folder.tryToFold(op, collectOps, preReplaceAction,
                                      &inPlaceUpdate)))) {
        LLVM_DEBUG(logResultWithLine("success", "operation was folded"));

        changed = true;
        if (!inPlaceUpdate)
          continue;
      }

      // Try to match one of the patterns. The rewriter is automatically
      // notified of any necessary changes, so there is nothing else to do
      // here.
#ifndef NDEBUG
      auto canApply = [&](const Pattern &pattern) {
        LLVM_DEBUG({
          logger.getOStream() << "\n";
          logger.startLine() << "* Pattern " << pattern.getDebugName() << " : '"
                             << op->getName() << " -> (";
          llvm::interleaveComma(pattern.getGeneratedOps(), logger.getOStream());
          logger.getOStream() << ")' {\n";
          logger.indent();
        });
        return true;
      };
      auto onFailure = [&](const Pattern &pattern) {
        LLVM_DEBUG(logResult("failure", "pattern failed to match"));
      };
      auto onSuccess = [&](const Pattern &pattern) {
        LLVM_DEBUG(logResult("success", "pattern applied successfully"));
        return success();
      };

      LogicalResult matchResult =
          //===------------------------------------------------------------===//
          // BEGIN single line change from
          // mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
          // END
          //===------------------------------------------------------------===//
          matcher.matchAndRewrite(op, rewriter, canApply, onFailure, onSuccess);
      if (succeeded(matchResult))
        LLVM_DEBUG(logResultWithLine("success", "pattern matched"));
      else
        LLVM_DEBUG(logResultWithLine("failure", "pattern failed to match"));
#else
      //===----------------------------------------------------------------===//
      // BEGIN single line change from
      // mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
      // END
      //===----------------------------------------------------------------===//
      LogicalResult matchResult = matcher.matchAndRewrite(op, rewriter);
#endif
      changed |= succeeded(matchResult);
    }

    // After applying patterns, make sure that the CFG of each of the regions
    // is kept up to date.
    if (config.enableRegionSimplification)
      //===----------------------------------------------------------------===//
      // BEGIN single line change from
      // mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
      // END
      //===----------------------------------------------------------------===//
      changed |= succeeded(simplifyRegions(rewriter, regions));
  } while (changed && (iteration++ < config.maxIterations ||
                       config.maxIterations == GreedyRewriteConfig::kNoLimit));

  // Whether the rewrite converges, i.e. wasn't changed in the last iteration.
  return !changed;
}

void GreedyPatternRewriteDriver::addToWorklist(Operation *op) {
  // Check to see if the worklist already contains this op.
  if (worklistMap.count(op))
    return;
  //===--------------------------------------------------------------------===//
  // END copied from mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
  //===--------------------------------------------------------------------===//
  // Enforce nested under constraint before adding to worklist.
  if (!rootOp->isProperAncestor(op))
    return;
  //===--------------------------------------------------------------------===//
  // BEGIN copied from mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
  //===--------------------------------------------------------------------===//

  worklistMap[op] = worklist.size();
  worklist.push_back(op);
}

Operation *GreedyPatternRewriteDriver::popFromWorklist() {
  auto *op = worklist.back();
  worklist.pop_back();

  // This operation is no longer in the worklist, keep worklistMap up to date.
  if (op)
    worklistMap.erase(op);
  return op;
}

void GreedyPatternRewriteDriver::removeFromWorklist(Operation *op) {
  auto it = worklistMap.find(op);
  if (it != worklistMap.end()) {
    assert(worklist[it->second] == op && "malformed worklist data structure");
    worklist[it->second] = nullptr;
    worklistMap.erase(it);
  }
}

void GreedyPatternRewriteDriver::notifyOperationInserted(Operation *op) {
  LLVM_DEBUG({
    logger.startLine() << "** Insert  : '" << op->getName() << "'(" << op
                       << ")\n";
  });
  addToWorklist(op);
}

void GreedyPatternRewriteDriver::notifyOperationModified(Operation *op) {
  LLVM_DEBUG({
    logger.startLine() << "** Modified: '" << op->getName() << "'(" << op
                       << ")\n";
  });
  addToWorklist(op);
}

void GreedyPatternRewriteDriver::addOperandsToWorklist(ValueRange operands) {
  for (Value operand : operands) {
    // If the use count of this operand is now < 2, we re-add the defining
    // operation to the worklist.
    // TODO: This is based on the fact that zero use operations
    // may be deleted, and that single use values often have more
    // canonicalization opportunities.
    if (!operand || (!operand.use_empty() && !operand.hasOneUse()))
      continue;
    if (auto *defOp = operand.getDefiningOp())
      addToWorklist(defOp);
  }
}

void GreedyPatternRewriteDriver::notifyOperationRemoved(Operation *op) {
  addOperandsToWorklist(op->getOperands());
  op->walk([this](Operation *operation) {
    removeFromWorklist(operation);
    folder.notifyRemoval(operation);
  });
}

void GreedyPatternRewriteDriver::notifyOperationReplaced(
    Operation *op, ValueRange replacement) {
  LLVM_DEBUG({
    logger.startLine() << "** Replace : '" << op->getName() << "'(" << op
                       << ")\n";
  });
  for (auto result : op->getResults())
    for (auto *user : result.getUsers())
      addToWorklist(user);
}

//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
//===----------------------------------------------------------------------===//
// This seems unused
// void GreedyPatternRewriteDriver::eraseOp(Operation *op) {
//   LLVM_DEBUG({
//     logger.startLine() << "** Erase   : '" << op->getName() << "'(" << op
//                        << ")\n";
//   });
//   PatternRewriter::eraseOp(op);
// }
//===----------------------------------------------------------------------===//
// BEGIN copied from mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
//===----------------------------------------------------------------------===//

LogicalResult GreedyPatternRewriteDriver::notifyMatchFailure(
    Location loc, function_ref<void(Diagnostic &)> reasonCallback) {
  LLVM_DEBUG({
    Diagnostic diag(loc, DiagnosticSeverity::Remark);
    reasonCallback(diag);
    logger.startLine() << "** Failure : " << diag.str() << "\n";
  });
  return failure();
}

/// Rewrite the regions of the specified operation, which must be isolated from
/// above, by repeatedly applying the highest benefit patterns in a greedy
/// work-list driven manner. Return success if no more patterns can be matched
/// in the result operation regions. Note: This does not apply patterns to the
/// top-level operation itself.
///
//===----------------------------------------------------------------------===//
// END copied from mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
//===----------------------------------------------------------------------===//
LogicalResult mlir::applyPatternsAndFoldGreedily(
    Operation *op, const FrozenRewritePatternSet &patterns,
    const GreedyRewriteConfig &config, RewriteListener *listener) {
  if (op->getRegions().empty())
    return success();

  // Start the pattern driver.
  GreedyPatternRewriteDriver driver(op, patterns, config, listener);
  auto regions = op->getRegions();
  //===--------------------------------------------------------------------===//
  // BEGIN copied from mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
  //===--------------------------------------------------------------------===//
  bool converged = driver.simplify(regions);
  LLVM_DEBUG(if (!converged) {
    llvm::dbgs() << "The pattern rewrite doesn't converge after scanning "
                 << config.maxIterations << " times\n";
  });
  return success(converged);
}
