// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Constant/ConstExpr.h"
#include "iree/compiler/Dialect/Util/Analysis/Constant/OpOracle.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#define DEBUG_TYPE "iree-constexpr"

using llvm::dbgs;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {
namespace {

static llvm::cl::opt<std::string> clPrintDotGraphToFile(
    "iree-util-hoist-into-globals-print-constexpr-dotgraph-to",
    llvm::cl::desc(
        "Prints a dot graph representing the const-expr analysis. The red "
        "nodes represent roots and the green nodes represent hoisted values."),
    llvm::cl::value_desc("filename"));

// Maps an original value in the program to the symbol name of a global.
using HoistedValueMap = llvm::DenseMap<Value, GlobalOp>;

// Hoist expressions into globals. It is not expected that such a greedy
// algorithm is great, but it is simple. Naive use of this algorithm very likely
// favors programs that consume more memory at runtime than is strictly
// necessary. Either this algorithm can be made smarter or a follow-on pass
// can sink globals into the program where it is profitable to reduce
// working set size.
class HoistIntoGlobalsPass : public HoistIntoGlobalsBase<HoistIntoGlobalsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registerConstExprDependentDialects(registry);
  }

  HoistIntoGlobalsPass(int64_t threshold) {
    this->maxSizeIncreaseThreshold.setValue(threshold);
  }

  void runOnOperation() override {
    SymbolTable moduleSymbols(getOperation());
    const auto &constExprs = getAnalysis<ConstExprAnalysis>();
    LLVM_DEBUG(dbgs() << constExprs);
    LLVM_DEBUG(dbgs() << "\n\n");
    ConstExprHoistingPolicy policy(constExprs, this->maxSizeIncreaseThreshold);
    policy.initialize();

    // Print analysis dot graph if requested.
    if (!clPrintDotGraphToFile.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream file(clPrintDotGraphToFile, ec);
      if (ec) {
        getOperation().emitError()
            << "failed to open file for printing dot graph: " << ec.message();
        return signalPassFailure();
      }
      policy.printDotGraph(file);
      file.close();
    }

    // Maps original values to newly materialized values.
    HoistedValueMap hoistedMap;

    // Walk all operations in the program and hoist any escapes from
    // const-expr values into globals. Note that we must walk the const-exprs
    // in topological order so that corresponding initializers will be created
    // in order without depending on globals that have not been initialized
    // yet.
    getOperation().walk<WalkOrder::PreOrder>([&](Operation *iterOp) {
      // We only want to look at const-expr ops (non roots) since they may
      // have interesting escapes. Early exit here for efficiency.
      auto *iterInfo = constExprs.lookup(iterOp);
      if (!iterInfo) {
        return WalkResult::advance();
      }

      for (Value constExprResult : iterOp->getResults()) {
        auto *resultInfo = constExprs.lookup(constExprResult);
        assert(resultInfo && "must have const-expr info");

        if (policy.getDecision(resultInfo)->getOutcome() !=
            ConstExprHoistingPolicy::ENABLE_HOIST) {
          continue;
        }

        hoistConstExpr(constExprResult, hoistedMap, moduleSymbols, constExprs);
      }
      return WalkResult::advance();
    });

    // Apply any remaining RAUW cleanups. We have to do these at the cleanup
    // phase since modifying the source program can invalidate the analysis.
    // Up to this point, we have only been cloning.
    OpBuilder builder(&getContext());
    for (auto it : hoistedMap) {
      Value originalValue = it.first;
      GlobalOp globalOp = it.second;
      builder.setInsertionPointAfterValue(originalValue);
      auto load = builder.create<GlobalLoadOp>(globalOp->getLoc(), globalOp);
      originalValue.replaceAllUsesWith(load);
    }
    cleanupDeadOps(constExprs);
  }

  GlobalOp hoistConstExpr(Value originalValue, HoistedValueMap &hoistedMap,
                          SymbolTable &moduleSymbols,
                          const ConstExprAnalysis &constExprs) {
    GlobalOp existingGlobal = hoistedMap.lookup(originalValue);

    if (!existingGlobal) {
      // No existing mapping.
      Location loc = originalValue.getLoc();
      OpBuilder builder = getModuleEndBuilder();
      auto initializerOp = builder.create<InitializerOp>(loc);
      // Signals that this initializer is eligible for constant evaluation
      // at compile time.
      initializerOp->setAttr("iree.compiler.consteval", builder.getUnitAttr());
      Block *entryBlock = initializerOp.addEntryBlock();
      OpBuilder initBuilder = OpBuilder::atBlockEnd(entryBlock);
      IRMapping valueMapping;
      cloneConstExprInto(initializerOp.getLoc(), initBuilder, originalValue,
                         hoistedMap, moduleSymbols, valueMapping, constExprs);

      existingGlobal = hoistedMap.lookup(originalValue);
    }
    assert(existingGlobal &&
           "hoisting const-expr should have mapped a global for the requested "
           "value");
    return existingGlobal;
  }

  void
  cloneProducerTreeInto(OpBuilder &builder,
                        const ConstExprAnalysis::ConstValueInfo *producerInfo,
                        HoistedValueMap &hoistedMap, IRMapping &cloneMapping,
                        const ConstExprAnalysis &constExprs) {
    if (cloneMapping.contains(producerInfo->constValue))
      return;

    // We either have a global associated already or we need to traverse
    // down and materialize producers.
    GlobalOp existingGlobal = hoistedMap.lookup(producerInfo->constValue);
    if (existingGlobal) {
      cloneMapping.map(producerInfo->constValue,
                       builder.create<GlobalLoadOp>(existingGlobal.getLoc(),
                                                    existingGlobal));
      return;
    }

    // Materialize all producers recursively.
    for (auto *producerInfo : producerInfo->producers) {
      cloneProducerTreeInto(builder, producerInfo, hoistedMap, cloneMapping,
                            constExprs);
    }

    // And clone the requested op.
    Operation *sourceOp = producerInfo->constValue.getDefiningOp();
    assert(sourceOp && "must have defining op for const-expr values");
    LLVM_DEBUG(dbgs() << "    CLONE OP: " << *sourceOp << "\n");
    Operation *clonedOp = sourceOp->clone(cloneMapping);
    builder.insert(clonedOp);
  }

  // Clones the const expr tree rooted at `constExprValue` into the given
  // initializer, noting any new hoisted value mappings that result. At
  // a minimum, a mapping will be created for the requested value.
  void cloneConstExprInto(Location loc, OpBuilder &builder,
                          Value constExprValue, HoistedValueMap &hoistedMap,
                          SymbolTable &moduleSymbols, IRMapping &cloneMapping,
                          const ConstExprAnalysis &constExprs) {
    // Do a depth first traversal of the producers, emitting them in a valid
    // def-use order.
    Operation *rootOp = constExprValue.getDefiningOp();
    assert(rootOp && "const-expr value should have a defining op");
    auto *rootInfo = constExprs.lookup(rootOp);
    assert(rootInfo && "must have const-value-info for const-expr root op");

    // Clone the whole tree as needed.
    cloneProducerTreeInto(builder, rootInfo, hoistedMap, cloneMapping,
                          constExprs);

    // And for each result, create a global and store into it.
    OpBuilder globalBuilder = getModuleBeginBuilder();
    for (Value origResult : rootOp->getResults()) {
      Value clonedResult = cloneMapping.lookup(origResult);
      GlobalOp globalOp = globalBuilder.create<GlobalOp>(loc, "hoisted", false,
                                                         origResult.getType());
      StringAttr globalSymbol = moduleSymbols.insert(globalOp);
      SymbolTable::setSymbolVisibility(globalOp,
                                       SymbolTable::Visibility::Private);

      // Save the mapping for the future.
      hoistedMap[origResult] = globalOp;

      // And store into it.
      LLVM_DEBUG(dbgs() << "    CREATE GLOBAL " << globalSymbol << " = "
                        << clonedResult << "\n");
      builder.create<GlobalStoreOp>(loc, clonedResult, globalSymbol);
    }

    builder.create<InitializerReturnOp>(loc);
  }

  void cleanupDeadOps(const ConstExprAnalysis &constExprs) {
    llvm::DenseSet<Operation *> allOps;
    constExprs.populateConstExprOperations(allOps);

    // Since we are mutating the const-expr ops, the ConstExprAnalysis will no
    // longer be valid after this point.
    SmallVector<Operation *> worklist;
    worklist.reserve(allOps.size());
    bool madeChanges = true;
    while (madeChanges) {
      madeChanges = false;

      // Prepare worklist.
      worklist.clear();
      worklist.append(allOps.begin(), allOps.end());

      for (Operation *checkOp : worklist) {
        if (checkOp->use_empty()) {
          // Bingo.
          LLVM_DEBUG(dbgs() << "ERASE DEAD OP: " << *checkOp << "\n");
          madeChanges = true;
          allOps.erase(checkOp);
          checkOp->erase();
        }
      }
    }
  }

  OpBuilder getModuleBeginBuilder() {
    Block *block = getOperation().getBody();
    return OpBuilder::atBlockBegin(block);
  }

  OpBuilder getModuleEndBuilder() {
    Block *block = getOperation().getBody();
    return OpBuilder::atBlockEnd(block);
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createHoistIntoGlobalsPass(int64_t maxSizeIncreaseThreshold) {
  return std::make_unique<HoistIntoGlobalsPass>(maxSizeIncreaseThreshold);
}

} // namespace Util
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
