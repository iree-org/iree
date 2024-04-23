// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Constant/ConstExpr.h"
#include "iree/compiler/Dialect/Util/Analysis/Constant/OpOracle.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#define DEBUG_TYPE "iree-constexpr"

namespace mlir::iree_compiler::IREE::Util {
namespace {

static llvm::cl::opt<std::string> clPrintDotGraphToFile(
    "iree-util-hoist-into-globals-print-constexpr-dotgraph-to",
    llvm::cl::desc(
        "Prints a dot graph representing the const-expr analysis. The red "
        "nodes represent roots and the green nodes represent hoisted values."),
    llvm::cl::value_desc("filename"));

// Maps an original value in the program to the symbol name of a global.
using HoistedValueMap = llvm::DenseMap<Value, GlobalOp>;

// Walks |fromOp| and up to gather all dialect attributes that want to be
// hoisted along with it. If the same named attribute is present on multiple
// ancestors only the most narrowly scoped value will be used.
static void gatherHoistableAttrs(Operation *fromOp,
                                 NamedAttrList &dialectAttrs) {
  for (auto attr : fromOp->getDialectAttrs()) {
    if (auto hoistableAttr =
            dyn_cast<IREE::Util::HoistableAttrInterface>(attr.getValue())) {
      if (hoistableAttr.shouldAttachToHoistedOps() &&
          !dialectAttrs.get(attr.getName())) {
        dialectAttrs.push_back(attr);
      }
    }
  }
  if (auto *parentOp = fromOp->getParentOp())
    gatherHoistableAttrs(parentOp, dialectAttrs);
}

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
    if (this->registerDependentDialectsFn) {
      (*registerDependentDialectsFn)(registry);
    }
  }

  HoistIntoGlobalsPass(const ExprHoistingOptions &options)
      : registerDependentDialectsFn(options.registerDependentDialectsFn) {
    this->maxSizeIncreaseThreshold.setValue(options.maxSizeIncreaseThreshold);
  }

  void runOnOperation() override {
    SymbolTable moduleSymbols(getOperation());
    const auto &constExprs = getAnalysis<ConstExprAnalysis>();
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
    for (auto funcOp : getOperation().getOps<FunctionOpInterface>()) {
      // Ignore initializers.
      if (isa<IREE::Util::InitializerOpInterface>(funcOp.getOperation()))
        continue;
      auto walkRes = funcOp.walk<WalkOrder::PreOrder>([&](Operation *iterOp) {
        // We only want to look at const-expr ops (non roots) since they may
        // have interesting escapes. Early exit here for efficiency.
        auto *iterInfo = constExprs.lookup(iterOp);
        if (!iterInfo)
          return WalkResult::advance();
        for (Value constExprResult : iterOp->getResults()) {
          auto *resultInfo = constExprs.lookup(constExprResult);
          assert(resultInfo && "must have const-expr info");
          if (policy.getDecision(resultInfo)->getOutcome() !=
              ConstExprHoistingPolicy::ENABLE_HOIST) {
            continue;
          }
          if (failed(hoistConstExpr(constExprResult, hoistedMap, moduleSymbols,
                                    constExprs))) {
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (walkRes.wasInterrupted())
        return signalPassFailure();
    }

    // Apply any remaining RAUW cleanups. We have to do these at the cleanup
    // phase since modifying the source program can invalidate the analysis.
    // Up to this point, we have only been cloning.
    OpBuilder builder(&getContext());
    for (auto [originalValue, globalOp] : hoistedMap) {
      builder.setInsertionPointAfterValue(originalValue);
      Value load = globalOp.createLoadOp(globalOp->getLoc(), builder)
                       .getLoadedGlobalValue();
      // Call user hook to cast back to the original type.
      if (auto hoistableType = dyn_cast<IREE::Util::HoistableTypeInterface>(
              originalValue.getType())) {
        load = hoistableType.decodeStorageType(builder, load.getLoc(),
                                               originalValue.getType(), load);
      }
      if (load.getType() != originalValue.getType()) {
        getOperation().emitError()
            << "Unresolved conflict between casted global of type "
            << load.getType() << " and original type "
            << originalValue.getType();
        return signalPassFailure();
      }
      originalValue.replaceAllUsesWith(load);
    }
    cleanupDeadOps(constExprs);
  }

  Operation *getTopLevelOp(Operation *childOp) {
    auto *moduleBlock = getOperation().getBody();
    auto *op = childOp;
    while (op->getBlock() != moduleBlock)
      op = op->getParentOp();
    return op;
  }

  LogicalResult hoistConstExpr(Value originalValue, HoistedValueMap &hoistedMap,
                               SymbolTable &moduleSymbols,
                               const ConstExprAnalysis &constExprs) {
    IREE::Util::GlobalOp existingGlobal = hoistedMap.lookup(originalValue);
    if (existingGlobal)
      return success();

    // Gather any dialect attributes we may need to preserve.
    auto *topLevelOp = getTopLevelOp(originalValue.getDefiningOp());
    NamedAttrList dialectAttrs;
    gatherHoistableAttrs(topLevelOp, dialectAttrs);

    // No existing mapping - create a new global.
    OpBuilder moduleBuilder(topLevelOp);
    auto initializerOp =
        moduleBuilder.create<IREE::Util::InitializerOp>(originalValue.getLoc());
    initializerOp->setDialectAttrs(dialectAttrs);
    auto initializerBuilder =
        OpBuilder::atBlockEnd(initializerOp.addEntryBlock());
    moduleBuilder.setInsertionPoint(initializerOp);
    if (failed(cloneConstExprInto(initializerOp.getLoc(), moduleBuilder,
                                  initializerBuilder, originalValue,
                                  dialectAttrs, hoistedMap, moduleSymbols,
                                  constExprs))) {
      return failure();
    }

    existingGlobal = hoistedMap.lookup(originalValue);
    (void)existingGlobal;
    assert(existingGlobal &&
           "hoisting const-expr should have mapped a global for the requested "
           "value");
    return success();
  }

  void
  cloneProducerTreeInto(OpBuilder &initializerBuilder,
                        const ConstExprAnalysis::ConstValueInfo *producerInfo,
                        HoistedValueMap &hoistedMap, IRMapping &cloneMapping,
                        const ConstExprAnalysis &constExprs) {
    if (cloneMapping.contains(producerInfo->constValue))
      return;

    // We either have a global associated already or we need to traverse
    // down and materialize producers.
    IREE::Util::GlobalOp existingGlobal =
        hoistedMap.lookup(producerInfo->constValue);
    if (existingGlobal) {
      Value newGlobal =
          existingGlobal
              .createLoadOp(existingGlobal.getLoc(), initializerBuilder)
              .getLoadedGlobalValue();
      // Call user hook to cast back to the original type.
      if (auto hoistableType = dyn_cast<IREE::Util::HoistableTypeInterface>(
              producerInfo->constValue.getType())) {
        newGlobal = hoistableType.decodeStorageType(
            initializerBuilder, newGlobal.getLoc(),
            producerInfo->constValue.getType(), newGlobal);
      }
      cloneMapping.map(producerInfo->constValue, newGlobal);
      return;
    }

    // Materialize all producers recursively.
    for (auto *producerInfo : producerInfo->producers) {
      cloneProducerTreeInto(initializerBuilder, producerInfo, hoistedMap,
                            cloneMapping, constExprs);
    }

    // And clone the requested op.
    Operation *sourceOp = producerInfo->constValue.getDefiningOp();
    assert(sourceOp && "must have defining op for const-expr values");
    LLVM_DEBUG({
      llvm::dbgs() << "[HoistIntoGlobals]    + clone op: ";
      sourceOp->print(llvm::dbgs(), constExprs.getAsmState());
      llvm::dbgs() << "\n";
    });
    Operation *clonedOp = sourceOp->clone(cloneMapping);
    initializerBuilder.insert(clonedOp);
  }

  // Clones the const expr tree rooted at `constExprValue` into the given
  // initializer, noting any new hoisted value mappings that result. At
  // a minimum, a mapping will be created for the requested value.
  LogicalResult cloneConstExprInto(Location loc, OpBuilder &moduleBuilder,
                                   OpBuilder &initializerBuilder,
                                   Value constExprValue,
                                   NamedAttrList dialectAttrs,
                                   HoistedValueMap &hoistedMap,
                                   SymbolTable &moduleSymbols,
                                   const ConstExprAnalysis &constExprs) {
    // Do a depth first traversal of the producers, emitting them in a valid
    // def-use order.
    Operation *rootOp = constExprValue.getDefiningOp();
    assert(rootOp && "const-expr value should have a defining op");
    auto *rootInfo = constExprs.lookup(rootOp);
    assert(rootInfo && "must have const-value-info for const-expr root op");

    // Clone the whole tree as needed.
    IRMapping cloneMapping;
    cloneProducerTreeInto(initializerBuilder, rootInfo, hoistedMap,
                          cloneMapping, constExprs);

    // And for each result, create a global and store into it.
    for (Value origResult : rootOp->getResults()) {
      Value clonedResult = cloneMapping.lookup(origResult);
      Type globalType = origResult.getType();
      // If the original type is registered as hoistable, invoke the interface
      // functions for setting the preferred storage type.
      auto hoistableType =
          dyn_cast<IREE::Util::HoistableTypeInterface>(globalType);
      // Get the preferred global storage type.
      if (hoistableType) {
        globalType = hoistableType.getPreferredStorageType();
      }
      auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
          loc, "hoisted", false, globalType);
      moduleSymbols.insert(globalOp);
      SymbolTable::setSymbolVisibility(globalOp,
                                       SymbolTable::Visibility::Private);
      globalOp->setDialectAttrs(dialectAttrs);

      // Save the mapping for the future.
      hoistedMap[origResult] = globalOp;

      // And store into it.
      LLVM_DEBUG({
        llvm::dbgs() << "[HoistIntoGlobals]    + create global @"
                     << globalOp.getSymName() << " = ";
        clonedResult.print(llvm::dbgs());
        llvm::dbgs() << "\n";
      });
      // Cast to the preferred global storage type.
      if (hoistableType) {
        clonedResult = hoistableType.encodeStorageType(
            initializerBuilder, clonedResult.getLoc(), globalType,
            clonedResult);
      }
      if (clonedResult.getType() != globalType) {
        globalOp.emitError()
            << "Unresolved conflict between global of type " << globalType
            << " and stored type " << clonedResult.getType();
        return failure();
      }
      globalOp.createStoreOp(loc, clonedResult, initializerBuilder);
    }

    initializerBuilder.create<IREE::Util::ReturnOp>(loc);
    return success();
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
          LLVM_DEBUG({
            llvm::dbgs() << "[HoistIntoGlobals] erase dead op: ";
            checkOp->print(llvm::dbgs(), constExprs.getAsmState());
            llvm::dbgs() << "\n";
          });
          madeChanges = true;
          allOps.erase(checkOp);
          checkOp->erase();
        }
      }
    }
  }

private:
  const std::optional<ExprHoistingOptions::RegisterDialectsFn>
      registerDependentDialectsFn;
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createHoistIntoGlobalsPass(const ExprHoistingOptions &options) {
  return std::make_unique<HoistIntoGlobalsPass>(options);
}

std::unique_ptr<OperationPass<mlir::ModuleOp>> createHoistIntoGlobalsPass() {
  IREE::Util::ExprHoistingOptions options;
  return std::make_unique<HoistIntoGlobalsPass>(options);
}

} // namespace mlir::iree_compiler::IREE::Util
