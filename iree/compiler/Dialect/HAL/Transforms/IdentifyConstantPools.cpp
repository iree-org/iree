// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class IdentifyConstantPoolsPass
    : public PassWrapper<IdentifyConstantPoolsPass, OperationPass<ModuleOp>> {
 public:
  IdentifyConstantPoolsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<IREE::Flow::FlowDialect>();
    registry.insert<IREE::HAL::HALDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  StringRef getArgument() const override {
    return "iree-hal-identify-constant-pools";
  }

  StringRef getDescription() const override {
    return "Combines constant globals into one or more hal.constant_pools "
           "based on usage semantics.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Gather constant globals. We assume that prior passes/pipelines have
    // hoisted anything worth pooling to util.globals at the module scope.
    // We expect that immutable globals have already been de-duped and that
    // mutable globals that remain may have identical initializers.
    SmallVector<IREE::Util::GlobalOp, 16> mutableOps;
    SmallVector<IREE::Util::GlobalOp, 16> immutableOps;
    for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOp>()) {
      if (!globalOp.initial_value().hasValue()) continue;
      auto globalType = globalOp.type().dyn_cast<RankedTensorType>();
      if (!globalType) continue;
      if (globalOp.is_mutable()) {
        mutableOps.push_back(globalOp);
      } else {
        immutableOps.push_back(globalOp);
      }
    }
    if (mutableOps.empty() && immutableOps.empty()) {
      return;
    }

    // Derive buffer constraints based on target backends.
    auto bufferConstraints =
        IREE::HAL::DeviceTargetAttr::lookupConservativeBufferConstraints(
            moduleOp);

    SymbolTable moduleSymbolTable(moduleOp);
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
    auto globalUsages = gatherGlobalUsages(moduleOp);

    // Process the mutable ops where each constant is only used as an
    // initializer. The lifetime of these is short as we only use them to
    // populate the initial global buffer contents.
    makeConstantPool("_const_pool_init", mutableOps, bufferConstraints,
                     globalUsages, moduleOp, moduleSymbolTable, moduleBuilder);

    // Process the immutable ops where the same buffer will be used for the
    // lifetime of the module.
    makeConstantPool("_const_pool", immutableOps, bufferConstraints,
                     globalUsages, moduleOp, moduleSymbolTable, moduleBuilder);

    // NOTE: pools now contain the values but they are in an undefined order.
    // We should have following passes that reorder the values to cluster them
    // by usage time locality so that there's a better chance of them landing
    // in the same runtime buffers and prefetched mapped storage pages.
  }

 private:
  enum GlobalUsage {
    kAddress = 1 << 0,
    kLoad = 1 << 1,
  };

  // Gathers information about the usage of all globals in the module.
  DenseMap<StringRef, GlobalUsage> gatherGlobalUsages(mlir::ModuleOp moduleOp) {
    DenseMap<StringRef, GlobalUsage> uses;
    for (auto funcOp : moduleOp.getOps<mlir::FuncOp>()) {
      funcOp.walk([&](Operation *op) {
        if (auto addressOp = dyn_cast<IREE::Util::GlobalAddressOp>(op)) {
          auto it = uses.find(addressOp.global());
          if (it == uses.end()) {
            uses[addressOp.global()] = GlobalUsage::kAddress;
          } else {
            uses[addressOp.global()] =
                static_cast<GlobalUsage>(it->second | GlobalUsage::kAddress);
          }
        } else if (auto loadOp = dyn_cast<IREE::Util::GlobalLoadOp>(op)) {
          auto it = uses.find(loadOp.global());
          if (it == uses.end()) {
            uses[loadOp.global()] = GlobalUsage::kLoad;
          } else {
            uses[loadOp.global()] =
                static_cast<GlobalUsage>(it->second | GlobalUsage::kLoad);
          }
        }
      });
    }
    return uses;
  }

  // Makes a new hal.constant_pool containing the values of the given
  // global ops. The globals will be erased and all global loads will be
  // replaced with constant loads. Returns the constant pool, if it was created.
  Optional<ConstantPoolOp> makeConstantPool(
      StringRef poolName, ArrayRef<IREE::Util::GlobalOp> globalOps,
      BufferConstraintsAttr bufferConstraints,
      DenseMap<StringRef, GlobalUsage> &globalUsages, mlir::ModuleOp moduleOp,
      SymbolTable &moduleSymbolTable, OpBuilder &moduleBuilder) {
    // Create the pool to be filled with constant values.
    auto poolOp = OpBuilder(moduleBuilder.getContext())
                      .create<ConstantPoolOp>(moduleBuilder.getUnknownLoc(),
                                              poolName, bufferConstraints);
    moduleSymbolTable.insert(poolOp, moduleBuilder.getInsertionPoint());
    poolOp.setPrivate();

    // Replace each global and keep track of the mapping from global->value.
    // This allows us to do one run through the module to replace usages as a
    // post-processing step.
    DenseMap<StringRef, IREE::HAL::ConstantPoolValueOp> constantReplacements;
    SmallVector<Operation *, 4> deadOps;
    auto poolBuilder = OpBuilder::atBlockBegin(poolOp.getBody());
    for (auto globalOp : globalOps) {
      // Grab the constant value from the global that we'll be pooling.
      auto value =
          globalOp.initial_value().getValue().dyn_cast_or_null<ElementsAttr>();
      assert(value && "value precondition not met: must be elements attr");

      // Create the constant in the pool.
      auto valueOp = poolBuilder.create<ConstantPoolValueOp>(
          globalOp.getLoc(), globalOp.getName(), value);
      valueOp.setNested();

      // If the global is an immutable constant and used in compatible
      // ways we can turn them into constant loads instead. These will avoid
      // the additional runtime overhead of global lifetime tracking and
      // allow further optimizations at use sites where we know the values
      // come from constant memory.
      auto globalUsage = globalUsages[globalOp.getName()];
      if (!globalOp.is_mutable() && (globalUsage & GlobalUsage::kAddress)) {
        globalOp.emitWarning() << "global is used indirectly; currently "
                                  "unsupported for constant pooling";
        continue;
      }

      if (!globalOp.is_mutable()) {
        // Replace all loads of the global with loads of the constant.
        // We do the actual replacement in a post-processing step so we don't
        // modify the IR during this loop.
        constantReplacements[globalOp.getName()] = valueOp;
        deadOps.push_back(globalOp);
      } else {
        // Build an initializer function to populate the global with the
        // constant value on startup.
        changeToGlobalInitializerFunc(globalOp, valueOp);
      }
    }

    // Remove the pool if it didn't end up with any constants.
    if (poolOp.getBody()->front().hasTrait<OpTrait::IsTerminator>()) {
      poolOp.erase();
      return None;
    }

    // Process pending usage replacements.
    replaceConstantGlobalLoads(moduleOp, constantReplacements);

    // Cleanup any inlined globals we no longer need after replacement.
    for (auto deadOp : deadOps) {
      deadOp->erase();
    }

    return poolOp;
  }

  // Constructs a function that can be used as an initializer for a global
  // and inserts it by the global op in the module.
  FuncOp changeToGlobalInitializerFunc(IREE::Util::GlobalOp globalOp,
                                       IREE::HAL::ConstantPoolValueOp valueOp) {
    // Create the function and make the global point to it for init.
    OpBuilder moduleBuilder(globalOp.getContext());
    moduleBuilder.setInsertionPointAfter(globalOp);
    auto initializerName = (globalOp.getName() + "_initializer").str();
    auto initializerFunc = moduleBuilder.create<FuncOp>(
        globalOp.getLoc(), initializerName,
        moduleBuilder.getFunctionType({}, {globalOp.type()}));
    initializerFunc.setPrivate();
    globalOp->removeAttr("initial_value");
    globalOp->setAttr("initializer",
                      moduleBuilder.getSymbolRefAttr(initializerFunc));

    // Emit a constant load that will later on be turned into a runtime buffer
    // reference.
    auto funcBuilder = OpBuilder::atBlockBegin(initializerFunc.addEntryBlock());
    auto constValue = funcBuilder.createOrFold<IREE::HAL::ConstantPoolLoadOp>(
        globalOp.getLoc(), globalOp.type(),
        funcBuilder.getSymbolRefAttr(
            valueOp->getParentOfType<ConstantPoolOp>().getName(),
            {funcBuilder.getSymbolRefAttr(valueOp)}));
    funcBuilder.create<mlir::ReturnOp>(globalOp.getLoc(), constValue);

    return initializerFunc;
  }

  // Replaces uses of each global with references to the constant pool value.
  void replaceConstantGlobalLoads(
      mlir::ModuleOp moduleOp,
      DenseMap<StringRef, IREE::HAL::ConstantPoolValueOp> &replacements) {
    SmallVector<
        std::pair<IREE::Util::GlobalLoadOp, IREE::HAL::ConstantPoolValueOp>, 8>
        loadValues;
    for (auto funcOp : moduleOp.getOps<mlir::FuncOp>()) {
      funcOp.walk([&](IREE::Util::GlobalLoadOp loadOp) {
        auto replacement = replacements.find(loadOp.global());
        if (replacement != replacements.end()) {
          loadValues.push_back(std::make_pair(loadOp, replacement->second));
        }
      });
    }
    for (auto &loadValue : loadValues) {
      replaceGlobalLoadWithConstantLoad(loadValue.first, loadValue.second);
    }
  }

  // Replaces a util.global.load with a hal.constant_pool.load of a pooled
  // value.
  void replaceGlobalLoadWithConstantLoad(IREE::Util::GlobalLoadOp globalLoadOp,
                                         ConstantPoolValueOp valueOp) {
    OpBuilder builder(globalLoadOp);
    auto poolLoadOp = builder.create<ConstantPoolLoadOp>(
        globalLoadOp.getLoc(), globalLoadOp.getType(),
        builder.getSymbolRefAttr(
            valueOp->getParentOfType<ConstantPoolOp>().getName(),
            {builder.getSymbolRefAttr(valueOp)}));
    globalLoadOp.replaceAllUsesWith(poolLoadOp.result());
    globalLoadOp.erase();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createIdentifyConstantPoolsPass() {
  return std::make_unique<IdentifyConstantPoolsPass>();
}

static PassRegistration<IdentifyConstantPoolsPass> pass([] {
  return std::make_unique<IdentifyConstantPoolsPass>();
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
