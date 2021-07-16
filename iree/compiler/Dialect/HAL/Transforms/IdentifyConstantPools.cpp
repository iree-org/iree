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
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
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
  explicit IdentifyConstantPoolsPass(TargetOptions targetOptions)
      : targetOptions_(targetOptions) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect>();
    registry.insert<IREE::Flow::FlowDialect>();
    registry.insert<IREE::HAL::HALDialect>();
  }

  StringRef getArgument() const override {
    return "iree-hal-identify-constant-pools";
  }

  StringRef getDescription() const override {
    return "Combines constant variables into one or more hal.constant_pools "
           "based on usage semantics.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Gather constant variables. We assume that prior passes/pipelines have
    // hoisted anything worth pooling to flow.variables at the module scope.
    // We expect that immutable variables have already been de-duped and that
    // mutable variables that remain may have identical initializers.
    SmallVector<IREE::Flow::VariableOp, 16> mutableOps;
    SmallVector<IREE::Flow::VariableOp, 16> immutableOps;
    for (auto variableOp : moduleOp.getOps<IREE::Flow::VariableOp>()) {
      if (!variableOp.initial_value().hasValue()) continue;
      auto variableType = variableOp.type().dyn_cast<RankedTensorType>();
      if (!variableType) continue;
      if (variableOp.is_mutable()) {
        mutableOps.push_back(variableOp);
      } else {
        immutableOps.push_back(variableOp);
      }
    }
    if (mutableOps.empty() && immutableOps.empty()) {
      return;
    }

    // Derive buffer constraints based on target backends.
    auto bufferConstraints = computeConservativeBufferConstraints(
        targetOptions_, moduleOp.getContext());
    if (!bufferConstraints) {
      moduleOp.emitWarning() << "no target backends provided buffer "
                                "constraints; falling back to host default";
      bufferConstraints =
          TargetBackend::makeDefaultBufferConstraints(moduleOp.getContext());
    }

    SymbolTable moduleSymbolTable(moduleOp);
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
    auto variableUsages = gatherVariableUsages(moduleOp);

    // Process the mutable ops where each constant is only used as an
    // initializer. The lifetime of these is short as we only use them to
    // populate the initial variable buffer contents.
    makeConstantPool("_const_pool_init", mutableOps, bufferConstraints,
                     variableUsages, moduleOp, moduleSymbolTable,
                     moduleBuilder);

    // Process the immutable ops where the same buffer will be used for the
    // lifetime of the module.
    makeConstantPool("_const_pool", immutableOps, bufferConstraints,
                     variableUsages, moduleOp, moduleSymbolTable,
                     moduleBuilder);

    // NOTE: pools now contain the values but they are in an undefined order.
    // We should have following passes that reorder the values to cluster them
    // by usage time locality so that there's a better chance of them landing
    // in the same runtime buffers and prefetched mapped storage pages.
  }

 private:
  enum VariableUsage {
    kAddress = 1 << 0,
    kLoad = 1 << 1,
  };

  // Gathers information about the usage of all variables in the module.
  DenseMap<StringRef, VariableUsage> gatherVariableUsages(
      mlir::ModuleOp moduleOp) {
    DenseMap<StringRef, VariableUsage> uses;
    for (auto funcOp : moduleOp.getOps<mlir::FuncOp>()) {
      funcOp.walk([&](Operation *op) {
        if (auto addressOp = dyn_cast<IREE::Flow::VariableAddressOp>(op)) {
          auto it = uses.find(addressOp.variable());
          if (it == uses.end()) {
            uses[addressOp.variable()] = VariableUsage::kAddress;
          } else {
            uses[addressOp.variable()] = static_cast<VariableUsage>(
                it->second | VariableUsage::kAddress);
          }
        } else if (auto loadOp = dyn_cast<IREE::Flow::VariableLoadOp>(op)) {
          auto it = uses.find(loadOp.variable());
          if (it == uses.end()) {
            uses[loadOp.variable()] = VariableUsage::kLoad;
          } else {
            uses[loadOp.variable()] =
                static_cast<VariableUsage>(it->second | VariableUsage::kLoad);
          }
        }
      });
    }
    return uses;
  }

  // Tries to find the min/max constraints on buffers across all target
  // backends. This should really be done per pool based on the usage of the
  // constants (if pool 0 is used by device A and pool 1 is used by device B
  // then they should not need to have matching constraints).
  BufferConstraintsAttr computeConservativeBufferConstraints(
      const TargetOptions &targetOptions, MLIRContext *context) {
    auto targetBackends = getTargetBackends(targetOptions.targets);
    BufferConstraintsAttr attr = {};
    for (auto &targetBackend : targetBackends) {
      if (attr) {
        attr = intersectBufferConstraints(
            attr, targetBackend->queryBufferConstraints(context));
      } else {
        attr = targetBackend->queryBufferConstraints(context);
      }
    }
    return attr;
  }

  // Makes a new hal.constant_pool containing the values of the given
  // variable ops. The variables will be erased and all variable loads will be
  // replaced with constant loads. Returns the constant pool, if it was created.
  Optional<ConstantPoolOp> makeConstantPool(
      StringRef poolName, ArrayRef<IREE::Flow::VariableOp> variableOps,
      BufferConstraintsAttr bufferConstraints,
      DenseMap<StringRef, VariableUsage> &variableUsages,
      mlir::ModuleOp moduleOp, SymbolTable &moduleSymbolTable,
      OpBuilder &moduleBuilder) {
    // Create the pool to be filled with constant values.
    auto poolOp = OpBuilder(moduleBuilder.getContext())
                      .create<ConstantPoolOp>(moduleBuilder.getUnknownLoc(),
                                              poolName, bufferConstraints);
    moduleSymbolTable.insert(poolOp, moduleBuilder.getInsertionPoint());
    poolOp.setPrivate();

    // Replace each variable and keep track of the mapping from variable->value.
    // This allows us to do one run through the module to replace usages as a
    // post-processing step.
    DenseMap<StringRef, IREE::HAL::ConstantPoolValueOp> constantReplacements;
    SmallVector<Operation *, 4> deadOps;
    auto poolBuilder = OpBuilder::atBlockBegin(poolOp.getBody());
    for (auto variableOp : variableOps) {
      // Grab the constant value from the variable that we'll be pooling.
      auto value = variableOp.initial_value()
                       .getValue()
                       .dyn_cast_or_null<ElementsAttr>();
      assert(value && "value precondition not met: must be elements attr");

      // Create the constant in the pool.
      auto valueOp = poolBuilder.create<ConstantPoolValueOp>(
          variableOp.getLoc(), variableOp.getName(), value);
      valueOp.setNested();

      // If the variable is an immutable constant and used in compatible
      // ways we can turn them into constant loads instead. These will avoid
      // the additional runtime overhead of variable lifetime tracking and
      // allow further optimizations at use sites where we know the values
      // come from constant memory.
      auto variableUsage = variableUsages[variableOp.getName()];
      if (!variableOp.is_mutable() &&
          (variableUsage & VariableUsage::kAddress)) {
        variableOp.emitWarning() << "variable is used indirectly; currently "
                                    "unsupported for constant pooling";
        continue;
      }

      if (!variableOp.is_mutable()) {
        // Replace all loads of the variable with loads of the constant.
        // We do the actual replacement in a post-processing step so we don't
        // modify the IR during this loop.
        constantReplacements[variableOp.getName()] = valueOp;
        deadOps.push_back(variableOp);
      } else {
        // Build an initializer function to populate the variable with the
        // constant value on startup.
        changeToVariableInitializerFunc(variableOp, valueOp);
      }
    }

    // Remove the pool if it didn't end up with any constants.
    if (poolOp.getBody()->front().hasTrait<OpTrait::IsTerminator>()) {
      poolOp.erase();
      return None;
    }

    // Process pending usage replacements.
    replaceConstantVariableLoads(moduleOp, constantReplacements);

    // Cleanup any inlined variables we no longer need after replacement.
    for (auto deadOp : deadOps) {
      deadOp->erase();
    }

    return poolOp;
  }

  // Constructs a function that can be used as an initializer for a variable
  // and inserts it by the variable op in the module.
  FuncOp changeToVariableInitializerFunc(
      IREE::Flow::VariableOp variableOp,
      IREE::HAL::ConstantPoolValueOp valueOp) {
    // Create the function and make the variable point to it for init.
    OpBuilder moduleBuilder(variableOp.getContext());
    moduleBuilder.setInsertionPointAfter(variableOp);
    auto initializerName = (variableOp.getName() + "_initializer").str();
    auto initializerFunc = moduleBuilder.create<FuncOp>(
        variableOp.getLoc(), initializerName,
        moduleBuilder.getFunctionType({}, {variableOp.type()}));
    initializerFunc.setPrivate();
    variableOp->removeAttr("initial_value");
    variableOp->setAttr("initializer",
                        moduleBuilder.getSymbolRefAttr(initializerFunc));

    // Emit a constant load that will later on be turned into a runtime buffer
    // reference.
    auto funcBuilder = OpBuilder::atBlockBegin(initializerFunc.addEntryBlock());
    auto constValue = funcBuilder.createOrFold<IREE::HAL::ConstantPoolLoadOp>(
        variableOp.getLoc(), variableOp.type(),
        funcBuilder.getSymbolRefAttr(
            valueOp->getParentOfType<ConstantPoolOp>().getName(),
            {funcBuilder.getSymbolRefAttr(valueOp)}));
    funcBuilder.create<mlir::ReturnOp>(variableOp.getLoc(), constValue);

    return initializerFunc;
  }

  // Replaces uses of each variable with references to the constant pool value.
  void replaceConstantVariableLoads(
      mlir::ModuleOp moduleOp,
      DenseMap<StringRef, IREE::HAL::ConstantPoolValueOp> &replacements) {
    SmallVector<
        std::pair<IREE::Flow::VariableLoadOp, IREE::HAL::ConstantPoolValueOp>,
        8>
        loadValues;
    for (auto funcOp : moduleOp.getOps<mlir::FuncOp>()) {
      funcOp.walk([&](IREE::Flow::VariableLoadOp loadOp) {
        auto replacement = replacements.find(loadOp.variable());
        if (replacement != replacements.end()) {
          loadValues.push_back(std::make_pair(loadOp, replacement->second));
        }
      });
    }
    for (auto &loadValue : loadValues) {
      replaceVariableLoadWithConstantLoad(loadValue.first, loadValue.second);
    }
  }

  // Replaces a flow.variable.load with a hal.constant_pool.load of a pooled
  // value.
  void replaceVariableLoadWithConstantLoad(
      IREE::Flow::VariableLoadOp variableLoadOp, ConstantPoolValueOp valueOp) {
    OpBuilder builder(variableLoadOp);
    auto loadOp = builder.create<ConstantPoolLoadOp>(
        variableLoadOp.getLoc(), variableLoadOp.getType(),
        builder.getSymbolRefAttr(
            valueOp->getParentOfType<ConstantPoolOp>().getName(),
            {builder.getSymbolRefAttr(valueOp)}));
    variableLoadOp.replaceAllUsesWith(loadOp.result());
    variableLoadOp.erase();
  }

  TargetOptions targetOptions_;
};

std::unique_ptr<OperationPass<ModuleOp>> createIdentifyConstantPoolsPass(
    TargetOptions targetOptions) {
  return std::make_unique<IdentifyConstantPoolsPass>(targetOptions);
}

static PassRegistration<IdentifyConstantPoolsPass> pass([] {
  auto options = getTargetOptionsFromFlags();
  return std::make_unique<IdentifyConstantPoolsPass>(options);
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
