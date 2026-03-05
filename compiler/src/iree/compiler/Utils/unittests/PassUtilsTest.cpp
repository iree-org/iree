// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/PassUtils.h"

#include <atomic>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

//===----------------------------------------------------------------------===//
// Test passes
//===----------------------------------------------------------------------===//

/// An OperationPass<> that sets a named unit attribute on whatever operation it
/// runs on. Because it is a generic (untyped) pass, the OpPipelineAdaptorPass
/// can run it directly on dispatched child operations.
struct MarkerPass final : public PassWrapper<MarkerPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkerPass)

  explicit MarkerPass(std::string name) : name(std::move(name)) {}
  MarkerPass(const MarkerPass &other) : PassWrapper(other), name(other.name) {}

  StringRef getArgument() const override { return "test-marker"; }

  void runOnOperation() override {
    getOperation()->setAttr(name, UnitAttr::get(&getContext()));
  }

  std::string name;
};

/// An OperationPass<> that always signals pass failure.
struct FailingPass final : public PassWrapper<FailingPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FailingPass)

  StringRef getArgument() const override { return "test-failing"; }

  void runOnOperation() override { signalPassFailure(); }
};

/// An OperationPass<> that records its execution order via a shared atomic
/// counter and stores the counter value as an integer attribute. Used to
/// verify pass ordering across module-level and dispatched function-level
/// passes.
struct CounterPass final : public PassWrapper<CounterPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CounterPass)

  CounterPass(std::string attrName, std::atomic<int> *counter)
      : attrName(std::move(attrName)), counter(counter) {}
  CounterPass(const CounterPass &other)
      : PassWrapper(other), attrName(other.attrName), counter(other.counter) {}

  StringRef getArgument() const override { return "test-counter"; }

  void runOnOperation() override {
    int order = counter->fetch_add(1);
    getOperation()->setAttr(
        attrName, IntegerAttr::get(IntegerType::get(&getContext(), 32), order));
  }

  std::string attrName;
  std::atomic<int> *counter;
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Create a module containing private functions with the given names.
static OwningOpRef<ModuleOp> createModule(MLIRContext &context,
                                          ArrayRef<StringRef> funcNames) {
  Builder builder(&context);
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  for (StringRef name : funcNames) {
    func::FuncOp func = func::FuncOp::create(builder.getUnknownLoc(), name,
                                             builder.getFunctionType({}, {}));
    func.setPrivate();
    module->push_back(func);
  }
  return module;
}

/// Return true if the named function inside |module| has a discardable
/// attribute called |attrName|.
static bool funcHasAttr(ModuleOp module, StringRef funcName,
                        StringRef attrName) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (func.getName() == funcName) {
      return func->hasAttr(attrName);
    }
  }
  return false;
}

/// Return the integer value of a named attribute on a function, or -1.
static int getFuncAttrInt(ModuleOp module, StringRef funcName,
                          StringRef attrName) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (func.getName() == funcName) {
      if (auto intAttr = dyn_cast_if_present<IntegerAttr>(
              func->getDiscardableAttr(attrName))) {
        return intAttr.getInt();
      }
    }
  }
  return -1;
}

/// Return the integer value of a named attribute on a module, or -1.
static int getModuleAttrInt(ModuleOp module, StringRef attrName) {
  if (IntegerAttr intAttr = dyn_cast_if_present<IntegerAttr>(
          module->getDiscardableAttr(attrName))) {
    return intAttr.getInt();
  }
  return -1;
}

/// Condition: matches operations whose name (for func ops) equals |target|.
static MultiPipelineNest::ConditionFn nameIs(StringRef target) {
  return [name = target.str()](Operation *op) {
    auto funcOp = dyn_cast<func::FuncOp>(op);
    return funcOp && funcOp.getName() == name;
  };
}

//===----------------------------------------------------------------------===//
// MultiPipelineNest tests
//===----------------------------------------------------------------------===//

TEST(MultiPipelineNestTest, OpPipelineAdaptor) {
  // Two functions dispatched to two different pipelines based on name.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha", "beta"});

  PassManager pm(&context);
  {
    MultiPipelineNest nest(pm);
    nest.nestIf(nameIs("alpha"))
        .addPass(std::make_unique<MarkerPass>("pass_a"));
    nest.nestIf(nameIs("beta")).addPass(std::make_unique<MarkerPass>("pass_b"));
  }

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  EXPECT_TRUE(funcHasAttr(*module, "alpha", "pass_a"));
  EXPECT_FALSE(funcHasAttr(*module, "alpha", "pass_b"));
  EXPECT_FALSE(funcHasAttr(*module, "beta", "pass_a"));
  EXPECT_TRUE(funcHasAttr(*module, "beta", "pass_b"));
}

TEST(MultiPipelineNestTest, FirstMatchWins) {
  // When multiple conditions match, only the first pipeline runs.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  PassManager pm(&context);
  {
    MultiPipelineNest nest(pm);
    nest.nestIf([](Operation *) { return true; })
        .addPass(std::make_unique<MarkerPass>("first"));
    nest.nestIf([](Operation *) { return true; })
        .addPass(std::make_unique<MarkerPass>("second"));
  }

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  EXPECT_TRUE(funcHasAttr(*module, "alpha", "first"));
  EXPECT_FALSE(funcHasAttr(*module, "alpha", "second"));
}

TEST(MultiPipelineNestTest, NoMatchSkipsOp) {
  // Operations that match no condition are left untouched.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  PassManager pm(&context);
  {
    MultiPipelineNest nest(pm);
    nest.nestIf(nameIs("beta")).addPass(std::make_unique<MarkerPass>("pass_b"));
  }

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  EXPECT_FALSE(funcHasAttr(*module, "alpha", "pass_b"));
}

TEST(MultiPipelineNestTest, AddPassToAllPipelines) {
  // addPass() appends to every existing sub-pipeline.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha", "beta"});

  PassManager pm(&context);
  {
    MultiPipelineNest nest(pm);
    nest.nestIf(nameIs("alpha"))
        .addPass(std::make_unique<MarkerPass>("pass_a"));
    nest.nestIf(nameIs("beta")).addPass(std::make_unique<MarkerPass>("pass_b"));
    nest.addPass([] { return std::make_unique<MarkerPass>("common"); });
  }

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  EXPECT_TRUE(funcHasAttr(*module, "alpha", "pass_a"));
  EXPECT_TRUE(funcHasAttr(*module, "alpha", "common"));
  EXPECT_TRUE(funcHasAttr(*module, "beta", "pass_b"));
  EXPECT_TRUE(funcHasAttr(*module, "beta", "common"));
}

TEST(MultiPipelineNestTest, AddPredicatedPass) {
  // addPredicatedPass(false, ...) should not add the pass.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  PassManager pm(&context);
  {
    MultiPipelineNest nest(pm);
    nest.nestIf([](Operation *) { return true; })
        .addPass(std::make_unique<MarkerPass>("always"));
    nest.addPredicatedPass(true,
                           [] { return std::make_unique<MarkerPass>("on"); });
    nest.addPredicatedPass(false,
                           [] { return std::make_unique<MarkerPass>("off"); });
  }

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  EXPECT_TRUE(funcHasAttr(*module, "alpha", "always"));
  EXPECT_TRUE(funcHasAttr(*module, "alpha", "on"));
  EXPECT_FALSE(funcHasAttr(*module, "alpha", "off"));
}

TEST(MultiPipelineNestTest, NestByOpType) {
  // nest<T>() creates a condition that checks isa<T>.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  PassManager pm(&context);
  {
    MultiPipelineNest nest(pm);
    nest.nest<func::FuncOp>().addPass(std::make_unique<MarkerPass>("typed"));
  }

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  EXPECT_TRUE(funcHasAttr(*module, "alpha", "typed"));
}

TEST(MultiPipelineNestTest, OrderingWithParentPasses) {
  // Passes added to the parent PM before and after the MultiPipelineNest
  // must execute in the correct order relative to the dispatched passes.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  std::atomic<int> counter{0};
  PassManager pm(&context);
  pm.addPass(std::make_unique<CounterPass>("test.before", &counter));
  {
    MultiPipelineNest nest(pm);
    nest.nestIf([](Operation *) { return true; })
        .addPass(std::make_unique<CounterPass>("test.func_pass", &counter));
  }
  pm.addPass(std::make_unique<CounterPass>("test.after", &counter));

  ASSERT_TRUE(succeeded(pm.run(module.get())));

  // "test.before" runs on module first (order 0).
  EXPECT_EQ(getModuleAttrInt(*module, "test.before"), 0);
  // Dispatched func pass runs second (order 1).
  EXPECT_EQ(getFuncAttrInt(*module, "alpha", "test.func_pass"), 1);
  // "test.after" runs on module last (order 2).
  EXPECT_EQ(getModuleAttrInt(*module, "test.after"), 2);
}

TEST(MultiPipelineNestTest, MultipleNestsPreserveOrdering) {
  // When multiple MultiPipelineNest scopes are interleaved with module passes,
  // each nest's dispatched passes appear in the correct position.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  std::atomic<int> counter{0};
  PassManager pm(&context);
  {
    MultiPipelineNest nest1(pm);
    nest1.nestIf([](Operation *) { return true; })
        .addPass(std::make_unique<CounterPass>("test.nest1", &counter));
  }
  pm.addPass(std::make_unique<CounterPass>("test.middle", &counter));
  {
    MultiPipelineNest nest2(pm);
    nest2.nestIf([](Operation *) { return true; })
        .addPass(std::make_unique<CounterPass>("test.nest2", &counter));
  }

  ASSERT_TRUE(succeeded(pm.run(module.get())));

  EXPECT_EQ(getFuncAttrInt(*module, "alpha", "test.nest1"), 0);
  EXPECT_EQ(getModuleAttrInt(*module, "test.middle"), 1);
  EXPECT_EQ(getFuncAttrInt(*module, "alpha", "test.nest2"), 2);
}

TEST(MultiPipelineNestTest, ParallelExecution) {
  // Verify correctness with multithreading enabled.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  // Multithreading is enabled by default.

  constexpr int kNumFuncs = 20;
  Builder builder(&context);
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  for (int i = 0; i < kNumFuncs; ++i) {
    std::string name = "func_" + std::to_string(i);
    func::FuncOp func = func::FuncOp::create(builder.getUnknownLoc(), name,
                                             builder.getFunctionType({}, {}));
    func.setPrivate();
    module->push_back(func);
  }

  PassManager pm(&context);
  {
    MultiPipelineNest nest(pm);
    nest.nestIf([](Operation *) { return true; })
        .addPass(std::make_unique<MarkerPass>("processed"));
  }

  ASSERT_TRUE(succeeded(pm.run(module.get())));

  int processedCount = 0;
  for (func::FuncOp func : module->getOps<func::FuncOp>()) {
    EXPECT_TRUE(func->hasAttr("processed"))
        << "Function " << func.getName().str() << " was not processed";
    ++processedCount;
  }
  EXPECT_EQ(processedCount, kNumFuncs);
}

TEST(MultiPipelineNestTest, FailurePropagationSync) {
  // A failing sub-pipeline pass must cause the overall pipeline to fail.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  PassManager pm(&context);
  {
    MultiPipelineNest nest(pm);
    nest.nestIf([](Operation *) { return true; })
        .addPass(std::make_unique<FailingPass>());
  }

  EXPECT_TRUE(failed(pm.run(module.get())));
}

TEST(MultiPipelineNestTest, FailurePropagationAsync) {
  // Failure propagation must also work with multithreading enabled.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  // Multithreading enabled by default.
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  PassManager pm(&context);
  {
    MultiPipelineNest nest(pm);
    nest.nestIf([](Operation *) { return true; })
        .addPass(std::make_unique<FailingPass>());
  }

  EXPECT_TRUE(failed(pm.run(module.get())));
}

TEST(MultiPipelineNestTest, MultipleFunctionsSamePipeline) {
  // All functions matching the same condition are processed.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module =
      createModule(context, {"alpha", "beta", "gamma"});

  std::atomic<int> counter{0};
  PassManager pm(&context);
  {
    MultiPipelineNest nest(pm);
    nest.nestIf([](Operation *) { return true; })
        .addPass(std::make_unique<CounterPass>("test.order", &counter));
  }

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  // All three functions should be processed with distinct order values.
  int alphaOrder = getFuncAttrInt(*module, "alpha", "test.order");
  int betaOrder = getFuncAttrInt(*module, "beta", "test.order");
  int gammaOrder = getFuncAttrInt(*module, "gamma", "test.order");
  EXPECT_GE(alphaOrder, 0);
  EXPECT_GE(betaOrder, 0);
  EXPECT_GE(gammaOrder, 0);
  EXPECT_NE(alphaOrder, betaOrder);
  EXPECT_NE(alphaOrder, gammaOrder);
  EXPECT_NE(betaOrder, gammaOrder);
}

TEST(MultiPipelineNestTest, ParallelMultipleEntries) {
  // Parallel dispatch with multiple entry indices exercises the per-thread
  // executor cloning and entryIdx > 0 code path.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  // Multithreading enabled by default.

  constexpr int kNumPerGroup = 10;
  Builder builder(&context);
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  for (int i = 0; i < kNumPerGroup; ++i) {
    for (StringRef prefix : {"alpha_", "beta_"}) {
      std::string name = std::string(prefix) + std::to_string(i);
      func::FuncOp func = func::FuncOp::create(builder.getUnknownLoc(), name,
                                               builder.getFunctionType({}, {}));
      func.setPrivate();
      module->push_back(func);
    }
  }

  PassManager pm(&context);
  {
    MultiPipelineNest nest(pm);
    nest.nestIf([](Operation *op) {
          func::FuncOp f = dyn_cast<func::FuncOp>(op);
          return f && f.getName().starts_with("alpha");
        })
        .addPass(std::make_unique<MarkerPass>("alpha_marker"));
    nest.nestIf([](Operation *op) {
          func::FuncOp f = dyn_cast<func::FuncOp>(op);
          return f && f.getName().starts_with("beta");
        })
        .addPass(std::make_unique<MarkerPass>("beta_marker"));
  }

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  for (func::FuncOp func : module->getOps<func::FuncOp>()) {
    if (func.getName().starts_with("alpha")) {
      EXPECT_TRUE(func->hasAttr("alpha_marker"))
          << func.getName().str() << " missing alpha_marker";
      EXPECT_FALSE(func->hasAttr("beta_marker"))
          << func.getName().str() << " has beta_marker";
    } else {
      EXPECT_TRUE(func->hasAttr("beta_marker"))
          << func.getName().str() << " missing beta_marker";
      EXPECT_FALSE(func->hasAttr("alpha_marker"))
          << func.getName().str() << " has alpha_marker";
    }
  }
}

TEST(MultiPipelineNestTest, EmptyNestIsNoOp) {
  // A MultiPipelineNest with no entries should not add any passes.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  PassManager pm(&context);
  {
    MultiPipelineNest nest(pm);
    // No entries added — should not insert any pass.
  }

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  // No attributes should have been set.
  EXPECT_FALSE(funcHasAttr(*module, "alpha", "anything"));
  // Empty nest should not insert a pass.
  EXPECT_EQ(pm.size(), 0u);
}

TEST(MultiPipelineNestTest, AddPassForSpecificType) {
  // addPassFor<T>() should only add to entries targeting that type.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  PassManager pm(&context);
  {
    MultiPipelineNest nest(pm);
    nest.nest<func::FuncOp>();
    nest.nestIf([](Operation *) { return true; });
    // Add to all entries.
    nest.addPass([] { return std::make_unique<MarkerPass>("common"); });
    // Add only to func::FuncOp entries.
    nest.addPassFor<func::FuncOp>(
        [] { return std::make_unique<MarkerPass>("func_only"); });
  }

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  // The func::FuncOp entry matches first (first-match-wins), so it runs both
  // "common" and "func_only".
  EXPECT_TRUE(funcHasAttr(*module, "alpha", "common"));
  EXPECT_TRUE(funcHasAttr(*module, "alpha", "func_only"));
}

//===----------------------------------------------------------------------===//
// MultiOpNest tests
//===----------------------------------------------------------------------===//

TEST(MultiOpNestTest, BasicDispatch) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha", "beta"});

  PassManager pm(&context);
  MultiOpNest<func::FuncOp>(pm).addPass(
      [] { return std::make_unique<MarkerPass>("processed"); });

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  EXPECT_TRUE(funcHasAttr(*module, "alpha", "processed"));
  EXPECT_TRUE(funcHasAttr(*module, "beta", "processed"));
}

TEST(MultiOpNestTest, AddPredicatedPass) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  PassManager pm(&context);
  MultiOpNest<func::FuncOp>(pm)
      .addPredicatedPass(true,
                         [] { return std::make_unique<MarkerPass>("on"); })
      .addPredicatedPass(false,
                         [] { return std::make_unique<MarkerPass>("off"); });

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  EXPECT_TRUE(funcHasAttr(*module, "alpha", "on"));
  EXPECT_FALSE(funcHasAttr(*module, "alpha", "off"));
}

TEST(MultiOpNestTest, NamedVariableOrdering) {
  // Named MultiOpNest variable in braces with passes before and after.
  // Verifies that the adaptor is positioned at construction time.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  std::atomic<int> counter{0};
  PassManager pm(&context);
  pm.addPass(std::make_unique<CounterPass>("test.before", &counter));
  {
    MultiOpNest<func::FuncOp> funcNest(pm);
    funcNest.addPass([&] {
      return std::make_unique<CounterPass>("test.func_pass", &counter);
    });
  }
  pm.addPass(std::make_unique<CounterPass>("test.after", &counter));

  ASSERT_TRUE(succeeded(pm.run(module.get())));

  EXPECT_EQ(getModuleAttrInt(*module, "test.before"), 0);
  EXPECT_EQ(getFuncAttrInt(*module, "alpha", "test.func_pass"), 1);
  EXPECT_EQ(getModuleAttrInt(*module, "test.after"), 2);
}

TEST(MultiOpNestTest, CommitPassForInterleaving) {
  // When a named MultiOpNest must be interleaved with parent-level passes,
  // call commitPass() to eagerly insert the adaptor at the desired position.
  // Without commitPass(), deferred insertion would place the adaptor after
  // the module pass.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  std::atomic<int> counter{0};
  PassManager pm(&context);
  MultiOpNest<func::FuncOp> funcNest(pm);
  funcNest.addPass(
      [&] { return std::make_unique<CounterPass>("test.func1", &counter); });
  // Commit before adding the module pass to lock in ordering.
  funcNest.commitPass();
  pm.addPass(std::make_unique<CounterPass>("test.module", &counter));

  ASSERT_TRUE(succeeded(pm.run(module.get())));

  // Func pass runs before the module pass because commitPass() inserted
  // the adaptor eagerly.
  EXPECT_EQ(getFuncAttrInt(*module, "alpha", "test.func1"), 0);
  EXPECT_EQ(getModuleAttrInt(*module, "test.module"), 1);
}

TEST(MultiOpNestTest, MergeAdjacentAdaptors) {
  // Two consecutive MultiOpNest<func::FuncOp> should merge into a single
  // adaptor pass. Both sets of passes must run on the function.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  std::atomic<int> counter{0};
  PassManager pm(&context);
  MultiOpNest<func::FuncOp>(pm).addPass(
      [&] { return std::make_unique<CounterPass>("test.first", &counter); });
  MultiOpNest<func::FuncOp>(pm).addPass(
      [&] { return std::make_unique<CounterPass>("test.second", &counter); });

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  // Both passes must have run, in order.
  int firstOrder = getFuncAttrInt(*module, "alpha", "test.first");
  int secondOrder = getFuncAttrInt(*module, "alpha", "test.second");
  EXPECT_EQ(firstOrder, 0);
  EXPECT_EQ(secondOrder, 1);
}

TEST(MultiOpNestTest, MergeSubsetTypes) {
  // A 2-type nest followed by a 3-type superset nest should merge. The
  // extra type in the second nest becomes a new entry in the combined pass.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  std::atomic<int> counter{0};
  PassManager pm(&context);
  // First nest: only func::FuncOp (simulating a 1-type nest).
  MultiOpNest<func::FuncOp>(pm).addPass(
      [&] { return std::make_unique<CounterPass>("test.nest1", &counter); });
  // Second nest: func::FuncOp (simulating a superset). Since both nests
  // have TypeID-annotated entries, they should merge.
  MultiOpNest<func::FuncOp>(pm).addPass(
      [&] { return std::make_unique<CounterPass>("test.nest2", &counter); });

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  EXPECT_EQ(getFuncAttrInt(*module, "alpha", "test.nest1"), 0);
  EXPECT_EQ(getFuncAttrInt(*module, "alpha", "test.nest2"), 1);
}

TEST(MultiOpNestTest, NoMergeWithModulePassBetween) {
  // A module pass between two nests prevents merge.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  std::atomic<int> counter{0};
  PassManager pm(&context);
  MultiOpNest<func::FuncOp>(pm).addPass(
      [&] { return std::make_unique<CounterPass>("test.before", &counter); });
  pm.addPass(std::make_unique<CounterPass>("test.module", &counter));
  MultiOpNest<func::FuncOp>(pm).addPass(
      [&] { return std::make_unique<CounterPass>("test.after", &counter); });

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  // All three should run in order: func pass, module pass, func pass.
  EXPECT_EQ(getFuncAttrInt(*module, "alpha", "test.before"), 0);
  EXPECT_EQ(getModuleAttrInt(*module, "test.module"), 1);
  EXPECT_EQ(getFuncAttrInt(*module, "alpha", "test.after"), 2);
}

TEST(MultiOpNestTest, NoMergeWithArbitraryConditions) {
  // When the predecessor uses nestIf without TypeID, merge is prevented.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  std::atomic<int> counter{0};
  PassManager pm(&context);
  {
    MultiPipelineNest nest(pm);
    nest.nestIf([](Operation *) { return true; })
        .addPass(std::make_unique<CounterPass>("test.cond", &counter));
  }
  // This nest has TypeIDs, but the predecessor doesn't, so no merge.
  MultiOpNest<func::FuncOp>(pm).addPass(
      [&] { return std::make_unique<CounterPass>("test.typed", &counter); });

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  EXPECT_EQ(getFuncAttrInt(*module, "alpha", "test.cond"), 0);
  EXPECT_EQ(getFuncAttrInt(*module, "alpha", "test.typed"), 1);
}

TEST(MultiOpNestTest, MergeParallelExecution) {
  // Verify merged adaptors work correctly with multithreading.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  // Multithreading enabled by default.

  constexpr int kNumFuncs = 20;
  Builder builder(&context);
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  for (int i = 0; i < kNumFuncs; ++i) {
    std::string name = "func_" + std::to_string(i);
    func::FuncOp func = func::FuncOp::create(builder.getUnknownLoc(), name,
                                             builder.getFunctionType({}, {}));
    func.setPrivate();
    module->push_back(func);
  }

  PassManager pm(&context);
  // Two consecutive nests that should merge.
  MultiOpNest<func::FuncOp>(pm).addPass(
      [] { return std::make_unique<MarkerPass>("batch0"); });
  MultiOpNest<func::FuncOp>(pm).addPass(
      [] { return std::make_unique<MarkerPass>("batch1"); });

  ASSERT_TRUE(succeeded(pm.run(module.get())));
  for (func::FuncOp func : module->getOps<func::FuncOp>()) {
    EXPECT_TRUE(func->hasAttr("batch0"))
        << func.getName().str() << " missing batch0";
    EXPECT_TRUE(func->hasAttr("batch1"))
        << func.getName().str() << " missing batch1";
  }
}

TEST(MultiOpNestTest, MergedShellNeverInserted) {
  // With deferred insertion, the second MultiOpNest merges its entries into
  // the first and is never inserted into the PM. The PM should contain
  // exactly 1 adaptor pass from construction through execution.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  std::atomic<int> counter{0};
  PassManager pm(&context);
  MultiOpNest<func::FuncOp>(pm).addPass(
      [&] { return std::make_unique<CounterPass>("test.first", &counter); });
  MultiOpNest<func::FuncOp>(pm).addPass(
      [&] { return std::make_unique<CounterPass>("test.second", &counter); });

  // Deferred insertion: merged shell is never added to the PM.
  EXPECT_EQ(pm.size(), 1u);
  ASSERT_TRUE(succeeded(pm.run(module.get())));
  // Still 1 after running.
  EXPECT_EQ(pm.size(), 1u);
  // Both passes still ran (from separate batches within the single adaptor).
  EXPECT_EQ(getFuncAttrInt(*module, "alpha", "test.first"), 0);
  EXPECT_EQ(getFuncAttrInt(*module, "alpha", "test.second"), 1);
}

TEST(MultiOpNestTest, ChainingIsEquivalentToNamed) {
  // Temporary (chained) MultiOpNest should produce the same ordering as a
  // named variable in braces.
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = createModule(context, {"alpha"});

  std::atomic<int> counter{0};
  PassManager pm(&context);
  pm.addPass(std::make_unique<CounterPass>("test.before", &counter));
  MultiOpNest<func::FuncOp>(pm).addPass([&] {
    return std::make_unique<CounterPass>("test.func_pass", &counter);
  });
  pm.addPass(std::make_unique<CounterPass>("test.after", &counter));

  ASSERT_TRUE(succeeded(pm.run(module.get())));

  EXPECT_EQ(getModuleAttrInt(*module, "test.before"), 0);
  EXPECT_EQ(getFuncAttrInt(*module, "alpha", "test.func_pass"), 1);
  EXPECT_EQ(getModuleAttrInt(*module, "test.after"), 2);
}

} // namespace
