// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {

// TODO(benvanik): make this an interface pattern instead to allow for
// backend-specific query ops.
template <typename T>
class DeviceMatchPatternExpansion : public OpRewritePattern<T> {
 public:
  DeviceMatchPatternExpansion(MLIRContext *context, StringRef queryNamespace)
      : OpRewritePattern<T>(context), queryNamespace(queryNamespace) {}

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    auto queryOp = rewriter.create<IREE::HAL::DeviceQueryOp>(
        op.getLoc(), rewriter.getI1Type(), rewriter.getI1Type(), op.device(),
        rewriter.getStringAttr(queryNamespace), op.patternAttr(),
        rewriter.getZeroAttr(rewriter.getI1Type()));
    rewriter.replaceOp(op, {queryOp.value()});
    return success();
  }

 private:
  StringRef queryNamespace;
};

}  // namespace

// Expands various hal.device.match.* ops to their lowered query form.
// This allows the memoization logic to deal with the simpler case of
// hal.device.query.
//
// We have the hal.device.match.* ops so that we can perform global
// optimizations and simplifications based on the semantics of the query, while
// once this runs and we have just the raw query ops we are only able to assume
// equality (vs. target-aware ranges/etc).
static LogicalResult expandMatchOps(ModuleOp moduleOp) {
  auto *context = moduleOp.getContext();
  OwningRewritePatternList patterns(context);
  patterns.insert<
      DeviceMatchPatternExpansion<IREE::HAL::DeviceMatchArchitectureOp>>(
      context, "hal.device.architecture");
  patterns.insert<DeviceMatchPatternExpansion<IREE::HAL::DeviceMatchFeatureOp>>(
      context, "hal.device.feature");
  patterns.insert<DeviceMatchPatternExpansion<IREE::HAL::DeviceMatchIDOp>>(
      context, "hal.device.id");
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  return applyPatternsAndFoldGreedily(moduleOp, frozenPatterns);
}

// NOTE: this implementation is just for a single active device. As we start to
// support multiple devices we'll need to change this to be per-device.
class MemoizeDeviceQueriesPass
    : public PassWrapper<MemoizeDeviceQueriesPass, OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const override {
    return "iree-hal-memoize-device-queries";
  }

  StringRef getDescription() const override {
    return "Caches hal.device.query results for use across the entire module";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Expand hal.device.match.* ops to hal.device.query ops.
    if (failed(expandMatchOps(moduleOp))) {
      return signalPassFailure();
    }

    // Find all query ops we want to memoize and group them together.
    // This lets us easily replace all usages of a match with a single variable.
    SmallVector<Attribute, 4> deviceQueryKeys;
    DenseMap<Attribute, std::vector<IREE::HAL::DeviceQueryOp>> deviceQueryOps;
    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      funcOp.walk([&](IREE::HAL::DeviceQueryOp queryOp) {
        auto fullKey = ArrayAttr::get(
            moduleOp.getContext(),
            {
                StringAttr::get(moduleOp.getContext(),
                                queryOp.category() + queryOp.key()),
                queryOp.default_value().hasValue() ? queryOp.default_valueAttr()
                                                   : Attribute{},
            });
        auto lookup = deviceQueryOps.try_emplace(
            fullKey, std::vector<IREE::HAL::DeviceQueryOp>{});
        if (lookup.second) {
          deviceQueryKeys.push_back(std::move(fullKey));
        }
        lookup.first->second.push_back(queryOp);
        return WalkResult::advance();
      });
    }

    // Create each query variable and replace the uses with loads.
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
    for (auto queryKey : llvm::enumerate(deviceQueryKeys)) {
      auto queryOps = deviceQueryOps[queryKey.value()];
      auto anyQueryOp = queryOps.front();
      auto queryType = anyQueryOp.value().getType();

      // Merge all the locs as we are deduping the original query ops.
      auto fusedLoc =
          moduleBuilder.getFusedLoc(llvm::to_vector<4>(llvm::map_range(
              queryOps, [&](Operation *op) { return op->getLoc(); })));

      // The initializer will perform the query once and store it in the
      // variable.
      std::string variableName =
          "_device_query_" + std::to_string(queryKey.index());
      auto initializerOp = moduleBuilder.create<FuncOp>(
          fusedLoc, variableName + "_initializer",
          moduleBuilder.getFunctionType({}, {queryType}));
      initializerOp.setPrivate();
      moduleBuilder.setInsertionPoint(initializerOp);
      auto variableOp = moduleBuilder.create<IREE::HAL::VariableOp>(
          fusedLoc, variableName,
          /*isMutable=*/false, initializerOp);
      variableOp.setPrivate();
      moduleBuilder.setInsertionPointAfter(initializerOp);

      auto funcBuilder = OpBuilder::atBlockBegin(initializerOp.addEntryBlock());
      auto device =
          funcBuilder.createOrFold<IREE::HAL::ExSharedDeviceOp>(fusedLoc);
      auto queryOp = funcBuilder.create<IREE::HAL::DeviceQueryOp>(
          fusedLoc, funcBuilder.getI1Type(), queryType, device,
          anyQueryOp.categoryAttr(), anyQueryOp.keyAttr(),
          anyQueryOp.default_valueAttr());
      funcBuilder.create<mlir::ReturnOp>(fusedLoc, queryOp.value());

      for (auto queryOp : queryOps) {
        OpBuilder replaceBuilder(queryOp);
        auto loadOp = replaceBuilder.create<IREE::HAL::VariableLoadOp>(
            fusedLoc, queryType, variableOp.getName());
        queryOp.replaceAllUsesWith(ValueRange{
            replaceBuilder.createOrFold<ConstantIntOp>(
                loadOp.getLoc(), /*value=*/1, /*width=*/1),
            loadOp.result(),
        });
        queryOp.erase();
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createMemoizeDeviceQueriesPass() {
  return std::make_unique<MemoizeDeviceQueriesPass>();
}

static PassRegistration<MemoizeDeviceQueriesPass> pass;

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
