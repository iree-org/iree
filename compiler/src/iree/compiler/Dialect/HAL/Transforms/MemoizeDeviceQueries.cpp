// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

    // Find all query ops we want to memoize and group them together.
    // This lets us easily replace all usages of a match with a single variable.
    SmallVector<Attribute, 4> deviceQueryKeys;
    DenseMap<Attribute, std::vector<IREE::HAL::DeviceQueryOp>> deviceQueryOps;
    for (auto callableOp : moduleOp.getOps<mlir::CallableOpInterface>()) {
      callableOp.walk([&](IREE::HAL::DeviceQueryOp queryOp) {
        auto fullKey = ArrayAttr::get(
            moduleOp.getContext(),
            {
                StringAttr::get(moduleOp.getContext(),
                                queryOp.getCategory() + queryOp.getKey()),
                queryOp.getDefaultValue().has_value()
                    ? queryOp.getDefaultValueAttr()
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
    SymbolTable symbolTable(moduleOp);
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
    for (auto queryKey : llvm::enumerate(deviceQueryKeys)) {
      auto queryOps = deviceQueryOps[queryKey.value()];
      auto anyQueryOp = queryOps.front();
      auto queryType = anyQueryOp.getValue().getType();

      // Merge all the locs as we are deduping the original query ops.
      auto fusedLoc =
          moduleBuilder.getFusedLoc(llvm::to_vector<4>(llvm::map_range(
              queryOps, [&](Operation *op) { return op->getLoc(); })));

      // The initializer will perform the query once and store it in the
      // variable.
      std::string variableName =
          "_device_query_" + std::to_string(queryKey.index());
      auto valueGlobalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
          fusedLoc, variableName,
          /*isMutable=*/false, queryType);
      symbolTable.insert(valueGlobalOp);
      valueGlobalOp.setPrivate();
      auto okGlobalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
          fusedLoc, variableName + "_ok",
          /*isMutable=*/false, moduleBuilder.getI1Type());
      symbolTable.insert(okGlobalOp);
      okGlobalOp.setPrivate();

      auto initializerOp =
          moduleBuilder.create<IREE::Util::InitializerOp>(fusedLoc);
      auto funcBuilder = OpBuilder::atBlockBegin(initializerOp.addEntryBlock());
      auto device =
          funcBuilder.createOrFold<IREE::HAL::ExSharedDeviceOp>(fusedLoc);
      auto queryOp = funcBuilder.create<IREE::HAL::DeviceQueryOp>(
          fusedLoc, funcBuilder.getI1Type(), queryType, device,
          anyQueryOp.getCategoryAttr(), anyQueryOp.getKeyAttr(),
          anyQueryOp.getDefaultValueAttr());
      funcBuilder.create<IREE::Util::GlobalStoreOp>(fusedLoc, queryOp.getOk(),
                                                    okGlobalOp.getName());
      funcBuilder.create<IREE::Util::GlobalStoreOp>(
          fusedLoc, queryOp.getValue(), valueGlobalOp.getName());
      funcBuilder.create<IREE::Util::InitializerReturnOp>(fusedLoc);

      for (auto queryOp : queryOps) {
        OpBuilder replaceBuilder(queryOp);
        auto okLoadOp = replaceBuilder.create<IREE::Util::GlobalLoadOp>(
            fusedLoc, okGlobalOp.getType(), okGlobalOp.getName());
        auto resultLoadOp = replaceBuilder.create<IREE::Util::GlobalLoadOp>(
            fusedLoc, valueGlobalOp.getType(), valueGlobalOp.getName());
        queryOp.replaceAllUsesWith(ValueRange{
            okLoadOp.getResult(),
            resultLoadOp.getResult(),
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
