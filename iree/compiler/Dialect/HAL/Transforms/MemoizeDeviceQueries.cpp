// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// NOTE: this implementation is just for a single active device. As we start to
// support multiple devices we'll need to change this to be per-device.
class MemoizeDeviceQueriesPass
    : public PassWrapper<MemoizeDeviceQueriesPass, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    // Find all match ops we want to memoize and group them together.
    // This lets us easily replace all usages of a match with a single variable.
    DenseMap<Attribute, std::vector<IREE::HAL::DeviceMatchIDOp>>
        deviceIDMatchOps;
    SmallVector<Attribute, 4> deviceIDMatchKeys;
    auto moduleOp = getOperation();
    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      funcOp.walk([&](IREE::HAL::DeviceMatchIDOp matchOp) {
        auto key = matchOp.patternAttr().cast<Attribute>();
        auto lookup = deviceIDMatchOps.try_emplace(
            key, std::vector<IREE::HAL::DeviceMatchIDOp>{});
        if (lookup.second) {
          deviceIDMatchKeys.push_back(key);
        }
        lookup.first->second.push_back(matchOp);
        return WalkResult::advance();
      });
    }

    // Create each match variable and replace the uses with loads.
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
    for (auto matchKey : llvm::enumerate(deviceIDMatchKeys)) {
      auto matchOps = deviceIDMatchOps[matchKey.value()];
      auto pattern = matchOps.front().pattern();

      // Merge all the locs as we are deduping the original query ops.
      auto fusedLoc = FusedLoc::get(
          llvm::to_vector<4>(llvm::map_range(
              matchOps, [&](Operation *op) { return op->getLoc(); })),
          moduleOp.getContext());

      // The initializer will perform the query once and store it in the
      // variable.
      std::string variableName =
          "_device_match_id_" + std::to_string(matchKey.index());
      auto initializerOp = moduleBuilder.create<FuncOp>(
          fusedLoc, variableName + "_initializer",
          moduleBuilder.getFunctionType({}, {moduleBuilder.getI1Type()}));
      SymbolTable::setSymbolVisibility(initializerOp,
                                       SymbolTable::Visibility::Private);
      auto variableOp = moduleBuilder.create<IREE::HAL::VariableOp>(
          fusedLoc, variableName,
          /*isMutable=*/false, initializerOp);
      SymbolTable::setSymbolVisibility(variableOp,
                                       SymbolTable::Visibility::Private);

      auto funcBuilder = OpBuilder::atBlockBegin(initializerOp.addEntryBlock());
      auto device =
          funcBuilder.createOrFold<IREE::HAL::ExSharedDeviceOp>(fusedLoc);
      auto matchOp = funcBuilder.create<IREE::HAL::DeviceMatchIDOp>(
          fusedLoc, funcBuilder.getI1Type(), device, pattern);
      funcBuilder.create<mlir::ReturnOp>(fusedLoc, matchOp.getResult());

      for (auto matchOp : matchOps) {
        OpBuilder replaceBuilder(matchOp);
        auto loadOp = replaceBuilder.create<IREE::HAL::VariableLoadOp>(
            fusedLoc, matchOp.getResult().getType(), variableOp.getName());
        matchOp.replaceAllUsesWith(loadOp.result());
        matchOp.erase();
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createMemoizeDeviceQueriesPass() {
  return std::make_unique<MemoizeDeviceQueriesPass>();
}

static PassRegistration<MemoizeDeviceQueriesPass> pass(
    "iree-hal-memoize-device-queries",
    "Caches hal.device.query results for use across the entire module");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
