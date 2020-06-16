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
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
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

class MaterializeResourceCachesPass
    : public PassWrapper<MaterializeResourceCachesPass,
                         OperationPass<ModuleOp>> {
 public:
  MaterializeResourceCachesPass()
      : targetOptions_(getTargetOptionsFromFlags()) {}
  explicit MaterializeResourceCachesPass(TargetOptions targetOptions)
      : targetOptions_(targetOptions) {}

  void runOnOperation() override {
    auto moduleOp = getOperation();
    moduleBuilder = OpBuilder(&moduleOp.getBody()->front());

    // Declare executable variables so that we can reference them during lookup
    // replacement.
    auto executableOps = llvm::to_vector<8>(moduleOp.getOps<ExecutableOp>());
    for (auto executableOp : executableOps) {
      defineExecutableOp(executableOp);
    }

    // Declare all layouts used by the executables. This will ensure that the
    // initialization order is correct as any executable layout needed (and its
    // dependencies) will be created prior to the executable cache below. The
    // other nice thing is that we get ordering similar to the executable
    // variables above.
    for (auto executableOp : executableOps) {
      auto interfaceOp = executableOp.getInterfaceOp();
      defineExecutableLayoutOp(interfaceOp.getLoc(),
                               interfaceOp.getExecutableSetLayoutsAttr(),
                               interfaceOp.push_constantsAttr());
    }

    // Generate cached resource singletons and replace lookup ops with direct
    // loads from variables.
    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      for (auto &block : funcOp) {
        for (auto &op : llvm::make_early_inc_range(block)) {
          if (auto lookupOp = dyn_cast<DescriptorSetLayoutLookupOp>(op)) {
            replaceDescriptorSetLayoutLookupOp(lookupOp);
          } else if (auto lookupOp = dyn_cast<ExecutableLayoutLookupOp>(op)) {
            replaceExecutableLayoutLookupOp(lookupOp);
          } else if (auto lookupOp = dyn_cast<ExecutableLookupOp>(op)) {
            replaceExecutableLookupOp(lookupOp);
          }
        }
      }
    }

    // Create the shared default executable cache we will use to prepare all
    // of the executables.
    //
    // In the future we can instead cluster executables based on usage into
    // multiple caches to speed up load time in models that may have different
    // usage characteristics; for now this prepares all executables at startup.
    if (!executableOps.empty()) {
      defineExecutableCacheOp(executableOps);
    }
  }

 private:
  VariableOp defineExecutableOp(ExecutableOp executableOp) {
    auto symbolName =
        (StringRef("_executable_") + executableOp.sym_name()).str();

    auto executableType = ExecutableType::get(executableOp.getContext());
    auto variableOp = moduleBuilder.create<VariableOp>(
        executableOp.getLoc(), symbolName, /*isMutable=*/true, executableType);
    SymbolTable::setSymbolVisibility(variableOp,
                                     SymbolTable::Visibility::Private);
    executableCache_.try_emplace(executableOp.sym_name(), variableOp);
    return variableOp;
  }

  VariableOp defineDescriptorSetLayoutOp(Location loc, ArrayAttr bindingsAttr) {
    auto existingIt = descriptorSetLayoutCache_.find(bindingsAttr);
    if (existingIt != descriptorSetLayoutCache_.end()) {
      return existingIt->second;
    }

    auto symbolName = (StringRef("_descriptor_set_layout_") +
                       std::to_string(nextUniqueDescriptorSetLayoutId++))
                          .str();
    auto initializerName = symbolName + "_initializer";

    auto layoutType = DescriptorSetLayoutType::get(loc.getContext());
    auto variableOp = moduleBuilder.create<VariableOp>(
        loc, symbolName,
        /*isMutable=*/false, layoutType, StringRef(initializerName),
        llvm::None);
    SymbolTable::setSymbolVisibility(variableOp,
                                     SymbolTable::Visibility::Private);
    descriptorSetLayoutCache_.try_emplace(bindingsAttr, variableOp);

    auto initializerOp = moduleBuilder.create<FuncOp>(
        loc, initializerName, moduleBuilder.getFunctionType({}, {layoutType}),
        ArrayRef<NamedAttribute>{});
    SymbolTable::setSymbolVisibility(initializerOp,
                                     SymbolTable::Visibility::Private);
    auto *block = initializerOp.addEntryBlock();
    OpBuilder blockBuilder = OpBuilder::atBlockEnd(block);
    auto deviceValue = blockBuilder.createOrFold<ExSharedDeviceOp>(loc);
    auto layoutUsage = IREE::HAL::DescriptorSetLayoutUsageType::PushOnly;
    auto layoutValue = blockBuilder.createOrFold<DescriptorSetLayoutCreateOp>(
        loc, layoutType, deviceValue, layoutUsage, bindingsAttr);
    blockBuilder.create<mlir::ReturnOp>(loc, layoutValue);

    return variableOp;
  }

  VariableOp defineExecutableLayoutOp(Location loc,
                                      ArrayAttr setLayoutsArrayAttr,
                                      IntegerAttr pushConstantsAttr) {
    auto existingIt = executableLayoutCache_.find(setLayoutsArrayAttr);
    if (existingIt != executableLayoutCache_.end()) {
      return existingIt->second;
    }

    // First lookup (or create) all the required descriptor sets. This ensures
    // they end up in the proper initialization order.
    SmallVector<VariableOp, 4> setLayoutVariableOps;
    for (auto setLayoutsAttr : setLayoutsArrayAttr) {
      setLayoutVariableOps.push_back(
          defineDescriptorSetLayoutOp(loc, setLayoutsAttr.cast<ArrayAttr>()));
    }

    auto symbolName = (StringRef("_executable_layout_") +
                       std::to_string(nextUniqueExecutableLayoutId++))
                          .str();
    auto initializerName = symbolName + "_initializer";

    auto layoutType = ExecutableLayoutType::get(loc.getContext());
    auto variableOp = moduleBuilder.create<VariableOp>(
        loc, symbolName, /*isMutable=*/false, layoutType,
        StringRef(initializerName), llvm::None);
    SymbolTable::setSymbolVisibility(variableOp,
                                     SymbolTable::Visibility::Private);
    executableLayoutCache_.try_emplace(setLayoutsArrayAttr, variableOp);

    auto initializerOp = moduleBuilder.create<FuncOp>(
        loc, initializerName, moduleBuilder.getFunctionType({}, {layoutType}),
        ArrayRef<NamedAttribute>{});
    SymbolTable::setSymbolVisibility(initializerOp,
                                     SymbolTable::Visibility::Private);
    auto *block = initializerOp.addEntryBlock();
    OpBuilder blockBuilder = OpBuilder::atBlockEnd(block);
    SmallVector<Value, 4> setLayoutValues;
    for (auto setLayoutVariableOp : setLayoutVariableOps) {
      auto setLayoutValue = blockBuilder.createOrFold<VariableLoadOp>(
          loc, DescriptorSetLayoutType::get(loc.getContext()),
          setLayoutVariableOp.sym_name());
      setLayoutValues.push_back(setLayoutValue);
    }
    auto deviceValue = blockBuilder.createOrFold<ExSharedDeviceOp>(loc);
    // Push constants are optional but we always provide the value.
    if (!pushConstantsAttr) {
      pushConstantsAttr = blockBuilder.getI32IntegerAttr(0);
    }
    auto layoutValue = blockBuilder.createOrFold<ExecutableLayoutCreateOp>(
        loc, layoutType, deviceValue, setLayoutValues, pushConstantsAttr);
    blockBuilder.create<mlir::ReturnOp>(loc, layoutValue);

    return variableOp;
  }

  VariableOp defineExecutableCacheOp(ArrayRef<ExecutableOp> executableOps) {
    auto loc = moduleBuilder.getUnknownLoc();

    auto symbolName = std::string("_executable_cache");
    auto initializerName = symbolName + "_initializer";

    auto executableCacheType = ExecutableCacheType::get(loc.getContext());
    auto variableOp = moduleBuilder.create<VariableOp>(
        loc, symbolName,
        /*isMutable=*/false, executableCacheType, StringRef(initializerName),
        llvm::None);

    // TODO(#1146): we define this as public right now to ensure it remains
    // after DCE.
    SymbolTable::setSymbolVisibility(variableOp,
                                     SymbolTable::Visibility::Public);

    auto initializerOp = moduleBuilder.create<FuncOp>(
        loc, initializerName,
        moduleBuilder.getFunctionType({}, {executableCacheType}),
        ArrayRef<NamedAttribute>{});
    SymbolTable::setSymbolVisibility(initializerOp,
                                     SymbolTable::Visibility::Private);
    auto *block = initializerOp.addEntryBlock();
    OpBuilder blockBuilder = OpBuilder::atBlockEnd(block);
    auto deviceValue = blockBuilder.createOrFold<ExSharedDeviceOp>(loc);
    auto executableCacheValue =
        blockBuilder.createOrFold<ExecutableCacheCreateOp>(
            loc, executableCacheType, deviceValue,
            blockBuilder.getStringAttr("default"));

    // TODO(benvanik): use targetOptions_ to determine these flags.
    auto cachingMode = ExecutableCachingModeBitfield::AliasProvidedData |
                       ExecutableCachingModeBitfield::AllowPersistentCaching |
                       ExecutableCachingModeBitfield::AllowOptimization;
    for (auto executableOp : executableOps) {
      auto executableIt = executableCache_.find(executableOp.sym_name());
      assert(executableIt != executableCache_.end() &&
             "executable must have been cached");
      auto executableVariableOp = executableIt->second;

      // TODO(benvanik): support multiple interfaces. We'd probably want to
      // store each executable+interface as a variable.
      auto interfaceOp = executableOp.getInterfaceOp();

      auto executableLayoutVariableOp = defineExecutableLayoutOp(
          executableOp.getLoc(), interfaceOp.getExecutableSetLayoutsAttr(),
          interfaceOp.push_constantsAttr());
      auto executableLayoutValue = blockBuilder.createOrFold<VariableLoadOp>(
          loc, ExecutableLayoutType::get(loc.getContext()),
          executableLayoutVariableOp.sym_name());
      auto executableValue =
          blockBuilder.createOrFold<ExecutableCachePrepareOp>(
              loc, ExecutableType::get(loc.getContext()), executableCacheValue,
              executableLayoutValue, cachingMode, executableOp.sym_name());
      blockBuilder.create<VariableStoreOp>(loc, executableValue,
                                           executableVariableOp.sym_name());
    }

    blockBuilder.create<mlir::ReturnOp>(loc, executableCacheValue);

    return variableOp;
  }

  void replaceDescriptorSetLayoutLookupOp(
      DescriptorSetLayoutLookupOp &lookupOp) {
    OpBuilder builder(lookupOp);
    auto variableOp =
        defineDescriptorSetLayoutOp(lookupOp.getLoc(), lookupOp.bindings());
    auto loadOp = builder.create<VariableLoadOp>(
        lookupOp.getLoc(), DescriptorSetLayoutType::get(lookupOp.getContext()),
        variableOp.sym_name());
    lookupOp.replaceAllUsesWith(loadOp.getOperation());
    lookupOp.erase();
  }

  void replaceExecutableLayoutLookupOp(ExecutableLayoutLookupOp &lookupOp) {
    OpBuilder builder(lookupOp);
    auto variableOp =
        defineExecutableLayoutOp(lookupOp.getLoc(), lookupOp.set_layouts(),
                                 lookupOp.push_constantsAttr());
    auto loadOp = builder.create<VariableLoadOp>(
        lookupOp.getLoc(), ExecutableLayoutType::get(lookupOp.getContext()),
        variableOp.sym_name());
    lookupOp.replaceAllUsesWith(loadOp.getOperation());
    lookupOp.erase();
  }

  void replaceExecutableLookupOp(ExecutableLookupOp &lookupOp) {
    OpBuilder builder(lookupOp);
    auto executableIt = executableCache_.find(lookupOp.executable());
    assert(executableIt != executableCache_.end() &&
           "executable must have been cached");
    auto variableOp = executableIt->second;
    auto loadOp = builder.create<VariableLoadOp>(
        lookupOp.getLoc(), ExecutableType::get(lookupOp.getContext()),
        variableOp.sym_name());
    lookupOp.replaceAllUsesWith(loadOp.getOperation());
    lookupOp.erase();
  }

  TargetOptions targetOptions_;

  OpBuilder moduleBuilder{static_cast<MLIRContext *>(nullptr)};
  DenseMap<Attribute, VariableOp> descriptorSetLayoutCache_;
  DenseMap<Attribute, VariableOp> executableLayoutCache_;
  DenseMap<StringRef, VariableOp> executableCache_;

  int nextUniqueExecutableLayoutId = 0;
  int nextUniqueDescriptorSetLayoutId = 0;
};

std::unique_ptr<OperationPass<ModuleOp>> createMaterializeResourceCachesPass(
    TargetOptions targetOptions) {
  return std::make_unique<MaterializeResourceCachesPass>(
      targetOptions);  // NOLINT
}

static PassRegistration<MaterializeResourceCachesPass> pass(
    "iree-hal-materialize-resource-caches",
    "Materializes hal.executable resource caches and rewrites lookups.");

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
