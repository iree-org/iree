// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Utils/DeviceSwitchBuilder.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
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

class MaterializeResourceCachesPass
    : public PassWrapper<MaterializeResourceCachesPass,
                         OperationPass<ModuleOp>> {
 public:
  explicit MaterializeResourceCachesPass(TargetOptions targetOptions)
      : targetOptions_(targetOptions) {}

  StringRef getArgument() const override {
    return "iree-hal-materialize-resource-caches";
  }

  StringRef getDescription() const override {
    return "Materializes hal.executable resource caches and rewrites lookups.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty()) return;
    moduleBuilder = OpBuilder(&moduleOp.getBody()->front());

    auto executableOps = llvm::to_vector<8>(moduleOp.getOps<ExecutableOp>());

    // Declare all layouts used by the executables. This will ensure that the
    // initialization order is correct as any executable layout needed (and its
    // dependencies) will be created prior to the executable cache below. The
    // other nice thing is that we get ordering similar to the executable
    // variables above.
    for (auto executableOp : executableOps) {
      for (auto interfaceOp :
           executableOp.getBlock().getOps<IREE::HAL::InterfaceOp>()) {
        defineExecutableLayoutOp(interfaceOp.getLoc(),
                                 interfaceOp.getExecutableSetLayoutsAttr(),
                                 interfaceOp.push_constantsAttr());
      }
    }

    // Declare executable variables so that we can reference them during lookup
    // replacement.
    for (auto executableOp : executableOps) {
      if (!defineExecutableOp(executableOp)) {
        signalPassFailure();
        return;
      }
    }

    // Generate cached resource singletons and replace lookup ops with direct
    // loads from variables.
    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      for (auto &block : funcOp) {
        block.walk([&](Operation *op) {
          if (auto lookupOp = dyn_cast<DescriptorSetLayoutLookupOp>(op)) {
            replaceDescriptorSetLayoutLookupOp(lookupOp);
          } else if (auto lookupOp = dyn_cast<ExecutableLayoutLookupOp>(op)) {
            replaceExecutableLayoutLookupOp(lookupOp);
          } else if (auto lookupOp = dyn_cast<ExecutableLookupOp>(op)) {
            replaceExecutableLookupOp(lookupOp);
          }
        });
      }
    }
  }

 private:
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
    variableOp.setPrivate();
    descriptorSetLayoutCache_.try_emplace(bindingsAttr, variableOp);

    auto initializerOp = moduleBuilder.create<FuncOp>(
        loc, initializerName, moduleBuilder.getFunctionType({}, {layoutType}));
    initializerOp.setPrivate();
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
    // Push constants are optional but we always provide the value.
    if (!pushConstantsAttr) {
      pushConstantsAttr = IntegerAttr::get(IndexType::get(loc.getContext()), 0);
    }

    // We key the layout cache on all attributes that compose an executable
    // layout.
    auto cacheKey = ArrayAttr::get(loc.getContext(),
                                   {setLayoutsArrayAttr, pushConstantsAttr});

    auto existingIt = executableLayoutCache_.find(cacheKey);
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
    variableOp.setPrivate();
    executableLayoutCache_.try_emplace(cacheKey, variableOp);

    auto initializerOp = moduleBuilder.create<FuncOp>(
        loc, initializerName, moduleBuilder.getFunctionType({}, {layoutType}));
    initializerOp.setPrivate();
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
    auto layoutValue = blockBuilder.createOrFold<ExecutableLayoutCreateOp>(
        loc, layoutType, deviceValue, pushConstantsAttr, setLayoutValues);
    blockBuilder.create<mlir::ReturnOp>(loc, layoutValue);

    return variableOp;
  }

  VariableOp defineExecutableOp(ExecutableOp executableOp) {
    auto loc = executableOp.getLoc();
    auto symbolName =
        (StringRef("_executable_") + executableOp.sym_name()).str();
    auto initializerName = symbolName + "_initializer";

    auto executableType = ExecutableType::get(executableOp.getContext());
    auto variableOp = moduleBuilder.create<VariableOp>(
        loc, symbolName, /*isMutable=*/false, executableType,
        StringRef(initializerName), llvm::None);
    variableOp.setPrivate();
    executableCache_.try_emplace(executableOp.sym_name(), variableOp);

    auto initializerOp = moduleBuilder.create<FuncOp>(
        loc, initializerName,
        moduleBuilder.getFunctionType({}, {executableType}));
    initializerOp.setPrivate();
    auto *block = initializerOp.addEntryBlock();
    OpBuilder blockBuilder = OpBuilder::atBlockEnd(block);
    auto deviceValue = blockBuilder.createOrFold<ExSharedDeviceOp>(loc);

    // Create a switch statement with a case for each backend.
    // Each case should then cache only executables which contain a matching
    // ExecutableVariantOp.
    // Afterwards, canonicalization will take care of de-duping/etc.
    DeviceSwitchBuilder switchBuilder(loc,
                                      /*resultTypes=*/TypeRange{executableType},
                                      deviceValue, blockBuilder);
    auto targetBackends = getTargetBackends(targetOptions_.targets);
    for (auto &targetBackend : targetBackends) {
      // Skip executables with no matching target ops.
      SmallVector<IREE::HAL::ExecutableVariantOp> executableVariantOps;
      for (auto executableVariantOp :
           executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
        if (executableVariantOp.target() == targetBackend->name()) {
          executableVariantOps.push_back(executableVariantOp);
        }
      }
      if (executableVariantOps.empty()) continue;

      // TODO(benvanik): support multiple target executables by adding a device
      // switch on supported format. This needs a new device match attr type.
      if (executableVariantOps.size() > 1) {
        executableOp.emitError()
            << "multiple matching executable targets are not yet supported";
        return nullptr;
      }
      auto executableVariantOp = executableVariantOps.front();

      auto *region = switchBuilder.addConditionRegion(
          IREE::HAL::DeviceMatchIDAttr::get(targetBackend->deviceID(),
                                            blockBuilder.getContext()),
          {deviceValue});
      auto &entryBlock = region->front();
      auto caseBuilder = OpBuilder::atBlockBegin(&entryBlock);
      auto caseDeviceValue = entryBlock.getArgument(0);

      // Gather each of the executable layouts needed for each entry point in
      // the executable.
      SmallVector<Value, 8> executableLayoutValues;
      for (auto entryPointOp :
           executableVariantOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
        auto interfaceOp =
            SymbolTable::lookupNearestSymbolFrom<IREE::HAL::InterfaceOp>(
                executableOp, entryPointOp.interface());
        assert(interfaceOp && "must have an interface available");
        auto executableLayoutVariableOp = defineExecutableLayoutOp(
            executableOp.getLoc(), interfaceOp.getExecutableSetLayoutsAttr(),
            interfaceOp.push_constantsAttr());
        executableLayoutValues.push_back(
            caseBuilder.createOrFold<VariableLoadOp>(
                loc, ExecutableLayoutType::get(loc.getContext()),
                executableLayoutVariableOp.sym_name()));
      }

      auto executableValue = caseBuilder.createOrFold<ExecutableCreateOp>(
          loc, ExecutableType::get(loc.getContext()), caseDeviceValue,
          SymbolRefAttr::get(
              loc.getContext(), executableOp.sym_name(),
              {SymbolRefAttr::get(loc.getContext(),
                                  executableVariantOp.sym_name())}),
          executableLayoutValues);

      caseBuilder.create<IREE::HAL::ReturnOp>(loc, executableValue);
    }

    auto *defaultRegion = switchBuilder.addConditionRegion(
        IREE::HAL::MatchAlwaysAttr::get(loc.getContext()), {});
    auto defaultBuilder = OpBuilder::atBlockBegin(&defaultRegion->front());
    auto nullValue =
        defaultBuilder.createOrFold<IREE::NullOp>(loc, executableType);
    defaultBuilder.create<IREE::HAL::ReturnOp>(loc, nullValue);

    auto switchOp = switchBuilder.build();
    auto executableValue = switchOp.getResult(0);
    blockBuilder.create<mlir::ReturnOp>(loc, executableValue);

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
  return std::make_unique<MaterializeResourceCachesPass>(targetOptions);
}

static PassRegistration<MaterializeResourceCachesPass> pass([] {
  auto options = getTargetOptionsFromFlags();
  return std::make_unique<MaterializeResourceCachesPass>(options);
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
