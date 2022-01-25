// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Utils/DeviceSwitchBuilder.h"
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
      for (auto variantOp :
           executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
        for (auto entryPointOp :
             variantOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
          defineExecutableLayoutOp(entryPointOp.getLoc(),
                                   entryPointOp.layout());
        }
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
    for (Operation &funcLikeOp : moduleOp.getOps()) {
      auto funcOp = llvm::dyn_cast<FunctionOpInterface>(funcLikeOp);
      if (!funcOp) continue;
      for (auto &block : funcOp.getBody()) {
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
  IREE::Util::GlobalOp defineDescriptorSetLayoutOp(Location loc,
                                                   ArrayAttr bindingAttrs) {
    auto existingIt = descriptorSetLayoutCache_.find(bindingAttrs);
    if (existingIt != descriptorSetLayoutCache_.end()) {
      return existingIt->second;
    }

    auto symbolName = (StringRef("_descriptor_set_layout_") +
                       std::to_string(nextUniqueDescriptorSetLayoutId++))
                          .str();

    auto layoutType = DescriptorSetLayoutType::get(loc.getContext());
    auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
        loc, symbolName,
        /*isMutable=*/false, layoutType);
    globalOp.setPrivate();
    descriptorSetLayoutCache_.try_emplace(bindingAttrs, globalOp);

    auto initializerOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);
    OpBuilder blockBuilder =
        OpBuilder::atBlockEnd(initializerOp.addEntryBlock());
    auto deviceValue = blockBuilder.createOrFold<ExSharedDeviceOp>(loc);
    auto layoutUsage = IREE::HAL::DescriptorSetLayoutUsageType::PushOnly;
    auto layoutValue = blockBuilder.createOrFold<DescriptorSetLayoutCreateOp>(
        loc, layoutType, deviceValue, layoutUsage, bindingAttrs);
    blockBuilder.create<IREE::Util::GlobalStoreOp>(loc, layoutValue,
                                                   globalOp.getName());
    blockBuilder.create<IREE::Util::InitializerReturnOp>(loc);

    return globalOp;
  }

  IREE::Util::GlobalOp defineExecutableLayoutOp(
      Location loc, IREE::HAL::ExecutableLayoutAttr layoutAttr) {
    auto existingIt = executableLayoutCache_.find(layoutAttr);
    if (existingIt != executableLayoutCache_.end()) {
      return existingIt->second;
    }

    // First lookup (or create) all the required descriptor sets. This ensures
    // they end up in the proper initialization order.
    SmallVector<IREE::Util::GlobalOp, 4> setLayoutGlobalOps;
    for (auto setLayoutAttr : layoutAttr.getSetLayouts()) {
      SmallVector<Attribute> bindingAttrs;
      for (auto bindingAttr : setLayoutAttr.getBindings()) {
        bindingAttrs.push_back(bindingAttr);
      }
      setLayoutGlobalOps.push_back(defineDescriptorSetLayoutOp(
          loc, ArrayAttr::get(loc.getContext(), bindingAttrs)));
    }

    auto symbolName = (StringRef("_executable_layout_") +
                       std::to_string(nextUniqueExecutableLayoutId++))
                          .str();

    auto layoutType = ExecutableLayoutType::get(loc.getContext());
    auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
        loc, symbolName, /*isMutable=*/false, layoutType);
    globalOp.setPrivate();
    executableLayoutCache_.try_emplace(layoutAttr, globalOp);

    auto initializerOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);
    OpBuilder blockBuilder =
        OpBuilder::atBlockEnd(initializerOp.addEntryBlock());
    SmallVector<Value, 4> setLayoutValues;
    for (auto setLayoutGlobalOp : setLayoutGlobalOps) {
      auto setLayoutValue = blockBuilder.createOrFold<IREE::Util::GlobalLoadOp>(
          loc, DescriptorSetLayoutType::get(loc.getContext()),
          setLayoutGlobalOp.sym_name());
      setLayoutValues.push_back(setLayoutValue);
    }
    auto deviceValue = blockBuilder.createOrFold<ExSharedDeviceOp>(loc);
    auto layoutValue = blockBuilder.createOrFold<ExecutableLayoutCreateOp>(
        loc, layoutType, deviceValue,
        blockBuilder.getIndexAttr(layoutAttr.getPushConstants()),
        setLayoutValues);
    blockBuilder.create<IREE::Util::GlobalStoreOp>(loc, layoutValue,
                                                   globalOp.getName());
    blockBuilder.create<IREE::Util::InitializerReturnOp>(loc);

    return globalOp;
  }

  IREE::Util::GlobalOp defineExecutableOp(ExecutableOp executableOp) {
    auto loc = executableOp.getLoc();
    auto symbolName =
        (StringRef("_executable_") + executableOp.sym_name()).str();

    auto executableType = ExecutableType::get(executableOp.getContext());
    auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
        loc, symbolName, /*isMutable=*/false, executableType);
    globalOp.setPrivate();
    executableCache_.try_emplace(executableOp.sym_name(), globalOp);

    auto initializerOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);
    OpBuilder blockBuilder =
        OpBuilder::atBlockEnd(initializerOp.addEntryBlock());
    auto deviceValue = blockBuilder.createOrFold<ExSharedDeviceOp>(loc);

    // Create a switch statement with a case for each variant.
    // Each case should then cache only executables which contain a matching
    // ExecutableVariantOp.
    // Afterwards, canonicalization will take care of de-duping/etc.
    DeviceSwitchBuilder switchBuilder(loc,
                                      /*resultTypes=*/TypeRange{executableType},
                                      deviceValue, blockBuilder);
    for (auto executableVariantOp :
         executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
      auto *region = switchBuilder.addConditionRegion(
          executableVariantOp.target().getMatchExpression());
      auto &entryBlock = region->front();
      auto caseBuilder = OpBuilder::atBlockBegin(&entryBlock);

      // Gather each of the executable layouts needed for each entry point in
      // the executable.
      SmallVector<Value, 8> executableLayoutValues;
      for (auto entryPointOp :
           executableVariantOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
        auto executableLayoutGlobalOp = defineExecutableLayoutOp(
            executableOp.getLoc(), entryPointOp.layout());
        executableLayoutValues.push_back(
            caseBuilder.createOrFold<IREE::Util::GlobalLoadOp>(
                loc, ExecutableLayoutType::get(loc.getContext()),
                executableLayoutGlobalOp.sym_name()));
      }

      auto executableValue = caseBuilder.createOrFold<ExecutableCreateOp>(
          loc, ExecutableType::get(loc.getContext()), deviceValue,
          SymbolRefAttr::get(
              executableOp.sym_nameAttr(),
              {SymbolRefAttr::get(executableVariantOp.sym_nameAttr())}),
          executableLayoutValues);

      caseBuilder.create<IREE::HAL::ReturnOp>(loc, executableValue);
    }

    auto *defaultRegion = switchBuilder.addConditionRegion(
        IREE::HAL::MatchAlwaysAttr::get(loc.getContext()));
    auto defaultBuilder = OpBuilder::atBlockBegin(&defaultRegion->front());
    auto nullValue =
        defaultBuilder.createOrFold<IREE::Util::NullOp>(loc, executableType);
    defaultBuilder.create<IREE::HAL::ReturnOp>(loc, nullValue);

    auto switchOp = switchBuilder.build();
    auto executableValue = switchOp.getResult(0);
    blockBuilder.create<IREE::Util::GlobalStoreOp>(loc, executableValue,
                                                   globalOp.getName());
    blockBuilder.create<IREE::Util::InitializerReturnOp>(loc);

    return globalOp;
  }

  void replaceDescriptorSetLayoutLookupOp(
      DescriptorSetLayoutLookupOp &lookupOp) {
    OpBuilder builder(lookupOp);
    auto globalOp =
        defineDescriptorSetLayoutOp(lookupOp.getLoc(), lookupOp.bindings());
    auto loadOp = builder.create<IREE::Util::GlobalLoadOp>(
        lookupOp.getLoc(), DescriptorSetLayoutType::get(lookupOp.getContext()),
        globalOp.sym_name());
    lookupOp.replaceAllUsesWith(loadOp.getOperation());
    lookupOp.erase();
  }

  void replaceExecutableLayoutLookupOp(ExecutableLayoutLookupOp &lookupOp) {
    OpBuilder builder(lookupOp);
    auto globalOp =
        defineExecutableLayoutOp(lookupOp.getLoc(), lookupOp.layout());
    auto loadOp = builder.create<IREE::Util::GlobalLoadOp>(
        lookupOp.getLoc(), ExecutableLayoutType::get(lookupOp.getContext()),
        globalOp.sym_name());
    lookupOp.replaceAllUsesWith(loadOp.getOperation());
    lookupOp.erase();
  }

  void replaceExecutableLookupOp(ExecutableLookupOp &lookupOp) {
    OpBuilder builder(lookupOp);
    auto executableIt = executableCache_.find(lookupOp.executable());
    assert(executableIt != executableCache_.end() &&
           "executable must have been cached");
    auto globalOp = executableIt->second;
    auto loadOp = builder.create<IREE::Util::GlobalLoadOp>(
        lookupOp.getLoc(), ExecutableType::get(lookupOp.getContext()),
        globalOp.sym_name());
    lookupOp.replaceAllUsesWith(loadOp.getOperation());
    lookupOp.erase();
  }

  TargetOptions targetOptions_;

  OpBuilder moduleBuilder{static_cast<MLIRContext *>(nullptr)};
  DenseMap<Attribute, IREE::Util::GlobalOp> descriptorSetLayoutCache_;
  DenseMap<Attribute, IREE::Util::GlobalOp> executableLayoutCache_;
  DenseMap<StringRef, IREE::Util::GlobalOp> executableCache_;

  int nextUniqueExecutableLayoutId = 0;
  int nextUniqueDescriptorSetLayoutId = 0;
};

std::unique_ptr<OperationPass<ModuleOp>> createMaterializeResourceCachesPass(
    TargetOptions targetOptions) {
  return std::make_unique<MaterializeResourceCachesPass>(targetOptions);
}

static PassRegistration<MaterializeResourceCachesPass> pass([] {
  auto options = TargetOptions::FromFlags::get();
  return std::make_unique<MaterializeResourceCachesPass>(options);
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
