// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Utils/DeviceSwitchBuilder.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::cf::ControlFlowDialect>();
    registry.insert<IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty()) return;
    moduleBuilder = OpBuilder(&moduleOp.getBody()->front());

    // Find all relevant ops. If we don't find any we skip the pass as it's
    // likely it's already been run. We could fix the pass to better support
    // partial materialization but there's no use cases for that today.
    auto executableOps = llvm::to_vector<8>(moduleOp.getOps<ExecutableOp>());
    SmallVector<IREE::HAL::DescriptorSetLayoutLookupOp>
        descriptorSetLayoutLookupOps;
    SmallVector<IREE::HAL::PipelineLayoutLookupOp> pipelineLayoutLookupOps;
    SmallVector<IREE::HAL::ExecutableLookupOp> executableLookupOps;
    for (Operation &funcLikeOp : moduleOp.getOps()) {
      auto funcOp = llvm::dyn_cast<FunctionOpInterface>(funcLikeOp);
      if (!funcOp) continue;
      for (auto &block : funcOp.getFunctionBody()) {
        block.walk([&](Operation *op) {
          if (auto lookupOp = dyn_cast<DescriptorSetLayoutLookupOp>(op)) {
            descriptorSetLayoutLookupOps.push_back(lookupOp);
          } else if (auto lookupOp = dyn_cast<PipelineLayoutLookupOp>(op)) {
            pipelineLayoutLookupOps.push_back(lookupOp);
          } else if (auto lookupOp = dyn_cast<ExecutableLookupOp>(op)) {
            executableLookupOps.push_back(lookupOp);
          }
        });
      }
    }
    if (descriptorSetLayoutLookupOps.empty() &&
        pipelineLayoutLookupOps.empty() && executableLookupOps.empty()) {
      return;
    }

    // Declare all layouts used by the executables. This will ensure that the
    // initialization order is correct as any pipeline layout needed (and its
    // dependencies) will be created prior to the executable cache below. The
    // other nice thing is that we get ordering similar to the executable
    // variables above.
    for (auto executableOp : executableOps) {
      for (auto variantOp :
           executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
        for (auto exportOp :
             variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
          definePipelineLayoutOp(exportOp.getLoc(), exportOp.getLayout());
        }
      }
    }

    // Declare executable variables so that we can reference them during lookup
    // replacement.
    for (auto executableOp : executableOps) {
      defineExecutableOp(executableOp);
    }

    // Generate cached resource singletons and replace lookup ops with direct
    // loads from variables.
    for (auto lookupOp : descriptorSetLayoutLookupOps) {
      replaceDescriptorSetLayoutLookupOp(lookupOp);
    }
    for (auto lookupOp : pipelineLayoutLookupOps) {
      replacePipelineLayoutLookupOp(lookupOp);
    }
    for (auto lookupOp : executableLookupOps) {
      replaceExecutableLookupOp(lookupOp);
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
    auto layoutFlags = IREE::HAL::DescriptorSetLayoutFlags::None;
    auto layoutValue = blockBuilder.createOrFold<DescriptorSetLayoutCreateOp>(
        loc, layoutType, deviceValue, layoutFlags, bindingAttrs);
    blockBuilder.create<IREE::Util::GlobalStoreOp>(loc, layoutValue,
                                                   globalOp.getName());
    blockBuilder.create<IREE::Util::InitializerReturnOp>(loc);

    return globalOp;
  }

  IREE::Util::GlobalOp definePipelineLayoutOp(
      Location loc, IREE::HAL::PipelineLayoutAttr layoutAttr) {
    auto existingIt = pipelineLayoutCache_.find(layoutAttr);
    if (existingIt != pipelineLayoutCache_.end()) {
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

    auto symbolName = (StringRef("_pipeline_layout_") +
                       std::to_string(nextUniquePipelineLayoutId++))
                          .str();

    auto layoutType = PipelineLayoutType::get(loc.getContext());
    auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
        loc, symbolName, /*isMutable=*/false, layoutType);
    globalOp.setPrivate();
    pipelineLayoutCache_.try_emplace(layoutAttr, globalOp);

    auto initializerOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);
    OpBuilder blockBuilder =
        OpBuilder::atBlockEnd(initializerOp.addEntryBlock());
    SmallVector<Value, 4> setLayoutValues;
    for (auto setLayoutGlobalOp : setLayoutGlobalOps) {
      auto setLayoutValue = blockBuilder.createOrFold<IREE::Util::GlobalLoadOp>(
          loc, DescriptorSetLayoutType::get(loc.getContext()),
          setLayoutGlobalOp.getSymName());
      setLayoutValues.push_back(setLayoutValue);
    }
    auto deviceValue = blockBuilder.createOrFold<ExSharedDeviceOp>(loc);
    auto layoutValue = blockBuilder.createOrFold<PipelineLayoutCreateOp>(
        loc, layoutType, deviceValue,
        blockBuilder.getIndexAttr(layoutAttr.getPushConstants()),
        setLayoutValues);
    blockBuilder.create<IREE::Util::GlobalStoreOp>(loc, layoutValue,
                                                   globalOp.getName());
    blockBuilder.create<IREE::Util::InitializerReturnOp>(loc);

    return globalOp;
  }

  void defineExecutableOp(ExecutableOp executableOp) {
    auto loc = executableOp.getLoc();
    auto symbolName =
        (StringRef("_executable_") + executableOp.getSymName()).str();

    auto executableType = ExecutableType::get(executableOp.getContext());
    auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
        loc, symbolName, /*isMutable=*/false, executableType);
    globalOp.setPrivate();
    executableCache_.try_emplace(executableOp.getSymName(), globalOp);

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
          executableVariantOp.getTarget().getMatchExpression());
      auto &entryBlock = region->front();
      auto caseBuilder = OpBuilder::atBlockBegin(&entryBlock);

      // Gather each of the pipeline layouts needed for each entry point in
      // the executable.
      SmallVector<Value, 8> pipelineLayoutValues;
      for (auto exportOp :
           executableVariantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
        auto pipelineLayoutGlobalOp =
            definePipelineLayoutOp(executableOp.getLoc(), exportOp.getLayout());
        pipelineLayoutValues.push_back(
            caseBuilder.createOrFold<IREE::Util::GlobalLoadOp>(
                loc, PipelineLayoutType::get(loc.getContext()),
                pipelineLayoutGlobalOp.getSymName()));
      }

      // Inline constant initializer from the variant.
      // We want these to all happen inside of this device switch case; they'll
      // get deduplicated/hoisted if possible in future canonicalization passes.
      SmallVector<Value> constantValues;
      for (auto blockOp : llvm::make_early_inc_range(
               executableVariantOp
                   .getOps<IREE::HAL::ExecutableConstantBlockOp>())) {
        constantValues.append(inlineConstantBlockOp(blockOp, moduleBuilder,
                                                    caseBuilder, deviceValue));
        blockOp.erase();
      }

      auto executableValue = caseBuilder.createOrFold<ExecutableCreateOp>(
          loc, ExecutableType::get(loc.getContext()), deviceValue,
          SymbolRefAttr::get(
              executableOp.getSymNameAttr(),
              {SymbolRefAttr::get(executableVariantOp.getSymNameAttr())}),
          pipelineLayoutValues, constantValues);

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
  }

  // Inlines a constant block as a function in |moduleBuilder| and then inserts
  // a call to it in |callerBuilder|.
  SmallVector<Value> inlineConstantBlockOp(ExecutableConstantBlockOp blockOp,
                                           OpBuilder &moduleBuilder,
                                           OpBuilder &callerBuilder,
                                           Value deviceValue) {
    // Create the function with the region contents of the constant block.
    auto funcName = (StringRef("__constant_block_") +
                     std::to_string(nextUniqueConstantBlockId++))
                        .str();
    auto funcOp = moduleBuilder.create<func::FuncOp>(blockOp.getLoc(), funcName,
                                                     blockOp.getFunctionType());
    funcOp.setPrivate();
    funcOp.getRegion().takeBody(blockOp.getRegion());

    // Replace the hal.return with a func.return.
    for (auto returnOp :
         llvm::make_early_inc_range(funcOp.getOps<IREE::HAL::ReturnOp>())) {
      OpBuilder(returnOp).create<func::ReturnOp>(returnOp.getLoc(),
                                                 returnOp.getOperands());
      returnOp.erase();
    }

    // Create the call passing in the device if needed.
    SmallVector<Value> callOperands;
    if (funcOp.getNumArguments() > 0) {
      callOperands.push_back(deviceValue);
    }
    auto callOp = callerBuilder.create<func::CallOp>(blockOp.getLoc(), funcOp,
                                                     callOperands);

    return llvm::to_vector(llvm::map_range(
        callOp.getResults(), [](OpResult result) -> Value { return result; }));
  }

  void replaceDescriptorSetLayoutLookupOp(
      DescriptorSetLayoutLookupOp &lookupOp) {
    OpBuilder builder(lookupOp);
    auto globalOp =
        defineDescriptorSetLayoutOp(lookupOp.getLoc(), lookupOp.getBindings());
    auto loadOp = builder.create<IREE::Util::GlobalLoadOp>(
        lookupOp.getLoc(), DescriptorSetLayoutType::get(lookupOp.getContext()),
        globalOp.getSymName());
    lookupOp.replaceAllUsesWith(loadOp.getOperation());
    lookupOp.erase();
  }

  void replacePipelineLayoutLookupOp(PipelineLayoutLookupOp &lookupOp) {
    OpBuilder builder(lookupOp);
    auto globalOp =
        definePipelineLayoutOp(lookupOp.getLoc(), lookupOp.getLayout());
    auto loadOp = builder.create<IREE::Util::GlobalLoadOp>(
        lookupOp.getLoc(), PipelineLayoutType::get(lookupOp.getContext()),
        globalOp.getSymName());
    lookupOp.replaceAllUsesWith(loadOp.getOperation());
    lookupOp.erase();
  }

  void replaceExecutableLookupOp(ExecutableLookupOp &lookupOp) {
    OpBuilder builder(lookupOp);
    auto executableIt = executableCache_.find(lookupOp.getExecutable());
    assert(executableIt != executableCache_.end() &&
           "executable must have been cached");
    auto globalOp = executableIt->second;
    auto loadOp = builder.create<IREE::Util::GlobalLoadOp>(
        lookupOp.getLoc(), ExecutableType::get(lookupOp.getContext()),
        globalOp.getSymName());
    lookupOp.replaceAllUsesWith(loadOp.getOperation());
    lookupOp.erase();
  }

  TargetOptions targetOptions_;

  OpBuilder moduleBuilder{static_cast<MLIRContext *>(nullptr)};
  DenseMap<Attribute, IREE::Util::GlobalOp> descriptorSetLayoutCache_;
  DenseMap<Attribute, IREE::Util::GlobalOp> pipelineLayoutCache_;
  DenseMap<StringRef, IREE::Util::GlobalOp> executableCache_;

  int nextUniqueConstantBlockId = 0;
  int nextUniquePipelineLayoutId = 0;
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
