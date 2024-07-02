// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-hal-materialize-resource-caches"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_MATERIALIZERESOURCECACHESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-materialize-resource-caches
//===----------------------------------------------------------------------===//

struct DescriptorSetLayout {
  // All locations that use the layout.
  SetVector<Location> locs;
  // Value within the initializer once materialized.
  Value initializerValue;
};
using DescriptorSetLayoutKey =
    std::pair<ArrayAttr, IREE::HAL::DescriptorSetLayoutFlags>;

struct PipelineLayout {
  // All locations that use the layout.
  SetVector<Location> locs;
  // Lookup ops for this layout.
  SmallVector<IREE::HAL::PipelineLayoutLookupOp> lookupOps;
  // Global once materialized.
  IREE::Util::GlobalOpInterface globalOp;
  // Value within the initializer once materialized.
  Value initializerValue;
};

struct Executable {
  // All locations that use the executable.
  SetVector<Location> locs;
  // Executable representing the program to load.
  IREE::HAL::ExecutableOp executableOp;
  // Lookup ops for this executable.
  SmallVector<IREE::HAL::ExecutableLookupOp> lookupOps;
  // Global once materialized.
  IREE::Util::GlobalOpInterface globalOp;
};

struct DeviceResources {
  DeviceResources() = default;
  explicit DeviceResources(IREE::Util::GlobalOpInterface deviceOp)
      : deviceOp(deviceOp) {}

  // Global !hal.device.
  IREE::Util::GlobalOpInterface deviceOp;

  // Fallback devices that should be checked for resources.
  // These are derived from the transitive set of #hal.device.fallback attrs.
  SetVector<DeviceResources *> fallbackDeviceResources;

  // Descriptor set layouts used on the device, keyed by [bindingAttrs, flags].
  llvm::MapVector<DescriptorSetLayoutKey, DescriptorSetLayout>
      descriptorSetLayouts;
  // Pipeline layouts used on the device, keyed by layout attr.
  llvm::MapVector<IREE::HAL::PipelineLayoutAttr, PipelineLayout>
      pipelineLayouts;
  // Executables used on the device, keyed by name.
  llvm::MapVector<StringAttr, Executable> executables;
};

static std::string getDeviceNamePrefix(IREE::Util::GlobalOpInterface deviceOp) {
  StringRef deviceName = deviceOp.getGlobalName().getValue();
  if (deviceName.starts_with("__")) {
    // Already prefixed.
    return deviceName.str();
  }
  return ("__" + deviceName).str();
}

static void declareDevicePipelineLayout(IREE::Util::GlobalOpInterface deviceOp,
                                        PipelineLayout &pipelineLayout,
                                        size_t pipelineLayoutIndex,
                                        OpBuilder &moduleBuilder) {
  // Create global in the module.
  auto symbolName = getDeviceNamePrefix(deviceOp) + "_pipeline_layout_" +
                    std::to_string(pipelineLayoutIndex);
  LLVM_DEBUG(DBGS() << "+ creating device `"
                    << deviceOp.getGlobalName().getValue()
                    << "` pipeline global `" << symbolName << "`\n");
  auto layoutType = moduleBuilder.getType<PipelineLayoutType>();
  auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
      moduleBuilder.getFusedLoc(llvm::to_vector(pipelineLayout.locs)),
      symbolName,
      /*isMutable=*/false, layoutType);
  globalOp.setPrivate();
  pipelineLayout.globalOp = globalOp;

  // Replace lookups with the global.
  for (auto lookupOp : pipelineLayout.lookupOps) {
    LLVM_DEBUG({
      DBGS() << "  - replacing lookup: ";
      lookupOp.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    OpBuilder lookupBuilder(lookupOp);
    auto loadedValue =
        pipelineLayout.globalOp.createLoadOp(lookupOp.getLoc(), lookupBuilder)
            .getLoadedGlobalValue();
    lookupOp.replaceAllUsesWith(loadedValue);
    lookupOp.erase();
  }
  pipelineLayout.lookupOps.clear();
}

static void declareDeviceExecutable(IREE::Util::GlobalOpInterface deviceOp,
                                    Executable &executable,
                                    size_t executableIndex,
                                    OpBuilder &moduleBuilder) {
  // Create global in the module.
  auto symbolName = (getDeviceNamePrefix(deviceOp) + "_executable_" +
                     std::to_string(executableIndex) + "_" +
                     executable.executableOp.getName())
                        .str();
  LLVM_DEBUG(DBGS() << "+ creating device `"
                    << deviceOp.getGlobalName().getValue()
                    << "` executable global `" << symbolName << "`\n");
  auto executableType = moduleBuilder.getType<IREE::HAL::ExecutableType>();
  auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
      moduleBuilder.getFusedLoc(llvm::to_vector(executable.locs)), symbolName,
      /*isMutable=*/false, executableType);
  globalOp.setPrivate();
  executable.globalOp = globalOp;

  // Replace lookups with the global.
  for (auto lookupOp : executable.lookupOps) {
    LLVM_DEBUG({
      DBGS() << "  - replacing lookup: ";
      lookupOp.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });
    OpBuilder lookupBuilder(lookupOp);
    auto loadedValue =
        executable.globalOp.createLoadOp(lookupOp.getLoc(), lookupBuilder)
            .getLoadedGlobalValue();
    lookupOp.replaceAllUsesWith(loadedValue);
    lookupOp.erase();
  }
  executable.lookupOps.clear();
}

static DescriptorSetLayoutKey
getDescriptorSetLayoutKey(IREE::HAL::DescriptorSetLayoutAttr setLayoutAttr) {
  auto bindingAttrs =
      llvm::to_vector_of<Attribute>(setLayoutAttr.getBindings());
  return DescriptorSetLayoutKey{
      ArrayAttr::get(setLayoutAttr.getContext(), bindingAttrs),
      setLayoutAttr.getFlags().value_or(
          IREE::HAL::DescriptorSetLayoutFlags::None),
  };
}

// Inlines a constant block as a function in |moduleBuilder| and then inserts
// a call to it in |callerBuilder|.
static SmallVector<Value> inlineConstantBlockOp(
    StringRef funcName, IREE::HAL::ExecutableConstantBlockOp blockOp,
    OpBuilder &moduleBuilder, OpBuilder &callerBuilder, Value callerDevice) {
  LLVM_DEBUG(DBGS() << "- inlining constant block `" << funcName << "`\n");

  // Create the function with the region contents of the constant block.
  auto funcOp = moduleBuilder.create<IREE::Util::FuncOp>(
      blockOp.getLoc(), funcName, blockOp.getFunctionType());
  funcOp.setPrivate();
  IRMapping mapping;
  blockOp.getRegion().cloneInto(&funcOp.getRegion(), mapping);

  // Replace the hal.return with a func.return.
  for (auto returnOp :
       llvm::make_early_inc_range(funcOp.getOps<IREE::HAL::ReturnOp>())) {
    OpBuilder(returnOp).create<IREE::Util::ReturnOp>(returnOp.getLoc(),
                                                     returnOp.getOperands());
    returnOp.erase();
  }

  // Create the call passing in the device if needed.
  SmallVector<Value> callOperands;
  if (funcOp.getNumArguments() > 0) {
    callOperands.push_back(callerDevice);
  }
  auto callOp = callerBuilder.create<IREE::Util::CallOp>(blockOp.getLoc(),
                                                         funcOp, callOperands);
  return llvm::to_vector_of<Value>(callOp.getResults());
}

static Value initializeExecutable(DeviceResources &deviceResources,
                                  Executable &executable,
                                  OpBuilder &moduleBuilder,
                                  Value initializerDevice,
                                  OpBuilder &initializerBuilder) {
  auto loc = executable.globalOp.getLoc();
  auto executableType = moduleBuilder.getType<IREE::HAL::ExecutableType>();

  // Create a switch statement with a case for each variant.
  // Each case should then cache only executables which contain a matching
  // ExecutableVariantOp.
  // Afterwards, canonicalization will take care of de-duping/etc.
  SmallVector<int64_t> caseIndices;
  SmallVector<IREE::HAL::ExecutableVariantOp> caseVariantOps;
  for (auto variantOp :
       executable.executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
    caseIndices.push_back(caseIndices.size());
    caseVariantOps.push_back(variantOp);
  }

  // Select the variant index.
  Value selectedIndex = buildIfElseTree(
      loc, caseVariantOps.size(),
      [&](Location loc, size_t i, OpBuilder &builder) {
        return caseVariantOps[i].buildCondition(initializerDevice, builder);
      },
      initializerBuilder);

  // Allow each variant to define how it is loaded and what pipeline it has.
  auto switchOp = initializerBuilder.create<scf::IndexSwitchOp>(
      loc, executableType, selectedIndex, caseIndices, caseIndices.size());
  for (auto [i, variantOp] : llvm::enumerate(caseVariantOps)) {
    auto &caseBlock = switchOp.getCaseRegions()[i].emplaceBlock();
    auto caseBuilder = OpBuilder::atBlockBegin(&caseBlock);

    // Gather each of the pipeline layouts needed for each entry point in
    // the executable.
    SmallVector<Value> pipelineLayoutValues;
    for (auto exportOp : variantOp.getExportOps()) {
      auto &pipelineLayout =
          deviceResources.pipelineLayouts[exportOp.getLayoutAttr()];
      pipelineLayoutValues.push_back(pipelineLayout.initializerValue);
    }

    // Inline constant initializer from the variant.
    // We want these to all happen inside of this device switch case; they'll
    // get deduplicated/hoisted if possible in future canonicalization passes.
    SmallVector<Value> constantValues;
    for (auto [blockIndex, blockOp] :
         llvm::enumerate(variantOp.getConstantBlockOps())) {
      auto blockName = (executable.globalOp.getGlobalName().getValue() +
                        "_constant_block_" + std::to_string(blockIndex))
                           .str();
      constantValues.append(inlineConstantBlockOp(
          blockName, blockOp, moduleBuilder, caseBuilder, initializerDevice));
    }

    Value executableValue =
        caseBuilder.createOrFold<IREE::HAL::ExecutableCreateOp>(
            loc, executableType, initializerDevice,
            SymbolRefAttr::get(
                executable.executableOp.getSymNameAttr(),
                {SymbolRefAttr::get(variantOp.getSymNameAttr())}),
            pipelineLayoutValues, constantValues);

    caseBuilder.create<scf::YieldOp>(loc, executableValue);
  }

  // Fallback for no available variant.
  auto &defaultBlock = switchOp.getDefaultRegion().emplaceBlock();
  auto defaultBuilder = OpBuilder::atBlockBegin(&defaultBlock);
  Value status = defaultBuilder.create<arith::ConstantIntOp>(
      loc, static_cast<int>(IREE::Util::StatusCode::Unavailable), 32);
  {
    std::string errorStr;
    llvm::raw_string_ostream errorStream(errorStr);
    errorStream << "HAL device `"
                << deviceResources.deviceOp.getGlobalName().getValue()
                << "` does not support any variant of executable `"
                << executable.executableOp.getName()
                << "`; available formats: [";
    llvm::interleaveComma(caseVariantOps, errorStream, [&](auto variantOp) {
      errorStream << variantOp.getTargetAttr().getFormat().getValue();
    });
    errorStream << "]";
    defaultBuilder.create<IREE::Util::StatusCheckOkOp>(loc, status, errorStr);
  }
  auto nullValue =
      defaultBuilder.createOrFold<IREE::Util::NullOp>(loc, executableType);
  defaultBuilder.create<scf::YieldOp>(loc, nullValue);

  return switchOp.getResult(0);
}

static void initializeDeviceResources(DeviceResources &deviceResources,
                                      OpBuilder &moduleBuilder,
                                      Value initializerDevice,
                                      OpBuilder &initializerBuilder) {
  // Initialize all descriptor set layouts for use by the pipeline layouts.
  auto setLayoutType = initializerBuilder.getType<DescriptorSetLayoutType>();
  for (auto [i, it] : llvm::enumerate(deviceResources.descriptorSetLayouts)) {
    auto [bindingAttrs, flags] = it.first;
    auto &descriptorSetLayout = it.second;
    descriptorSetLayout.initializerValue =
        initializerBuilder.createOrFold<IREE::HAL::DescriptorSetLayoutCreateOp>(
            initializerBuilder.getFusedLoc(
                llvm::to_vector(descriptorSetLayout.locs)),
            setLayoutType, initializerDevice, flags, bindingAttrs);
  }

  // Initialize all pipeline layouts required for executable creation.
  auto pipelineLayoutType = initializerBuilder.getType<PipelineLayoutType>();
  for (auto [i, it] : llvm::enumerate(deviceResources.pipelineLayouts)) {
    auto &[layoutAttr, pipelineLayout] = it;
    SmallVector<Value> setLayoutValues;
    for (auto setLayoutAttr : layoutAttr.getSetLayouts()) {
      auto key = getDescriptorSetLayoutKey(setLayoutAttr);
      setLayoutValues.push_back(
          deviceResources.descriptorSetLayouts[key].initializerValue);
    }
    pipelineLayout.initializerValue =
        initializerBuilder.createOrFold<IREE::HAL::PipelineLayoutCreateOp>(
            pipelineLayout.globalOp.getLoc(), pipelineLayoutType,
            initializerDevice,
            initializerBuilder.getIndexAttr(layoutAttr.getPushConstants()),
            setLayoutValues);
    pipelineLayout.globalOp.createStoreOp(pipelineLayout.globalOp.getLoc(),
                                          pipelineLayout.initializerValue,
                                          initializerBuilder);
  }

  // Initialize all executables.
  for (auto [i, it] : llvm::enumerate(deviceResources.executables)) {
    auto &[executableName, executable] = it;
    executable.globalOp.createStoreOp(
        executable.globalOp.getLoc(),
        initializeExecutable(deviceResources, executable, moduleBuilder,
                             initializerDevice, initializerBuilder),
        initializerBuilder);
  }
}

static void reuseFallbackDeviceResources(DeviceResources &deviceResources,
                                         DeviceResources &fallbackResources,
                                         Value initializerDevice,
                                         OpBuilder &initializerBuilder) {
  // Load fallback pipeline layouts for all required by this device.
  for (auto &[layoutAttr, pipelineLayout] : deviceResources.pipelineLayouts) {
    auto fallbackGlobalOp =
        fallbackResources.pipelineLayouts[layoutAttr].globalOp;
    assert(fallbackGlobalOp && "should have created global");
    Value fallbackPipelineLayout =
        fallbackGlobalOp
            .createLoadOp(pipelineLayout.globalOp.getLoc(), initializerBuilder)
            .getLoadedGlobalValue();
    pipelineLayout.globalOp.createStoreOp(pipelineLayout.globalOp.getLoc(),
                                          fallbackPipelineLayout,
                                          initializerBuilder);
  }

  // Load fallback executables for all required by this device.
  for (auto &[executableName, executable] : deviceResources.executables) {
    auto fallbackGlobalOp =
        fallbackResources.executables[executable.executableOp.getNameAttr()]
            .globalOp;
    assert(fallbackGlobalOp && "should have created global");
    Value fallbackExecutable =
        fallbackGlobalOp
            .createLoadOp(executable.globalOp.getLoc(), initializerBuilder)
            .getLoadedGlobalValue();
    executable.globalOp.createStoreOp(executable.globalOp.getLoc(),
                                      fallbackExecutable, initializerBuilder);
  }
}

static void buildDeviceResourceInitializer(DeviceResources &deviceResources,
                                           OpBuilder &moduleBuilder) {
  auto loc = deviceResources.deviceOp.getLoc();
  auto initializerOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);
  OpBuilder initializerBuilder =
      OpBuilder::atBlockEnd(initializerOp.addEntryBlock());
  Value initializerDevice =
      deviceResources.deviceOp.createLoadOp(loc, initializerBuilder)
          .getLoadedGlobalValue();

  // If there are any fallbacks then we need to handle referencing their
  // resources and otherwise will initialize our own.
  if (deviceResources.fallbackDeviceResources.empty()) {
    initializeDeviceResources(deviceResources, moduleBuilder, initializerDevice,
                              initializerBuilder);
  } else {
    SmallVector<int64_t> caseIndices;
    Value selectedIndex = buildIfElseTree(
        loc, deviceResources.fallbackDeviceResources.size(),
        [&](Location loc, size_t i, OpBuilder &caseBuilder) {
          caseIndices.push_back(caseIndices.size());
          auto *fallbackResources = deviceResources.fallbackDeviceResources[i];
          Value fallbackDevice =
              fallbackResources->deviceOp.createLoadOp(loc, caseBuilder)
                  .getLoadedGlobalValue();
          return caseBuilder.create<IREE::Util::CmpEQOp>(loc, initializerDevice,
                                                         fallbackDevice);
        },
        initializerBuilder);
    auto switchOp = initializerBuilder.create<scf::IndexSwitchOp>(
        loc, TypeRange{}, selectedIndex, caseIndices, caseIndices.size());
    for (auto [fallbackResources, caseRegion] :
         llvm::zip_equal(deviceResources.fallbackDeviceResources,
                         switchOp.getCaseRegions())) {
      auto &caseBlock = caseRegion.emplaceBlock();
      auto caseBuilder = OpBuilder::atBlockBegin(&caseBlock);
      reuseFallbackDeviceResources(deviceResources, *fallbackResources,
                                   initializerDevice, caseBuilder);
      caseBuilder.create<scf::YieldOp>(loc);
    }
    auto &defaultBlock = switchOp.getDefaultRegion().emplaceBlock();
    auto defaultBuilder = OpBuilder::atBlockBegin(&defaultBlock);
    initializeDeviceResources(deviceResources, moduleBuilder, initializerDevice,
                              defaultBuilder);
    defaultBuilder.create<scf::YieldOp>(loc);
  }

  initializerBuilder.create<IREE::Util::ReturnOp>(loc);
}

// Returns zero or more devices globals that may act as fallbacks for the
// given device, if analyzed. The result is in selection order.
static std::optional<SetVector<IREE::Util::GlobalOpInterface>>
getDeviceFallbackGlobals(IREE::Util::GlobalOpInterface deviceGlobal,
                         SymbolTable &symbolTable) {
  SetVector<IREE::Util::GlobalOpInterface> resultSet;
  auto processAttr = [&](Attribute attr) {
    if (!attr)
      return true; // ignore uninitialized devices
    return TypeSwitch<Attribute, bool>(attr)
        .Case<IREE::HAL::DeviceOrdinalAttr>([](auto attr) { return true; })
        .Case<IREE::HAL::DeviceTargetAttr>([](auto attr) { return true; })
        .Case<IREE::HAL::DeviceFallbackAttr>([&](auto fallbackAttr) {
          resultSet.insert(symbolTable.lookup<IREE::Util::GlobalOpInterface>(
              fallbackAttr.getName().getValue()));
          return true;
        })
        .Default([](auto attr) { return false; });
  };
  auto initialValue = deviceGlobal.getGlobalInitialValue();
  if (auto selectAttr =
          dyn_cast_if_present<IREE::HAL::DeviceSelectAttr>(initialValue)) {
    for (auto deviceAttr : selectAttr.getDevices()) {
      if (!processAttr(deviceAttr)) {
        // Fails if unsupported/unhandled device attribute type.
        return std::nullopt;
      }
    }
  } else {
    if (!processAttr(initialValue)) {
      // Fails if unsupported/unhandled device attribute type.
      return std::nullopt;
    }
  }
  return resultSet;
}

static LogicalResult gatherDeviceResources(
    ModuleOp &moduleOp, SymbolTable &symbolTable,
    DeviceAnalysis &deviceAnalysis,
    llvm::MapVector<Attribute, DeviceResources> &allDeviceResources) {
  // Allocate storage for the resource sets.
  for (auto deviceOp : deviceAnalysis.getDeviceGlobals()) {
    LLVM_DEBUG(DBGS() << "Gathering device `"
                      << deviceOp.getGlobalName().getValue()
                      << "` resources...\n");
    allDeviceResources.try_emplace(deviceOp.getGlobalName(),
                                   DeviceResources(deviceOp));
  }

  // Link fallbacks between the resources.
  for (auto deviceOp : deviceAnalysis.getDeviceGlobals()) {
    auto fallbackOps = getDeviceFallbackGlobals(deviceOp, symbolTable);
    if (!fallbackOps) {
      return deviceOp->emitOpError()
             << "analysis failed on device; currently analysis must succeed";
    }
    auto &deviceResources = allDeviceResources[deviceOp.getGlobalName()];
    for (auto fallbackOp : *fallbackOps) {
      LLVM_DEBUG(DBGS() << "* linking to fallback `"
                        << fallbackOp.getGlobalName().getValue() << "`\n");
      deviceResources.fallbackDeviceResources.insert(
          &allDeviceResources[fallbackOp.getGlobalName()]);
    }
  }

  // Find all relevant ops. If we don't find any we skip the pass as it's
  // likely it's already been run. We could fix the pass to better support
  // partial materialization but there's no use cases for that today.
  auto tryGetDeviceResources = [&](Operation *op,
                                   Value device) -> DeviceResources * {
    auto deviceGlobals = deviceAnalysis.lookupDeviceGlobals(device);
    if (!deviceGlobals || deviceGlobals->size() != 1) {
      op->emitOpError() << "analysis failed on device; currently analysis "
                           "must succeed with a single device";
      return nullptr;
    }
    auto deviceOp = deviceGlobals->front();
    return &allDeviceResources.find(deviceOp.getGlobalName())->second;
  };
  for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
    for (auto &block : funcOp.getFunctionBody()) {
      if (block
              .walk([&](Operation *op) -> WalkResult {
                if (auto lookupOp = dyn_cast<PipelineLayoutLookupOp>(op)) {
                  auto *deviceResources =
                      tryGetDeviceResources(lookupOp, lookupOp.getDevice());
                  if (!deviceResources) {
                    return WalkResult::interrupt();
                  }
                  auto layoutAttr = lookupOp.getLayoutAttr();
                  LLVM_DEBUG(DBGS()
                             << "+ requiring pipeline layout from lookup: `"
                             << layoutAttr << "`\n");
                  auto &pipelineLayout =
                      deviceResources->pipelineLayouts[layoutAttr];
                  pipelineLayout.locs.insert(lookupOp.getLoc());
                  pipelineLayout.lookupOps.push_back(lookupOp);
                  for (auto setLayoutAttr : layoutAttr.getSetLayouts()) {
                    LLVM_DEBUG(
                        DBGS()
                        << "+ requiring descriptor set layout from lookup: `"
                        << setLayoutAttr << "`\n");
                    auto key = getDescriptorSetLayoutKey(setLayoutAttr);
                    auto &setLayout =
                        deviceResources->descriptorSetLayouts[key];
                    setLayout.locs.insert(lookupOp.getLoc());
                  }
                } else if (auto lookupOp = dyn_cast<ExecutableLookupOp>(op)) {
                  auto *deviceResources =
                      tryGetDeviceResources(lookupOp, lookupOp.getDevice());
                  if (!deviceResources) {
                    return WalkResult::interrupt();
                  }
                  auto executableAttr = lookupOp.getExecutableAttr().getAttr();
                  LLVM_DEBUG(DBGS() << "+ requiring executable from lookup: `"
                                    << executableAttr.getValue() << "`\n");
                  auto &executable =
                      deviceResources->executables[executableAttr];
                  executable.locs.insert(lookupOp.getLoc());
                  executable.lookupOps.push_back(lookupOp);
                }
                return WalkResult::advance();
              })
              .wasInterrupted()) {
        return failure();
      }
    }
  }

  // Gather the executables referenced by all lookup ops.
  for (auto &[deviceName, deviceResources] : allDeviceResources) {
    for (auto &[executableName, executable] : deviceResources.executables) {
      executable.executableOp =
          symbolTable.lookup<IREE::HAL::ExecutableOp>(executableName);
      for (auto variantOp :
           executable.executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
        for (auto exportOp : variantOp.getExportOps()) {
          auto layoutAttr = exportOp.getLayoutAttr();
          LLVM_DEBUG(DBGS() << "+ requiring pipeline layout from export: `"
                            << layoutAttr << "`\n");
          auto &pipelineLayout = deviceResources.pipelineLayouts[layoutAttr];
          pipelineLayout.locs.insert(exportOp.getLoc());
          for (auto setLayoutAttr : layoutAttr.getSetLayouts()) {
            LLVM_DEBUG(DBGS()
                       << "+ requiring descriptor set layout from export: `"
                       << setLayoutAttr << "`\n");
            auto key = getDescriptorSetLayoutKey(setLayoutAttr);
            auto &setLayout = deviceResources.descriptorSetLayouts[key];
            setLayout.locs.insert(exportOp.getLoc());
          }
        }
      }
    }
  }

  // Merge all resources that may be used by way of fallbacks into each fallback
  // device. We could make this optional to improve startup performance by
  // adding these as optional and create them on demand but that's more complex.
  // For now we just always ensure the resources are available even if they end
  // up unused.
  for (auto &[deviceName, deviceResources] :
       llvm::reverse(allDeviceResources)) {
    for (auto *fallbackResources : deviceResources.fallbackDeviceResources) {
      LLVM_DEBUG(
          DBGS() << "-> requiring fallback resources from device `"
                 << fallbackResources->deviceOp.getGlobalName().getValue()
                 << "`\n");
      for (auto [setKey, setLayout] : deviceResources.descriptorSetLayouts) {
        auto &fallbackSetLayout =
            fallbackResources->descriptorSetLayouts[setKey];
        fallbackSetLayout.locs.insert(setLayout.locs.begin(),
                                      setLayout.locs.end());
      }
      for (auto [layoutAttr, pipelineLayout] :
           deviceResources.pipelineLayouts) {
        auto &fallbackPipelineLayout =
            fallbackResources->pipelineLayouts[layoutAttr];
        fallbackPipelineLayout.locs.insert(pipelineLayout.locs.begin(),
                                           pipelineLayout.locs.end());
      }
      for (auto [executableName, executable] : deviceResources.executables) {
        auto &fallbackExecutable =
            fallbackResources->executables[executableName];
        fallbackExecutable.locs.insert(executable.locs.begin(),
                                       executable.locs.end());
        fallbackExecutable.executableOp = executable.executableOp;
      }
    }
  }

  return success();
}

struct MaterializeResourceCachesPass
    : public IREE::HAL::impl::MaterializeResourceCachesPassBase<
          MaterializeResourceCachesPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    // Analyze the module to determine which devices are used where.
    LLVM_DEBUG(DBGS() << "Running device analysis...\n");
    DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run())) {
      return signalPassFailure();
    }

    // Build a table of all resources used by all devices in the program.
    LLVM_DEBUG(DBGS() << "Gathering device resources...\n");
    llvm::MapVector<Attribute, DeviceResources> allDeviceResources;
    if (failed(gatherDeviceResources(moduleOp, symbolTable, deviceAnalysis,
                                     allDeviceResources))) {
      return signalPassFailure();
    }

    // Materialize resources for each device (if any) and replace lookups.
    for (auto &[nameAttr, deviceResources] : allDeviceResources) {
      LLVM_DEBUG(DBGS() << "Materializing device `"
                        << deviceResources.deviceOp.getGlobalName().getValue()
                        << "` resources...\n");
      // Skip devices with no resources.
      if (deviceResources.pipelineLayouts.empty() &&
          deviceResources.executables.empty()) {
        LLVM_DEBUG(DBGS() << "~ skipping device with no resources\n");
        continue;
      }

      // TODO(benvanik): proper insertion order if devices are initialized via
      // an initializer. Today this assumes the device hasn't been materialized
      // yet if there are any lookups to them.
      if (!deviceResources.deviceOp.getGlobalInitialValue()) {
        deviceResources.deviceOp.emitOpError()
            << "is expected to be initialized with an attribute and not yet "
               "via a util.initializer";
        return signalPassFailure();
      }

      // Declare globals for each pipeline layout and executable and replace all
      // lookup ops to reference them.
      OpBuilder moduleBuilder(moduleOp);
      moduleBuilder.setInsertionPointAfter(deviceResources.deviceOp);
      for (auto [i, it] : llvm::enumerate(deviceResources.pipelineLayouts)) {
        auto &[layoutAttr, pipelineLayout] = it;
        declareDevicePipelineLayout(deviceResources.deviceOp, pipelineLayout, i,
                                    moduleBuilder);
      }
      for (auto [i, it] : llvm::enumerate(deviceResources.executables)) {
        auto &[executableName, executable] = it;
        declareDeviceExecutable(deviceResources.deviceOp, executable, i,
                                moduleBuilder);
      }

      // Create an initializer after the declared globals.
      buildDeviceResourceInitializer(deviceResources, moduleBuilder);
    }

    // Remove ops that are no longer required after materialization.
    for (auto executableOp : moduleOp.getOps<IREE::HAL::ExecutableOp>()) {
      for (auto variantOp :
           executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
        if (auto conditionOp = variantOp.getConditionOp()) {
          conditionOp.erase();
        }
        for (auto blockOp :
             llvm::make_early_inc_range(variantOp.getConstantBlockOps())) {
          blockOp.erase();
        }
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
