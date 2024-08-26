// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/HAL/Analysis/Captures.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_OUTLINEMEMOIZEREGIONSPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-outline-memoize-regions
//===----------------------------------------------------------------------===//

static std::string getMemoizeNamePrefix(IREE::HAL::DeviceMemoizeOp memoizeOp) {
  auto parentOp = memoizeOp->getParentOfType<FunctionOpInterface>();
  return ("__" + parentOp.getName() + "_memoize").str();
}

static StringRef getDeviceName(IREE::Util::GlobalOpInterface deviceOp) {
  StringRef deviceName = deviceOp.getGlobalName().getValue();
  return deviceName.starts_with("__") ? deviceName.substr(2) : deviceName;
}

// Results of a hal.device.memoize captured value analysis.
struct MemoizeAnalysis {
  // Queue affinity mask, if constant.
  std::optional<int64_t> queueAffinity;

  // Captured constant values in the original memoization site.
  SetVector<Value> constantValues;
  // Captured immutable loaded global values in the original memoization site.
  SetVector<Value> globalValues;
  // Captured dynamic values in the original memoization site.
  SetVector<Value> dynamicValues;

  // Location of each result returned from the region.
  SmallVector<Location> resultLocs;

  // Returns true if the region can be run at initialization time (all values
  // are derived from ones available at initialization time).
  bool canRunAtInitializationTime() const { return dynamicValues.empty(); }
};

// Analyzes a |memoizeOp| to find captured values and filter them by
// hoistability.
static MemoizeAnalysis
computeMemoizeAnalysis(IREE::HAL::DeviceMemoizeOp memoizeOp) {
  MemoizeAnalysis memoizeAnalysis;

  // Try to match constant affinities. This is the majority case today.
  APInt queueAffinityAttr;
  if (matchPattern(memoizeOp.getQueueAffinity(),
                   m_ConstantInt(&queueAffinityAttr))) {
    memoizeAnalysis.queueAffinity = queueAffinityAttr.getSExtValue();
  }

  // Get all values defined above (constants, globals, devices, etc).
  SetVector<Value> capturedValues;
  mlir::getUsedValuesDefinedAbove(memoizeOp.getBody(), capturedValues);

  // Filter out values by type.
  for (auto capturedValue : capturedValues) {
    // Ignore device/affinity as they will be substituted.
    if (capturedValue == memoizeOp.getDevice() ||
        capturedValue == memoizeOp.getQueueAffinity()) {
      continue;
    }

    // Match ops by type.
    // If we wanted to pull in entire IR slices this would have to use a
    // worklist (selects of globals based on globals, etc).
    switch (categorizeValue(capturedValue)) {
    default:
    case ValueOrigin::Unknown:
    case ValueOrigin::MutableGlobal:
      memoizeAnalysis.dynamicValues.insert(capturedValue);
      break;
    case ValueOrigin::LocalConstant:
      memoizeAnalysis.constantValues.insert(capturedValue);
      break;
    case ValueOrigin::ImmutableGlobal:
      memoizeAnalysis.globalValues.insert(capturedValue);
      break;
    }
  }

  // Gather the locations of all results in the region. This lets us preserve
  // fine-grained location information on each global instead of just using the
  // fused memoize op location.
  SmallVector<SetVector<Location>> allResultLocs;
  allResultLocs.resize(memoizeOp.getNumResults());
  for (auto returnOp : memoizeOp.getOps<IREE::HAL::ReturnOp>()) {
    for (auto result : llvm::enumerate(returnOp.getOperands())) {
      allResultLocs[result.index()].insert(result.value().getLoc());
    }
  }
  for (auto &resultLocs : allResultLocs) {
    Location resultLoc =
        resultLocs.size() == 1
            ? resultLocs.front()
            : FusedLoc::get(memoizeOp.getContext(), resultLocs.getArrayRef());
    memoizeAnalysis.resultLocs.push_back(resultLoc);
  }

  return memoizeAnalysis;
}

// Creates a function with the body of the given |memoizeOp|.
// Arguments will be added for device, affinity, and all captured non-constant
// values.
static IREE::Util::FuncOp outlineMemoizeRegionBody(
    IREE::HAL::DeviceMemoizeOp memoizeOp, MemoizeAnalysis &memoizeAnalysis,
    SymbolTable &moduleSymbolTable, OpBuilder &moduleBuilder) {
  auto name = getMemoizeNamePrefix(memoizeOp) + "_apply";

  // Create the function in the module with device, affinity, and dynamic args.
  SmallVector<Type> argTypes;
  argTypes.push_back(memoizeOp.getDevice().getType());
  argTypes.push_back(memoizeOp.getQueueAffinity().getType());
  llvm::append_range(
      argTypes, llvm::map_range(memoizeAnalysis.globalValues,
                                [](Value value) { return value.getType(); }));
  llvm::append_range(
      argTypes, llvm::map_range(memoizeAnalysis.dynamicValues,
                                [](Value value) { return value.getType(); }));
  auto funcType =
      moduleBuilder.getFunctionType(argTypes, memoizeOp.getResultTypes());
  auto funcOp = moduleBuilder.create<IREE::Util::FuncOp>(memoizeOp.getLoc(),
                                                         name, funcType);
  moduleSymbolTable.insert(funcOp);
  funcOp.setVisibility(SymbolTable::Visibility::Private);
  auto funcBuilder = OpBuilder::atBlockBegin(funcOp.addEntryBlock());

  // Remap any captured operands that have corresponding function arguments.
  auto capturedArgs = funcOp.getArguments();
  IRMapping mapping;
  mapping.map(memoizeOp.getDevice(), capturedArgs[0]);
  capturedArgs[0].setLoc(memoizeOp.getDevice().getLoc());
  mapping.map(memoizeOp.getQueueAffinity(), capturedArgs[1]);
  capturedArgs[1].setLoc(memoizeOp.getQueueAffinity().getLoc());
  capturedArgs = capturedArgs.drop_front(2);
  for (auto [callerValue, calleeValue] : llvm::zip_equal(
           memoizeAnalysis.globalValues,
           capturedArgs.take_front(memoizeAnalysis.globalValues.size()))) {
    mapping.map(callerValue, calleeValue);
    calleeValue.setLoc(callerValue.getLoc());
  }
  capturedArgs = capturedArgs.drop_front(memoizeAnalysis.globalValues.size());
  for (auto [callerValue, calleeValue] : llvm::zip_equal(
           memoizeAnalysis.dynamicValues,
           capturedArgs.take_front(memoizeAnalysis.dynamicValues.size()))) {
    mapping.map(callerValue, calleeValue);
    calleeValue.setLoc(callerValue.getLoc());
  }
  capturedArgs = capturedArgs.drop_front(memoizeAnalysis.dynamicValues.size());

  // Clone all constant values into the function.
  //
  // TODO(benvanik): take constants as args? That may allow for more sharing by
  // relying on IPO to inline uniform constant operands. Then structurally
  // equivalent command buffer recordings that use different constants (offsets
  // etc) could still share the same recording code. It's strictly a program
  // size optimization, though, so we don't bother now.
  for (auto callerValue : memoizeAnalysis.constantValues) {
    auto *callerOp = callerValue.getDefiningOp();
    funcBuilder.clone(*callerOp, mapping);
  }

  // Inline region using the mapping table to remap any captured values to
  // arguments or the cloned values.
  funcBuilder.cloneRegionBefore(memoizeOp.getBody(), funcOp.getRegion(),
                                funcOp.getRegion().end(), mapping);

  // cloneRegionBefore is unfriendly and requires that we poke into the block
  // list to get the first block and insert the branch to it.
  auto &entryBlock = funcOp.getBlocks().front();
  OpBuilder::atBlockEnd(&entryBlock)
      .create<cf::BranchOp>(memoizeOp.getLoc(), entryBlock.getNextNode());

  // Rewrite hal.return ops to util.return.
  for (auto returnOp :
       llvm::make_early_inc_range(funcOp.getOps<IREE::HAL::ReturnOp>())) {
    OpBuilder returnBuilder(returnOp);
    returnBuilder.create<IREE::Util::ReturnOp>(returnOp.getLoc(),
                                               returnOp.getOperands());
    returnOp.erase();
  }

  return funcOp;
}

// Replaces a |memoizeOp| with a call to the given |applyFuncOp|.
static void replaceMemoizeOpWithApply(IREE::HAL::DeviceMemoizeOp memoizeOp,
                                      MemoizeAnalysis &memoizeAnalysis,
                                      IREE::Util::FuncOp applyFuncOp) {
  // TODO(benvanik): set one-shot on any command buffer in the region? This may
  // be unsafe depending on if the memoize region is nested within another
  // memoize region (or something) so we don't do it here now. It's probably OK
  // to just run through and update any command buffer create op.

  // Pass the device, affinity, and all dynamic operands.
  SmallVector<Value> callOperands;
  callOperands.push_back(memoizeOp.getDevice());
  callOperands.push_back(memoizeOp.getQueueAffinity());
  llvm::append_range(callOperands, memoizeAnalysis.globalValues);
  llvm::append_range(callOperands, memoizeAnalysis.dynamicValues);

  // Call the function.
  OpBuilder callerBuilder(memoizeOp);
  auto callOp = callerBuilder.create<IREE::Util::CallOp>(
      memoizeOp.getLoc(), applyFuncOp, callOperands);

  // Replace memoize op with the results of the function call.
  memoizeOp.replaceAllUsesWith(callOp.getResults());
  memoizeOp.erase();
}

// Creates globals for each result of a memoize region with the result types.
// The globals will be for the specified |deviceGlobal|.
static SmallVector<IREE::Util::GlobalOpInterface> createMemoizedDeviceGlobals(
    IREE::HAL::DeviceMemoizeOp memoizeOp, MemoizeAnalysis &memoizeAnalysis,
    IREE::Util::FuncOp applyFuncOp, IREE::Util::GlobalOpInterface deviceGlobal,
    SymbolTable &moduleSymbolTable, OpBuilder &moduleBuilder) {
  auto namePrefix = getMemoizeNamePrefix(memoizeOp);
  auto deviceName = getDeviceName(deviceGlobal);

  // Creates the globals for the given device.
  SmallVector<IREE::Util::GlobalOpInterface> resultGlobalOps;
  for (auto [resultLoc, result] :
       llvm::zip_equal(memoizeAnalysis.resultLocs, memoizeOp.getResults())) {
    auto globalName =
        (namePrefix + "_result_" + std::to_string(result.getResultNumber()) +
         "_" + deviceName)
            .str();
    auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
        resultLoc, globalName, /*isMutable=*/false, result.getType());
    moduleSymbolTable.insert(globalOp);
    globalOp.setVisibility(SymbolTable::Visibility::Private);
    resultGlobalOps.push_back(globalOp);
  }

  // Create an initializer to call the apply function and store the results into
  // globals.
  auto initializerOp =
      moduleBuilder.create<IREE::Util::InitializerOp>(memoizeOp.getLoc());
  auto initializerBuilder =
      OpBuilder::atBlockBegin(initializerOp.addEntryBlock());

  IRMapping mapping;

  // Get the device this set of results is being created for.
  auto deviceLoadOp =
      deviceGlobal.createLoadOp(memoizeOp.getLoc(), initializerBuilder);
  Value deviceValue = deviceLoadOp.getLoadedGlobalValue();
  deviceLoadOp.setGlobalImmutable(true);
  mapping.map(memoizeOp.getDevice(), deviceValue);
  Value queueAffinityValue = initializerBuilder.create<arith::ConstantIntOp>(
      memoizeOp.getLoc(), memoizeAnalysis.queueAffinity.value_or(-1), 64);
  mapping.map(memoizeOp.getQueueAffinity(), queueAffinityValue);

  // We'll pass the device, affinity, and all dynamic operands to the apply
  // function.
  SmallVector<Value> callOperands;
  callOperands.push_back(deviceValue);
  callOperands.push_back(queueAffinityValue);

  // Clone global loads into the initializer.
  // Multiple initializers may have different sets of globals based on which
  // device they are for. We pass in these globals to the apply function.
  for (auto globalValue : memoizeAnalysis.globalValues) {
    auto *loadOp = globalValue.getDefiningOp();
    initializerBuilder.clone(*loadOp, mapping);
    callOperands.push_back(mapping.lookup(globalValue));
  }

  // Dynamic values aren't supported here as there's no way to get them.
  assert(memoizeAnalysis.dynamicValues.empty() &&
         "dynamic values not allowed in memoized initialization functions");

  // Call the apply function to produce the memoized results.
  auto callOp = initializerBuilder.create<IREE::Util::CallOp>(
      memoizeOp.getLoc(), applyFuncOp, callOperands);

  // Store the results to the globals.
  for (auto [result, resultGlobalOp] :
       llvm::zip_equal(callOp.getResults(), resultGlobalOps)) {
    resultGlobalOp.createStoreOp(memoizeOp.getLoc(), result,
                                 initializerBuilder);
  }

  initializerBuilder.create<IREE::Util::ReturnOp>(memoizeOp.getLoc());

  return resultGlobalOps;
}

// A map of !hal.device globals to one memoized global per returned value from
// the memoize op.
using DeviceResultMap =
    llvm::MapVector<IREE::Util::GlobalOpInterface,
                    SmallVector<IREE::Util::GlobalOpInterface>>;

// Recursively produces an if-else tree matching |device| against
// |deviceGlobalOp|. Calls |perDevice| in each then statement and either
// recurses to try the |remainingDeviceGlobalOps| or calls |fallback| to
// populate the else statement.
//
// Produces something roughly like:
//   %try_device = util.global.load @deviceGlobalOp
//   %is_device = util.cmp.eq %device, %try_device
//   %results... = scf.if %is_device {
//     %if_results... = <<perDevice()>>
//     scf.yield %if_results...
//   } else {
//     %else_results... = <<recurse or fallback()>>
//     scf.yield %else_results...
//   }
using PerDeviceBuilder =
    std::function<scf::ValueVector(IREE::Util::GlobalOpInterface, OpBuilder &)>;
using FallbackBuilder = std::function<scf::ValueVector(OpBuilder &)>;
static scf::ValueVector recursivelyEmitDeviceTree(
    Location loc, TypeRange resultTypes, Value device,
    IREE::Util::GlobalOpInterface deviceGlobalOp,
    ArrayRef<IREE::Util::GlobalOpInterface> remainingDeviceGlobalOps,
    PerDeviceBuilder perDevice, FallbackBuilder fallback, OpBuilder &builder) {

  Value deviceGlobal =
      deviceGlobalOp.createLoadOp(loc, builder).getLoadedGlobalValue();
  Value isDevice =
      builder.create<IREE::Util::CmpEQOp>(loc, device, deviceGlobal);
  auto ifOp = builder.create<scf::IfOp>(
      loc, resultTypes, isDevice, /*addThenBlock=*/true, /*addElseBlock=*/true);

  auto thenBuilder = ifOp.getThenBodyBuilder();
  scf::ValueVector thenResults = perDevice(deviceGlobalOp, thenBuilder);
  thenBuilder.create<scf::YieldOp>(loc, thenResults);

  auto elseBuilder = ifOp.getElseBodyBuilder();
  scf::ValueVector elseResults;
  if (remainingDeviceGlobalOps.empty()) {
    elseResults = fallback(elseBuilder);
  } else {
    elseResults = recursivelyEmitDeviceTree(
        loc, resultTypes, device, remainingDeviceGlobalOps.front(),
        remainingDeviceGlobalOps.drop_front(1), perDevice, fallback,
        elseBuilder);
  }
  elseBuilder.create<scf::YieldOp>(loc, elseResults);

  return llvm::to_vector_of<Value>(ifOp.getResults());
}
static scf::ValueVector recursivelyEmitDeviceTree(
    Location loc, TypeRange resultTypes, Value device,
    ArrayRef<IREE::Util::GlobalOpInterface> deviceGlobalOps,
    PerDeviceBuilder perDevice, FallbackBuilder fallback, OpBuilder &builder) {
  return recursivelyEmitDeviceTree(
      loc, resultTypes, device, deviceGlobalOps.front(),
      deviceGlobalOps.drop_front(1), perDevice, fallback, builder);
}

// Creates a nasty if-else tree comparing the provided |device| to those in
// the |deviceResultMap| and returns the appropriate result global values.
static scf::ValueVector
createDeviceResultSelectTree(Location loc, TypeRange resultTypes, Value device,
                             DeviceResultMap &deviceResultMap,
                             OpBuilder &parentBuilder) {
  auto deviceGlobalOps = llvm::map_to_vector(
      deviceResultMap,
      [](DeviceResultMap::value_type value) { return value.first; });
  return recursivelyEmitDeviceTree(
      loc, resultTypes, device, deviceGlobalOps,
      [&](IREE::Util::GlobalOpInterface deviceGlobalOp, OpBuilder &builder) {
        scf::ValueVector loadedValues;
        for (auto resultGlobalOp : deviceResultMap[deviceGlobalOp]) {
          loadedValues.push_back(
              resultGlobalOp.createLoadOp(resultGlobalOp.getLoc(), builder)
                  .getLoadedGlobalValue());
        }
        return loadedValues;
      },
      [&](OpBuilder &builder) {
        // TODO(benvanik): util.fail here? should not be reachable.
        scf::ValueVector fallbackValues;
        for (auto resultType : resultTypes) {
          Value fallbackValue;
          if (resultType.isIntOrIndexOrFloat()) {
            fallbackValue = builder.create<arith::ConstantOp>(
                loc, resultType, builder.getZeroAttr(resultType));
          } else {
            fallbackValue = builder.create<IREE::Util::NullOp>(loc, resultType);
          }
          fallbackValues.push_back(fallbackValue);
        }
        return fallbackValues;
      },
      parentBuilder);
}

// Creates a function that given a device, queue affinity, and dynamic values
// will return the appropriate result globals.
static IREE::Util::FuncOp createLookupFunc(IREE::HAL::DeviceMemoizeOp memoizeOp,
                                           MemoizeAnalysis &memoizeAnalysis,
                                           IREE::Util::FuncOp applyFuncOp,
                                           DeviceResultMap &deviceResultMap,
                                           SymbolTable &moduleSymbolTable,
                                           OpBuilder &moduleBuilder) {
  auto name = getMemoizeNamePrefix(memoizeOp) + "_lookup";

  // Create the function in the module with device, affinity, and dynamic
  // args.
  SmallVector<Type> argTypes;
  argTypes.push_back(memoizeOp.getDevice().getType());
  argTypes.push_back(memoizeOp.getQueueAffinity().getType());
  llvm::append_range(
      argTypes, llvm::map_range(memoizeAnalysis.dynamicValues,
                                [](Value value) { return value.getType(); }));
  auto funcType =
      moduleBuilder.getFunctionType(argTypes, memoizeOp.getResultTypes());
  auto funcOp = moduleBuilder.create<IREE::Util::FuncOp>(memoizeOp.getLoc(),
                                                         name, funcType);
  moduleSymbolTable.insert(funcOp);
  funcOp.setVisibility(SymbolTable::Visibility::Private);
  auto funcBuilder = OpBuilder::atBlockBegin(funcOp.addEntryBlock());

  // Remap any captured operands that have corresponding function arguments.
  auto capturedArgs = funcOp.getArguments();
  IRMapping mapping;
  mapping.map(memoizeOp.getDevice(), capturedArgs[0]);
  capturedArgs[0].setLoc(memoizeOp.getDevice().getLoc());
  mapping.map(memoizeOp.getQueueAffinity(), capturedArgs[1]);
  capturedArgs[1].setLoc(memoizeOp.getQueueAffinity().getLoc());
  capturedArgs = capturedArgs.drop_front(2);
  for (auto [callerValue, calleeValue] :
       llvm::zip_equal(memoizeAnalysis.dynamicValues,
                       funcOp.getArguments().take_front(
                           memoizeAnalysis.dynamicValues.size()))) {
    mapping.map(callerValue, calleeValue);
    calleeValue.setLoc(callerValue.getLoc());
  }
  capturedArgs = capturedArgs.drop_front(memoizeAnalysis.dynamicValues.size());

  // We don't currently handle dynamic values; if we did we'd compare each
  // dynamic value to a cached global copy from the last run. If all matched
  // we could return the existing memoized results and otherwise we'd run the
  // apply function to create new ones. To make this safer we'd need weak
  // references unless all dynamic values were primitives (shape
  // dimensions/etc) and would likely want an LRU. For now we filter out these
  // cases earlier on so that we can assert here.
  assert(memoizeAnalysis.dynamicValues.empty() &&
         "memoization of ops with dynamic captured values not yet implemented");

  // Create a tree selecting the appropriate result set based on the device
  // provided.
  auto selectedResults = createDeviceResultSelectTree(
      memoizeOp.getLoc(), funcType.getResults(),
      mapping.lookup(memoizeOp.getDevice()), deviceResultMap, funcBuilder);

  funcBuilder.create<IREE::Util::ReturnOp>(memoizeOp.getLoc(), selectedResults);

  return funcOp;
}

// Replaces a |memoizeOp| with a call to the given |lookupFuncOp|.
static void replaceMemoizeOpWithLookup(IREE::HAL::DeviceMemoizeOp memoizeOp,
                                       MemoizeAnalysis &memoizeAnalysis,
                                       IREE::Util::FuncOp lookupFuncOp) {
  // Pass the device, affinity, and all dynamic operands.
  SmallVector<Value> callOperands;
  callOperands.push_back(memoizeOp.getDevice());
  callOperands.push_back(memoizeOp.getQueueAffinity());
  llvm::append_range(callOperands, memoizeAnalysis.dynamicValues);

  // Call the function.
  OpBuilder callerBuilder(memoizeOp);
  auto callOp = callerBuilder.create<IREE::Util::CallOp>(
      memoizeOp.getLoc(), lookupFuncOp, callOperands);

  // Replace memoize op with the results of the function call.
  memoizeOp.replaceAllUsesWith(callOp.getResults());
  memoizeOp.erase();
}

// Creates globals to store memoized per-device results, an initializer to
// perform the memoization by calling an outlined apply function, and replaces
// the |memoizeOp| with a lookup function that can be used to get the
// appropriate per-device results. If the op cannot be memoized then it is
// replaced with an inline call to the outlined apply function.
static void memoizeRegionOp(IREE::HAL::DeviceMemoizeOp memoizeOp,
                            DeviceAnalysis &deviceAnalysis,
                            SymbolTable &moduleSymbolTable) {
  // Outline the memoize region to a function.
  auto memoizeAnalysis = computeMemoizeAnalysis(memoizeOp);
  auto parentFuncOp = memoizeOp->getParentOfType<FunctionOpInterface>();
  OpBuilder moduleBuilder(parentFuncOp);
  auto applyFuncOp = outlineMemoizeRegionBody(memoizeOp, memoizeAnalysis,
                                              moduleSymbolTable, moduleBuilder);

  // If we can't memoize the resources at initialization time then we need
  // to do it on-demand.
  if (!memoizeAnalysis.canRunAtInitializationTime()) {
    memoizeOp.emitWarning(
        "memoization failed: dynamic values captured at the call site");
    replaceMemoizeOpWithApply(memoizeOp, memoizeAnalysis, applyFuncOp);
    return;
  }

  // To memoize we must be able to figure out which devices the op is being
  // memoized for.
  auto deviceGlobals =
      deviceAnalysis.lookupDeviceGlobals(memoizeOp.getDevice());
  if (!deviceGlobals) {
    memoizeOp.emitWarning("memoization failed: unable to analyze devices "
                          "that may be used with memoized region");
    replaceMemoizeOpWithApply(memoizeOp, memoizeAnalysis, applyFuncOp);
    return;
  }

  // Create globals storing the memoized results for each device and an
  // initializer per device that runs the apply function to produce their
  // values.
  DeviceResultMap deviceResultMap;
  for (auto deviceGlobal : deviceGlobals.value()) {
    deviceResultMap[deviceGlobal] = createMemoizedDeviceGlobals(
        memoizeOp, memoizeAnalysis, applyFuncOp, deviceGlobal,
        moduleSymbolTable, moduleBuilder);
  }

  // Create a lookup function that returns the globals for a given device.
  auto lookupFuncOp =
      createLookupFunc(memoizeOp, memoizeAnalysis, applyFuncOp, deviceResultMap,
                       moduleSymbolTable, moduleBuilder);

  // Replace the memoize op with a call to the lookup function.
  replaceMemoizeOpWithLookup(memoizeOp, memoizeAnalysis, lookupFuncOp);
}

struct OutlineMemoizeRegionsPass
    : public IREE::HAL::impl::OutlineMemoizeRegionsPassBase<
          OutlineMemoizeRegionsPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Analyze the module to determine which devices are used where.
    DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run())) {
      return signalPassFailure();
    }

    // Gather all memoize ops in the module. We are changing the module during
    // processing and need to complete the walk before modification.
    SmallVector<IREE::HAL::DeviceMemoizeOp> memoizeOps;
    moduleOp.walk([&](IREE::HAL::DeviceMemoizeOp memoizeOp) {
      memoizeOps.push_back(memoizeOp);
    });

    // Try to outline all memoize ops. Some may fail analysis and be inlined.
    auto &moduleSymbolTable =
        deviceAnalysis.getExplorer().getSymbolTables().getSymbolTable(moduleOp);
    for (auto memoizeOp : memoizeOps) {
      memoizeRegionOp(memoizeOp, deviceAnalysis, moduleSymbolTable);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
