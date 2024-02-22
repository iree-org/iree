// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Utils/IndexSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

// NOTE: redundant bindings will result in unique buffer locations during the
// benchmark and will impact caching behavior.

#define DEBUG_TYPE "iree-dump-executable-benchmarks"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_DUMPEXECUTABLEBENCHMARKSPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

// We could use the resource constraints in the module when we have them.
static const int64_t kBufferAlignment = 256;

using Vec3 = std::tuple<unsigned, unsigned, unsigned>;

struct Binding {
  unsigned set = 0;
  unsigned binding = 0;
  int64_t size = 0;
};

// Combined data for all dispatches of a particular static workload size.
struct DispatchParams {
  // All locations that dispatch with these parameters.
  SmallVector<Location> locs;
  // All affinities that dispatch with these parameters.
  // Empty if no affinities were specified.
  SetVector<IREE::Stream::AffinityAttr> affinities;
  // Workload used as input to the workgroup count calculation function.
  SmallVector<unsigned> workload;
  // Analyzed minimum binding sizes.
  SmallVector<Binding> bindings;
  // Push constant operands that are known constant. May be null if dynamic.
  SmallVector<TypedAttr> uniformOperands;
};

using DispatchParamsMap =
    llvm::DenseMap<SymbolRefAttr, std::vector<DispatchParams>>;

// Walk |moduleOp| and gather all of the dispatches to each executable.
// Dispatch parameters are deduplicated by workload so that there's only ever
// one entry for all dispatches with a given workgroup count.
// Dispatches will be ignored if they have a dynamic workload or any dynamically
// sized resources.
static DispatchParamsMap gatherDispatchParams(mlir::ModuleOp moduleOp,
                                              SymbolTable &symbolTable) {
  DispatchParamsMap map;

  for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
    funcOp.walk([&](IREE::Stream::CmdDispatchOp dispatchOp) {
      auto affinityAttr = IREE::Stream::AffinityAttr::lookup(dispatchOp);

      auto workloadValues = dispatchOp.getWorkload();
      SmallVector<unsigned> workload;
      workload.reserve(workloadValues.size());
      for (auto workloadValue : workloadValues) {
        APInt workloadConstValue;
        if (!matchPattern(workloadValue, m_ConstantInt(&workloadConstValue))) {
          LLVM_DEBUG({
            auto firstEntryPoint = *dispatchOp.getEntryPointRefs().begin();
            llvm::dbgs() << "Skipping dispatch of entry point `"
                         << firstEntryPoint << "` (non-constant workload)\n";
          });
          return;
        }
        workload.push_back(workloadConstValue.getSExtValue());
      }

      SmallVector<TypedAttr> uniformOperands;
      for (auto operand : dispatchOp.getUniformOperands()) {
        TypedAttr uniformOperand;
        if (!matchPattern(operand, m_Constant(&uniformOperand))) {
          // TODO(benvanik): extract information from the executable annotations
          // or allow the dynamic value to be passed in as an additional arg.
          LLVM_DEBUG({
            auto firstEntryPoint = *dispatchOp.getEntryPointRefs().begin();
            llvm::dbgs() << "Skipping dispatch of entry point `"
                         << firstEntryPoint
                         << "` (non-constant uniform operand)\n";
          });
          return;
        }
        uniformOperands.push_back(uniformOperand);
      }

      // Work around needing a mutable key for the set; C++ was a mistake.
      dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPointAttr) {
        auto exportOp =
            symbolTable.lookupNearestSymbolFrom<IREE::HAL::ExecutableExportOp>(
                dispatchOp, entryPointAttr);
        auto bindingAttrs = IREE::HAL::getInterfaceBindingAttrs(
            exportOp, dispatchOp.getResources().size());

        SmallVector<Binding> bindings;
        for (auto [bindingAttr, resourceLength] :
             llvm::zip_equal(bindingAttrs, dispatchOp.getResourceLengths())) {
          APInt resourceLengthInt;
          if (!matchPattern(resourceLength,
                            m_ConstantInt(&resourceLengthInt))) {
            LLVM_DEBUG(llvm::dbgs() << "Skipping dispatch of entry point `"
                                    << entryPointAttr
                                    << "` (non-constant resource length)\n";);
            return;
          }
          bindings.push_back({(unsigned)bindingAttr.getSet(),
                              (unsigned)bindingAttr.getBinding(),
                              resourceLengthInt.getSExtValue()});
        }

        auto &dispatchParamsSet = map[entryPointAttr];
        DispatchParams *dispatchParams = nullptr;
        for (auto &it : dispatchParamsSet) {
          if (it.workload == workload) {
            dispatchParams = &it;
            break;
          }
        }
        if (!dispatchParams) {
          dispatchParamsSet.push_back({});
          dispatchParams = &dispatchParamsSet.back();
        }
        dispatchParams->locs.push_back(dispatchOp.getLoc());
        dispatchParams->affinities.insert(affinityAttr);
        dispatchParams->workload = workload;
        dispatchParams->bindings = std::move(bindings);
        dispatchParams->uniformOperands = std::move(uniformOperands);
      });
    });
  }

  return map;
}

// Appends a global hal.buffer initialized to the size required for all
// of the bindings in |dispatchParams| (plus alignment).
static IREE::Util::GlobalOp
appendGlobalBuffer(Location loc, StringRef baseName,
                   const DispatchParams &dispatchParams,
                   OpBuilder &moduleBuilder) {
  // Create a global to hold the HAL buffer.
  auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
      loc, (baseName + "_buffer").str(),
      /*isMutable=*/true,
      IREE::HAL::BufferType::get(moduleBuilder.getContext()));
  globalOp.setPrivate();

  // Compute the total size of the buffer based on all binding sizes when
  // aligned.
  int64_t totalLength = 0;
  for (auto binding : dispatchParams.bindings) {
    totalLength =
        IREE::Util::align(totalLength + binding.size, kBufferAlignment);
  }

  // Build an initializer to allocate the buffer.
  auto initOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);
  auto initBuilder = OpBuilder::atBlockBegin(initOp.addEntryBlock());
  IndexSet indexSet(loc, initBuilder);

  // TODO(multi-device): support multiple devices in benchmark generation.
  Value device = IREE::HAL::DeviceType::resolveAny(loc, initBuilder);
  auto allocator =
      initBuilder.create<IREE::HAL::DeviceAllocatorOp>(loc, device).getResult();

  auto queueAffinity = initBuilder.create<arith::ConstantIntOp>(loc, -1, 64);
  auto memoryTypes = IREE::HAL::MemoryTypeBitfield::DeviceLocal;
  auto bufferUsage = IREE::HAL::BufferUsageBitfield::Transfer |
                     IREE::HAL::BufferUsageBitfield::DispatchStorage;
  auto allocateOp = initBuilder.create<IREE::HAL::AllocatorAllocateOp>(
      loc, globalOp.getType(), allocator, queueAffinity, memoryTypes,
      bufferUsage, indexSet.get(totalLength));

  globalOp.createStoreOp(loc, allocateOp.getResult(), initBuilder);
  initBuilder.create<IREE::Util::ReturnOp>(loc);

  return globalOp;
}

// Appends a function calling the given |exportOp| with |dispatchParams|.
// This will add a global value for the resources required.
//
// Expects the runner to pass an i32 value indicating the number of dispatches
// to be made in one submission.
static void appendDispatchBenchmark(IREE::Stream::AffinityAttr affinityAttr,
                                    IREE::HAL::ExecutableOp executableOp,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    IREE::HAL::ExecutableExportOp exportOp,
                                    DispatchParams dispatchParams,
                                    OpBuilder &moduleBuilder) {
  auto loc = FusedLoc::get(executableOp.getContext(), dispatchParams.locs);

  std::string baseName = (executableOp.getName() + "_" + variantOp.getName() +
                          "_" + exportOp.getName())
                             .str();
  if (!dispatchParams.workload.empty()) {
    baseName += "_" + std::to_string(dispatchParams.workload[0]);
    for (size_t i = 1; i < dispatchParams.workload.size(); ++i) {
      baseName += "x" + std::to_string(dispatchParams.workload[i]);
    }
  }

  // Add a global variable holding an initialized buffer for the dispatch IO.
  auto bufferGlobalOp =
      appendGlobalBuffer(loc, baseName, dispatchParams, moduleBuilder);

  // Create an exported benchmark function that runs the dispatches.
  auto funcType =
      moduleBuilder.getFunctionType({moduleBuilder.getI32Type()}, {});
  auto funcOp =
      moduleBuilder.create<IREE::Util::FuncOp>(loc, baseName, funcType);
  funcOp.setVisibility(SymbolTable::Visibility::Public);

  // Mark the function as being a dispatch benchmark.
  // This tells iree-benchmark-module to pass in the arguments we need.
  funcOp->setAttr("iree.abi.stub", moduleBuilder.getUnitAttr());
  funcOp->setAttr(
      "iree.reflection",
      moduleBuilder.getDictionaryAttr({
          moduleBuilder.getNamedAttr("iree.benchmark",
                                     moduleBuilder.getStringAttr("dispatch")),
      }));

  // Build the function that runs the dispatches.
  auto *entryBlock = funcOp.addEntryBlock();
  OpBuilder funcBuilder = OpBuilder::atBlockBegin(entryBlock);
  IndexSet indexSet(loc, funcBuilder);
  auto batchSizeArg = funcBuilder.create<arith::IndexCastOp>(
      loc, funcBuilder.getIndexType(), entryBlock->getArgument(0));

  // TODO(multi-device): support multiple devices in benchmark generation.
  // For now we should just use the affinityAttr to resolve the device.
  Value device = IREE::HAL::DeviceType::resolveAny(loc, funcBuilder);

  // Create and begin command buffer.
  // TODO(benvanik): reuse the command buffer (initialize once and store).
  auto commandBufferModes =
      IREE::HAL::CommandBufferModeBitfield::OneShot |
      IREE::HAL::CommandBufferModeBitfield::AllowInlineExecution;
  auto commandBuffer =
      funcBuilder
          .create<IREE::HAL::CommandBufferCreateOp>(
              loc, funcBuilder.getType<IREE::HAL::CommandBufferType>(), device,
              commandBufferModes, IREE::HAL::CommandCategoryBitfield::Dispatch,
              /*binding_capacity=*/Value{})
          .getResult();

  // Get the layout required to set up the dispatches.
  auto layoutAttr = exportOp.getLayoutAttr();
  auto pipelineLayout =
      funcBuilder
          .create<IREE::HAL::PipelineLayoutLookupOp>(
              loc, IREE::HAL::PipelineLayoutType::get(loc.getContext()), device,
              layoutAttr)
          .getResult();

  // Push constant values.
  if (int64_t pushConstantCount = layoutAttr.getPushConstants()) {
    int pushConstantBase = 0; // always 0 today
    SmallVector<Value> pushConstants;
    pushConstants.reserve(pushConstantCount);
    for (int64_t i = 0; i < pushConstantCount; ++i) {
      pushConstants.push_back(funcBuilder.create<arith::ConstantOp>(
          loc, dispatchParams.uniformOperands[i]));
    }
    funcBuilder.create<IREE::HAL::CommandBufferPushConstantsOp>(
        loc, commandBuffer, pipelineLayout,
        funcBuilder.getIndexAttr(pushConstantBase), pushConstants);
  }

  // Push descriptor sets.
  Value buffer =
      bufferGlobalOp.createLoadOp(loc, funcBuilder).getLoadedGlobalValue();
  int64_t currentSet = -1;
  SmallVector<IREE::HAL::DescriptorSetBindingValue> bindingValues;
  auto flushSet = [&]() {
    funcBuilder.create<IREE::HAL::CommandBufferPushDescriptorSetOp>(
        loc, commandBuffer, pipelineLayout, currentSet, bindingValues);
    bindingValues.clear();
  };
  int64_t bufferOffset = 0;
  for (auto binding : dispatchParams.bindings) {
    if (currentSet != -1 && currentSet != binding.set)
      flushSet();
    currentSet = binding.set;
    IREE::HAL::DescriptorSetBindingValue bindingValue;
    bindingValue.ordinal =
        funcBuilder.create<arith::ConstantIndexOp>(loc, binding.binding);
    bindingValue.buffer = buffer;
    bindingValue.byteOffset = indexSet.get(bufferOffset);
    bindingValue.byteLength = indexSet.get(binding.size);
    bindingValues.push_back(bindingValue);
    bufferOffset =
        IREE::Util::align(bufferOffset + binding.size, kBufferAlignment);
  }
  if (currentSet != -1)
    flushSet();

  // @executable::@variant::@export
  auto exportRefAttr =
      SymbolRefAttr::get(executableOp.getNameAttr(),
                         {
                             SymbolRefAttr::get(variantOp.getNameAttr()),
                             SymbolRefAttr::get(exportOp.getNameAttr()),
                         });

  // Compute the workgroup parameters.
  auto workload = llvm::map_to_vector(
      dispatchParams.workload, [&](unsigned dim) { return indexSet.get(dim); });
  auto workgroupCountOp =
      funcBuilder.create<IREE::HAL::ExecutableCalculateWorkgroupsOp>(
          loc, funcBuilder.getIndexType(), funcBuilder.getIndexType(),
          funcBuilder.getIndexType(), device, exportRefAttr, workload);

  // Get the executable/entry point ordinal used to dispatch.
  Value executable = funcBuilder.create<IREE::HAL::ExecutableLookupOp>(
      loc, funcBuilder.getType<IREE::HAL::ExecutableType>(), device,
      exportRefAttr.getRootReference().getValue());
  Value ordinal = funcBuilder.create<IREE::HAL::ExecutableExportOrdinalOp>(
      loc, funcBuilder.getIndexType(), exportRefAttr);

  // Loop around dispatches based on batch size.
  // Note that we insert a barrier between each dispatch - we could make this
  // optional so that concurrent utilization is measured.
  funcBuilder.create<scf::ForOp>(
      loc, indexSet.get(0), batchSizeArg, indexSet.get(1), ValueRange{},
      [&](OpBuilder &forBuilder, Location loc, Value iv, ValueRange iters) {
        // Dispatch.
        forBuilder.create<IREE::HAL::CommandBufferDispatchOp>(
            loc, commandBuffer, executable, ordinal,
            workgroupCountOp.getWorkgroupX(), workgroupCountOp.getWorkgroupY(),
            workgroupCountOp.getWorkgroupZ());

        // Barrier following the dispatch to block the next dispatch.
        auto sourceStage = IREE::HAL::ExecutionStageBitfield::CommandRetire |
                           IREE::HAL::ExecutionStageBitfield::Dispatch;
        auto targetStage = IREE::HAL::ExecutionStageBitfield::CommandIssue |
                           IREE::HAL::ExecutionStageBitfield::Dispatch;
        auto barrierFlags = IREE::HAL::ExecutionBarrierFlagBitfield::None;
        forBuilder.create<IREE::HAL::CommandBufferExecutionBarrierOp>(
            loc, commandBuffer, sourceStage, targetStage, barrierFlags);

        forBuilder.create<scf::YieldOp>(loc);
      });

  funcBuilder.create<IREE::HAL::CommandBufferFinalizeOp>(loc, commandBuffer);

  // We begin executing immediately and then wait on a fence.
  // TODO(benvanik): add fences to ABI so the benchmark tool can pipeline.
  Value waitFence = funcBuilder.create<IREE::Util::NullOp>(
      loc, funcBuilder.getType<IREE::HAL::FenceType>());
  Value signalFence = funcBuilder.create<IREE::HAL::FenceCreateOp>(
      loc, funcBuilder.getType<IREE::HAL::FenceType>(), device,
      IREE::HAL::FenceFlagBitfield::None);

  // Queue execution.
  auto queueAffinity = funcBuilder.create<arith::ConstantIntOp>(loc, -1, 64);
  funcBuilder.create<IREE::HAL::DeviceQueueExecuteOp>(
      loc, device, queueAffinity, waitFence, signalFence,
      ValueRange{commandBuffer});

  // Block until it completes.
  Value timeoutMillis = funcBuilder.create<arith::ConstantIntOp>(loc, -1, 32);
  auto fenceOp = funcBuilder.create<IREE::HAL::FenceAwaitOp>(
      loc, funcBuilder.getI32Type(), timeoutMillis, signalFence);
  funcBuilder.create<IREE::Util::StatusCheckOkOp>(
      loc, fenceOp.getStatus(), "failed to wait on timepoint");

  funcBuilder.create<IREE::Util::ReturnOp>(loc);
}

// Builds a module exporting one function for each dispatch configuration
// targeting |sourceExecutableOp|.
static mlir::OwningOpRef<mlir::ModuleOp>
buildBenchmarkModule(IREE::HAL::ExecutableOp sourceExecutableOp,
                     IREE::HAL::ExecutableVariantOp sourceVariantOp,
                     const DispatchParamsMap &dispatchParamsMap) {
  // Empty module with default name.
  // We could use the original module name here to make tracking nicer.
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
      mlir::ModuleOp::create(sourceExecutableOp.getLoc());
  auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp->getBody());

  // Copy over the device targets from the original module.
  // TODO(benvanik): filter this by the target of the variant.
  moduleOp->getOperation()->setAttr(
      "hal.device.targets",
      sourceExecutableOp->getParentOfType<mlir::ModuleOp>()->getAttr(
          "hal.device.targets"));

  // Clone the executable variant into the new module.
  auto executableOp = moduleBuilder.create<IREE::HAL::ExecutableOp>(
      sourceExecutableOp.getLoc(), sourceExecutableOp.getName());
  executableOp.setVisibility(sourceExecutableOp.getVisibility());
  auto variantOp = cast<IREE::HAL::ExecutableVariantOp>(
      OpBuilder::atBlockBegin(&executableOp.getBlock())
          .clone(*sourceVariantOp.getOperation()));

  // Add functions to test each entry point with its various dispatch
  // parameters.
  bool hasAnyBenchmarks = false;
  for (auto exportOp : variantOp.getExportOps()) {
    auto symbolRefAttr =
        SymbolRefAttr::get(executableOp.getNameAttr(),
                           {
                               FlatSymbolRefAttr::get(variantOp.getNameAttr()),
                               FlatSymbolRefAttr::get(exportOp.getNameAttr()),
                           });
    auto dispatchParamsSet = dispatchParamsMap.find(symbolRefAttr);
    if (dispatchParamsSet != dispatchParamsMap.end()) {
      for (auto &dispatchParams : dispatchParamsSet->second) {
        if (dispatchParams.affinities.empty()) {
          appendDispatchBenchmark({}, executableOp, variantOp, exportOp,
                                  dispatchParams, moduleBuilder);
        } else {
          for (auto affinityAttr : dispatchParams.affinities) {
            appendDispatchBenchmark(affinityAttr, executableOp, variantOp,
                                    exportOp, dispatchParams, moduleBuilder);
          }
        }
        hasAnyBenchmarks = true;
      }
    }
  }

  // Skip the file when we could not generate any benchmarks.
  if (!hasAnyBenchmarks)
    return {};

  // Run CSE and the canonicalizer to pretty up the output.
  PassManager passManager(moduleOp->getContext());
  passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addPass(mlir::createCSEPass());
  if (failed(passManager.run(*moduleOp))) {
    moduleOp->emitError("failed to run canonicalizer; malformed output");
    return {};
  }

  return moduleOp;
}

static void dumpModuleToStream(mlir::ModuleOp moduleOp, StringRef fileName,
                               llvm::raw_ostream &os) {
  OpPrintingFlags flags;
  flags.useLocalScope(); // could use global scope, but IR gets messy fast
  moduleOp.print(os, flags);
  os << "\n"; // newline at end of file
}

//===----------------------------------------------------------------------===//
// --iree-hal-dump-executable-benchmarks
//===----------------------------------------------------------------------===//

struct DumpExecutableBenchmarksPass
    : public IREE::HAL::impl::DumpExecutableBenchmarksPassBase<
          DumpExecutableBenchmarksPass> {
  using IREE::HAL::impl::DumpExecutableBenchmarksPassBase<
      DumpExecutableBenchmarksPass>::DumpExecutableBenchmarksPassBase;
  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto moduleName = moduleOp.getName().value_or("module");
    SymbolTable symbolTable(moduleOp);

    // Analyze the module to find dispatch parameters.
    // This is a full walk of all stream.cmd.dispatch ops and will handle
    // filtering out dispatches that have dynamic parameters we don't
    // currently support.
    auto dispatchParamsMap = gatherDispatchParams(moduleOp, symbolTable);
    if (dispatchParamsMap.empty()) {
      mlir::emitRemark(moduleOp.getLoc())
          << "Executable benchmarks were requested but none were generated. "
             "Run with --debug-only=iree-dump-executable-benchmarks for more "
             "details.\n";
      return;
    }

    // Help people out and mkdir if needed.
    if (!path.empty() && path != "-") {
      llvm::sys::fs::create_directories(path);
    }

    // Produce one file per executable containing all exported entry points.
    for (auto executableOp : moduleOp.getOps<IREE::HAL::ExecutableOp>()) {
      for (auto variantOp :
           executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
        auto benchmarkModuleOp =
            buildBenchmarkModule(executableOp, variantOp, dispatchParamsMap);
        if (!benchmarkModuleOp)
          continue;
        auto fileName = (moduleName + "_" + executableOp.getName() + "_" +
                         variantOp.getName() + "_benchmark.mlir")
                            .str();
        if (path.empty() || path == "-") {
          dumpModuleToStream(*benchmarkModuleOp, fileName, llvm::outs());
        } else {
          auto filePath =
              (path + llvm::sys::path::get_separator() + fileName).str();
          std::string error;
          auto file = mlir::openOutputFile(filePath, &error);
          if (!file) {
            executableOp.emitError()
                << "while dumping to " << path << ": " << error;
            return signalPassFailure();
          }
          dumpModuleToStream(*benchmarkModuleOp, fileName, file->os());
          file->keep();
        }
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
