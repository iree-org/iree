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

#include "iree/compiler/Dialect/HAL/Target/SPIRVCommon/SPIRVTarget.h"

#include "iree/compiler/Conversion/Common/Attributes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Records a full execution barrier that forces visibility of all buffers.
static void recordFullExecutionBarrier(Value commandBuffer, Location loc,
                                       OpBuilder &builder) {
  builder.create<IREE::HAL::CommandBufferExecutionBarrierOp>(
      loc, commandBuffer, IREE::HAL::ExecutionStageBitfield::Dispatch,
      IREE::HAL::ExecutionStageBitfield::Dispatch,
      IREE::HAL::ExecutionBarrierFlagBitfield::None);
}

SPIRVTargetBackend::SPIRVTargetBackend(SPIRVCodegenOptions options)
    : spvCodeGenOptions_(std::move(options)) {}

void SPIRVTargetBackend::declareTargetOpsForEnv(
    IREE::Flow::ExecutableOp sourceOp, IREE::HAL::ExecutableOp executableOp,
    spirv::TargetEnvAttr spvTargetEnv) {
  auto targetBuilder = OpBuilder::atBlockTerminator(&executableOp.getBlock());
  auto targetOp = targetBuilder.create<IREE::HAL::ExecutableTargetOp>(
      sourceOp.getLoc(), name(), filter_pattern());

  auto containerBuilder = OpBuilder::atBlockTerminator(&targetOp.getBlock());
  auto innerModuleOp = containerBuilder.create<ModuleOp>(sourceOp.getLoc());

  // Attach SPIR-V target environment to the target's ModuleOp.
  // If we had multiple target environments we would generate one target op
  // per environment, with each setting its own environment attribute.
  innerModuleOp->setAttr(spirv::getTargetEnvAttrName(), spvTargetEnv);
}

void SPIRVTargetBackend::buildTranslationPassPipeline(
    OpPassManager &passManager) {
  buildSPIRVTransformPassPipeline(passManager, spvCodeGenOptions_);
}

LogicalResult SPIRVTargetBackend::recordDispatch(
    Location loc, DispatchState dispatchState,
    DeviceSwitchRewriter &switchRewriter) {
  // TODO(#4140): remove this legacy path when linalg-on-tensors is used.
  // In the linalg-on-tensors world where we are performing the tiling logic
  // in the flow dialect we don't even really need the ability to override
  // dispatch recording at all - just a way to allow targets to map workgroup
  // counts from the N-dimensional flow workgroup counts to the 3D hal counts.
  if (dispatchState.workgroupCount.size() == 3) {
    return TargetBackend::recordDispatch(loc, dispatchState, switchRewriter);
  }

  // Multiple entry points might be generated for a single dispatch function.
  // Under such circumstances, we will have a special attribute indicating the
  // schedule of the split entry points. Try to see if we can find such
  // schedule attribute first.
  ArrayAttr entryPointScheduleAttr;
  spirv::ModuleOp spvModuleOp;
  IREE::HAL::ExecutableOp executableOp = dispatchState.executableOp;
  for (auto executableTargetOp :
       executableOp.getBlock().getOps<IREE::HAL::ExecutableTargetOp>()) {
    if (matchPattern(executableTargetOp.target_backend_filter(),
                     filter_pattern())) {
      ModuleOp innerModuleOp = executableTargetOp.getInnerModule();
      auto spvModuleOps = innerModuleOp.getOps<spirv::ModuleOp>();
      assert(llvm::hasSingleElement(spvModuleOps));
      spvModuleOp = *spvModuleOps.begin();
      entryPointScheduleAttr = innerModuleOp->getAttrOfType<ArrayAttr>(
          iree_compiler::getEntryPointScheduleAttrName());
      if (!spvModuleOp)
        return executableOp.emitError("unable to find spv.module");

      SmallVector<IREE::HAL::ExecutableEntryPointOp, 2> entryPoints;
      if (!entryPointScheduleAttr) {
        entryPoints = llvm::to_vector<2>(
            executableTargetOp.getOps<IREE::HAL::ExecutableEntryPointOp>());
        if (!llvm::hasSingleElement(entryPoints)) {
          return executableTargetOp.emitError(
                     "expected a single entry point, found ")
                 << entryPoints.size();
        }
      } else {
        SymbolTable symTable(executableTargetOp);
        for (Attribute entryPointAttr : entryPointScheduleAttr) {
          auto entryPointOp =
              symTable.lookup<IREE::HAL::ExecutableEntryPointOp>(
                  entryPointAttr.cast<FlatSymbolRefAttr>().getValue());
          if (!entryPointOp) {
            return executableTargetOp.emitError(
                       "unable to find hal.executable.entry_point operation "
                       "for ")
                   << entryPointAttr.cast<FlatSymbolRefAttr>().getValue();
          }
          entryPoints.push_back(entryPointOp);
        }
      }

      auto *region = switchRewriter.addConditionRegion(
          IREE::HAL::DeviceMatchIDAttr::get(filter_pattern(), loc.getContext()),
          {
              dispatchState.workgroupCount[0],
              dispatchState.commandBuffer,
          });

      auto &entryBlock = region->front();
      ConversionPatternRewriter &rewriter = switchRewriter.getRewriter();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&entryBlock);
      auto workload = entryBlock.getArgument(0);
      auto commandBuffer = entryBlock.getArgument(1);

      // We have multiple entry points to dispatch. Record in the order
      // specified by entry point schedule and insert barrier between sequential
      // ones.
      for (auto entryPoint : llvm::enumerate(entryPoints)) {
        std::array<Value, 3> workgroupCount = calculateDispatchWorkgroupCount(
            loc, executableOp, entryPoint.value(), workload, rewriter);

        if (llvm::any_of(workgroupCount,
                         [](Value v) -> bool { return v == nullptr; })) {
          return entryPoint.value().emitError("unable to find workgroup count");
        }

        // Ordinals are fixed based on the precomputed schedule, so use
        // CommandBufferDispatchOp instead of CommandBufferDispatchSymbolOp.
        auto executable = rewriter
                              .create<IREE::HAL::ExecutableLookupOp>(
                                  loc, dispatchState.device,
                                  dispatchState.dispatchOp.executable())
                              .getResult();
        int32_t entryPointOrdinal = entryPoint.index();
        rewriter.create<IREE::HAL::CommandBufferDispatchOp>(
            loc, commandBuffer, executable,
            rewriter.getIndexAttr(entryPointOrdinal), workgroupCount[0],
            workgroupCount[1], workgroupCount[2]);
        if (entryPoint.index() + 1 != entryPoints.size()) {
          recordFullExecutionBarrier(commandBuffer, loc, rewriter);
        }
      }
      rewriter.create<IREE::HAL::ReturnOp>(loc);
    }
  }
  return success();
}

// Finds the spv.ExecutionMode operation to get the workgroup size from.
// TODO(ravishankarm): This might not be the only way this is specified. You
// could also have a spec constant, but that is not generated in the
// spv.module right now.
// TODO(ravishankarm): change workgroup size calculation to something we can
// query independently so that we don't need to lookup the value here.
std::array<Value, 3> SPIRVTargetBackend::calculateDispatchWorkgroupSize(
    Location loc, IREE::HAL::ExecutableOp executableOp,
    IREE::HAL::ExecutableEntryPointOp entryPointOp, ValueRange workload,
    OpBuilder &builder) {
  // TODO(ravishankarm): possibly emit different recordDispatch logic if the
  // workgroup sizes differ among targets.
  spirv::ModuleOp spvModuleOp;
  for (auto executableTargetOp :
       executableOp.getBlock().getOps<IREE::HAL::ExecutableTargetOp>()) {
    if (matchPattern(executableTargetOp.target_backend_filter(),
                     filter_pattern())) {
      ModuleOp innerModuleOp = executableTargetOp.getInnerModule();
      assert(!innerModuleOp->getAttr(
          iree_compiler::getEntryPointScheduleAttrName()));
      auto spvModuleOps = innerModuleOp.getOps<spirv::ModuleOp>();
      assert(llvm::hasSingleElement(spvModuleOps));
      spvModuleOp = *spvModuleOps.begin();
      break;
    }
  }
  return calculateDispatchWorkgroupSize(
      loc, spvModuleOp, entryPointOp.sym_name(), workload, builder);
}

std::array<Value, 3> SPIRVTargetBackend::calculateDispatchWorkgroupSize(
    Location loc, spirv::ModuleOp spvModuleOp, StringRef entryPointName,
    ValueRange workload, OpBuilder &builder) {
  std::array<Value, 3> workgroupSize;
  for (auto executionModeOp :
       spvModuleOp.getBlock().getOps<spirv::ExecutionModeOp>()) {
    if (executionModeOp.fn() == entryPointName &&
        executionModeOp.execution_mode() == spirv::ExecutionMode::LocalSize) {
      for (int i = 0; i < executionModeOp.values().size(); ++i) {
        workgroupSize[i] =
            builder.create<ConstantIndexOp>(loc, executionModeOp.values()[i]
                                                     .cast<IntegerAttr>()
                                                     .getValue()
                                                     .getZExtValue());
      }
      break;
    }
  }

  // Pad out the workgroup size with 1's (if the original rank was < 3).
  for (int i = 0; i < workgroupSize.size(); ++i) {
    if (!workgroupSize[i]) {
      workgroupSize[i] = builder.create<ConstantIndexOp>(loc, 1);
    }
  }

  return workgroupSize;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
