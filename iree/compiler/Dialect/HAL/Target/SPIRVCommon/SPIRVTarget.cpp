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

#include "iree/compiler/Conversion/LinalgToSPIRV/Attributes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Records a full execution barrier that forces visibility of all buffers.
static void recordFullExecutionBarrier(Value commandBuffer, Location loc,
                                       OpBuilder &builder) {
  Value memoryBarrier = builder.create<IREE::HAL::MakeMemoryBarrierOp>(
      loc, IREE::HAL::AccessScopeBitfield::DispatchWrite,
      IREE::HAL::AccessScopeBitfield::DispatchRead);
  builder.create<IREE::HAL::CommandBufferExecutionBarrierOp>(
      loc, commandBuffer, IREE::HAL::ExecutionStageBitfield::Dispatch,
      IREE::HAL::ExecutionStageBitfield::Dispatch,
      ArrayRef<Value>{memoryBarrier}, ArrayRef<Value>{});
}

/// Generates IR to compute the ceil(`numerator`, `denominator`).
static Value computeCeilDiv(Location loc, Value one, Value numerator,
                            Value denominator, OpBuilder &builder) {
  Value dm1 = builder.create<SubIOp>(loc, denominator, one);
  return builder.create<SignedDivIOp>(
      loc, builder.create<AddIOp>(loc, numerator, dm1), denominator);
}

/// Calculates the number of workgroups to use based on the shape of the result
/// of the dispatch region.  If the `resultShape` is {s0, s1, s2, s3, ....} and
/// `workgroupSize` is {wx, wy, wz}, the number of workgroups is {ceil(s0/wz),
/// ceil(s1/wy), ceil(s2/wx)}
static std::array<Value, 3> calculateDispatchWorkgroupCountFromResultShape(
    Location loc, ArrayRef<Value> resultShape,
    const std::array<Value, 3> &workgroupSize, OpBuilder &builder) {
  if (resultShape.size() > 3) resultShape = resultShape.take_front(3);
  SmallVector<Value, 4> reverseResultSize(reverse(resultShape));
  Value one = builder.create<ConstantOp>(loc, builder.getIndexAttr(1));
  reverseResultSize.resize(3, one);
  return {
      computeCeilDiv(loc, one, reverseResultSize[0], workgroupSize[0], builder),
      computeCeilDiv(loc, one, reverseResultSize[1], workgroupSize[1], builder),
      computeCeilDiv(loc, one, reverseResultSize[2], workgroupSize[2],
                     builder)};
}

/// Calculates the number of workgroups to use based on the linearized shape of
/// the result of the dispatch region. The `workgroupSize` is assumed to be of
/// the form {wx, 1, 1}.  If the `resultShape` is {s0, s1, s2, ... sn}, then the
/// number of workgroups is {ceil(s0*s1*s2*...*sn, wx)}
static std::array<Value, 3>
calculateDispatchWorkgroupCountFromLinearizedResultShape(
    Location loc, ArrayRef<Value> resultShape,
    const std::array<Value, 3> &workgroupSize, OpBuilder &builder) {
  if (!mlir::matchPattern(workgroupSize[1], m_One()) ||
      !mlir::matchPattern(workgroupSize[2], m_One())) {
    emitError(loc,
              "invalid workgroup size when computing workgroup count "
              "based linearized result shape");
    return {nullptr, nullptr, nullptr};
  }
  Value one = builder.create<ConstantOp>(loc, builder.getIndexAttr(1));
  Value linearizedSize = one;
  for (Value dim : resultShape)
    linearizedSize = builder.create<MulIOp>(loc, linearizedSize, dim);
  return {computeCeilDiv(loc, one, linearizedSize, workgroupSize[0], builder),
          one, one};
}

/// Calculates the number of workgroups to use for a dispatch region based on
/// the value of `workgroupCountMethodAttr`. This is obtained from an attribute
/// specified on the entry point functions that is added while lowering to
/// SPIR-V.
// TODO(ravishankarm): This method of using enums to specify methodology to
// compute workgroup count is very hard to maintain. The best approach would be
// that the lowering generates a function that is "inlined" here. Need to figure
// out the signature of that function so that it covers all use cases.
static std::array<Value, 3> calculateSPIRVDispatchWorkgroupCount(
    Location loc, ArrayRef<Value> resultShape,
    IntegerAttr workgroupCountMethodAttr,
    const std::array<Value, 3> &workgroupSize, OpBuilder &builder) {
  WorkgroupCountMethodology workgroupCountMethod =
      static_cast<WorkgroupCountMethodology>(
          workgroupCountMethodAttr.getValue().getZExtValue());
  switch (workgroupCountMethod) {
    case WorkgroupCountMethodology::Default:
      return {nullptr, nullptr, nullptr};
    case WorkgroupCountMethodology::LinearizeResultShape:
      return calculateDispatchWorkgroupCountFromLinearizedResultShape(
          loc, resultShape, workgroupSize, builder);
    case WorkgroupCountMethodology::ResultShape:
      return calculateDispatchWorkgroupCountFromResultShape(
          loc, resultShape, workgroupSize, builder);
  }
  return {nullptr, nullptr, nullptr};
}

/// Gets the shape of the result from the dispatchState.
static Optional<SmallVector<Value, 4>> getFirstResultShape(
    Location loc, TargetBackend::DispatchState dispatchState,
    OpBuilder &builder) {
  if (dispatchState.results.empty()) return llvm::None;
  Optional<TensorRewriteAdaptor> result = dispatchState.results[0];
  SmallVector<Value, 4> resultShape;
  // If the output is not a shaped type, assume it is a scalar, and return {1}.
  if (!result) {
    resultShape.push_back(
        builder.create<ConstantOp>(loc, builder.getIndexAttr(1)));
    return resultShape;
  }

  // TODO(ravishankarm): Using the result shape to get workgroup count, which
  // involes using `getShapeDims,` results in the shape values being captured
  // from outside of the switch statement in dynamic shape cases. This results
  // in an error since switch statements cannot capture. For now, use the
  // default path when the shape is dynamic.
  if (!result->getTensorType().hasStaticShape()) return llvm::None;

  return result->getShapeDims(builder);
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
  innerModuleOp.setAttr(spirv::getTargetEnvAttrName(), spvTargetEnv);
}

void SPIRVTargetBackend::buildTranslationPassPipeline(
    IREE::HAL::ExecutableTargetOp targetOp, OpPassManager &passManager) {
  buildSPIRVTransformPassPipeline(passManager, spvCodeGenOptions_);
}

LogicalResult SPIRVTargetBackend::recordDispatch(
    Location loc, DispatchState dispatchState,
    DeviceSwitchBuilder &switchBuilder) {
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
      entryPointScheduleAttr = innerModuleOp.getAttrOfType<ArrayAttr>(
          iree_compiler::getEntryPointScheduleAttrName());
      break;
    }
  }
  if (!spvModuleOp) return executableOp.emitError("unable to find spv.module");

  SmallVector<spirv::FuncOp, 2> spvEntryPointFns;
  if (!entryPointScheduleAttr) {
    for (spirv::FuncOp spvFuncOp : spvModuleOp.getOps<spirv::FuncOp>()) {
      if (SymbolTable::getSymbolVisibility(spvFuncOp) ==
          SymbolTable::Visibility::Public)
        spvEntryPointFns.push_back(spvFuncOp);
    }
    if (!llvm::hasSingleElement(spvEntryPointFns)) {
      return spvModuleOp.emitError(
                 "expected a single entry point function, found ")
             << spvEntryPointFns.size();
    }
  } else {
    llvm::StringMap<spirv::FuncOp> publicFns;
    for (spirv::FuncOp spvFuncOp : spvModuleOp.getOps<spirv::FuncOp>()) {
      if (SymbolTable::getSymbolVisibility(spvFuncOp) ==
          SymbolTable::Visibility::Public)
        publicFns[spvFuncOp.sym_name()] = spvFuncOp;
    }
    for (Attribute entryNameAttr : entryPointScheduleAttr) {
      StringRef entryName = entryNameAttr.cast<StringAttr>().getValue();
      spirv::FuncOp spvFuncOp = publicFns.lookup(entryName);
      if (!spvFuncOp)
        return spvModuleOp.emitError("unable to find entry point function ")
               << entryName;
      spvEntryPointFns.push_back(spvFuncOp);
    }
  }

  auto *region = switchBuilder.addConditionRegion(
      IREE::HAL::DeviceMatchIDAttr::get(filter_pattern(), loc.getContext()),
      {
          dispatchState.workload,
          dispatchState.commandBuffer,
          dispatchState.executable,
      });

  auto &entryBlock = region->front();
  auto builder = OpBuilder::atBlockBegin(&entryBlock);
  auto workload = entryBlock.getArgument(0);
  auto commandBuffer = entryBlock.getArgument(1);
  auto executable = entryBlock.getArgument(2);

  // We have multiple entry points to dispatch. Record in the order
  // specified by entry point schedule and insert barrier between sequential
  // ones.
  for (auto it : llvm::enumerate(spvEntryPointFns)) {
    spirv::FuncOp spvFuncOp = it.value();
    auto workgroupSize = calculateDispatchWorkgroupSize(
        loc, spvModuleOp, spvFuncOp.sym_name(), workload, builder);

    StringRef workgroupCountAttrName = getWorkgroupCountAttrName();
    IntegerAttr workgroupCountAttr =
        spvFuncOp.getAttrOfType<IntegerAttr>(workgroupCountAttrName);
    if (!workgroupCountAttr)
      return spvFuncOp.emitError("missing attribute ")
             << workgroupCountAttrName;

    // Assuming here that the shape of the first result value of the dispatch
    // region is enough to calculate the number of workgroups. Either
    // - All results have the same shape and the `workgroupCountMethod` is set
    //   to WorkgroupCountMethodology::ResultShape, or
    // - All the results have the same linearized shape and the
    //   `workgourpCountMethod` is set to
    //   WorkgroupCountMethodology::LinearizedResultShape.
    Optional<SmallVector<Value, 4>> resultShape =
        getFirstResultShape(loc, dispatchState, builder);

    WorkgroupCountMethodology workgroupCountMethod =
        static_cast<WorkgroupCountMethodology>(
            workgroupCountAttr.getValue().getZExtValue());

    std::array<Value, 3> workgroupCount = {nullptr, nullptr, nullptr};
    if (resultShape &&
        workgroupCountMethod != WorkgroupCountMethodology::Default) {
      workgroupCount = calculateSPIRVDispatchWorkgroupCount(
          loc, *resultShape, workgroupCountAttr, workgroupSize, builder);
    } else {
      workgroupCount = calculateDispatchWorkgroupCount(loc, workload,
                                                       workgroupSize, builder);
    }

    if (llvm::any_of(workgroupCount,
                     [](Value v) -> bool { return v == nullptr; }))
      return spvFuncOp.emitError("unable to find workgroup count");

    // Ordinals are fixed based on the precomputed schedule, so use
    // CommandBufferDispatchOp instead of CommandBufferDispatchSymbolOp.
    builder.create<IREE::HAL::CommandBufferDispatchOp>(
        loc, commandBuffer, executable,
        builder.getI32IntegerAttr(/*entryPointOrdinal=*/it.index()),
        workgroupCount[0], workgroupCount[1], workgroupCount[2]);
    if (it.index() + 1 != spvEntryPointFns.size()) {
      recordFullExecutionBarrier(commandBuffer, loc, builder);
    }
  }

  builder.create<IREE::HAL::ReturnOp>(loc);
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
    IREE::HAL::ExecutableEntryPointOp entryPointOp, Value workload,
    OpBuilder &builder) {
  // TODO(ravishankarm): possibly emit different recordDispatch logic if the
  // workgroup sizes differ among targets.
  spirv::ModuleOp spvModuleOp;
  for (auto executableTargetOp :
       executableOp.getBlock().getOps<IREE::HAL::ExecutableTargetOp>()) {
    if (matchPattern(executableTargetOp.target_backend_filter(),
                     filter_pattern())) {
      ModuleOp innerModuleOp = executableTargetOp.getInnerModule();
      assert(!innerModuleOp.getAttr(
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
    Value workload, OpBuilder &builder) {
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
