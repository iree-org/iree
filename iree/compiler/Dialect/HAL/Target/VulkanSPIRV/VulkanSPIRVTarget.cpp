// Copyright 2019 Google LLC
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

#include "iree/compiler/Dialect/HAL/Target/VulkanSPIRV/VulkanSPIRVTarget.h"

#include <map>

#include "flatbuffers/flatbuffers.h"
#include "iree/compiler/Conversion/HLOToLinalg/Passes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Attributes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"
#include "iree/compiler/Dialect/Vulkan/Utils/TargetEnvUtils.h"
#include "iree/schemas/spirv_executable_def_generated.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

VulkanSPIRVTargetOptions getVulkanSPIRVTargetOptionsFromFlags() {
  // TODO(antiagainst): Enable option categories once the following bug is
  // fixed: https://bugs.llvm.org/show_bug.cgi?id=44223 static
  // llvm::cl::OptionCategory halVulkanSPIRVOptionsCategory(
  //     "IREE Vulkan/SPIR-V backend options");

  static llvm::cl::opt<bool> clUseWorkgroupMemory(
      "iree-spirv-use-workgroup-memory",
      llvm::cl::desc(
          "Enable use of workgroup memory in SPIR-V code generation"),
      llvm::cl::init(false));

  static llvm::cl::list<unsigned> clWorkgroupSize(
      "iree-spirv-workgroup-size",
      llvm::cl::desc(
          "Workgroup size to use for XLA-HLO to Linalg to SPIR-V path"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

  static llvm::cl::list<unsigned> clTileSizes(
      "iree-spirv-tile-size",
      llvm::cl::desc("Tile size to use for tiling Linalg operations"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

  static llvm::cl::opt<std::string> clVulkanTargetEnv(
      "iree-vulkan-target-env",
      llvm::cl::desc(
          "Vulkan target environment as #vk.target_env attribute assembly"),
      llvm::cl::init(Vulkan::swiftShaderTargetEnvAssembly));

  VulkanSPIRVTargetOptions targetOptions;
  targetOptions.codegenOptions.workgroupSize.assign(clWorkgroupSize.begin(),
                                                    clWorkgroupSize.end());
  targetOptions.codegenOptions.tileSizes.assign(clTileSizes.begin(),
                                                clTileSizes.end());
  targetOptions.codegenOptions.useWorkgroupMemory = clUseWorkgroupMemory;
  targetOptions.vulkanTargetEnv = clVulkanTargetEnv;
  return targetOptions;
}

// Returns the Vulkan target environment for conversion.
static spirv::TargetEnvAttr getSPIRVTargetEnv(
    const std::string &vulkanTargetEnv, MLIRContext *context) {
  if (auto attr = mlir::parseAttribute(vulkanTargetEnv, context)) {
    if (auto vkTargetEnv = attr.dyn_cast<Vulkan::TargetEnvAttr>()) {
      return convertTargetEnv(vkTargetEnv);
    }
  }

  emitError(Builder(context).getUnknownLoc())
      << "cannot parse vulkan target environment as #vk.target_env attribute ";
  return {};
}

// Returns a list of entry point names matching the expected export ordinals.
static std::vector<std::string> populateEntryPointNames(
    spirv::ModuleOp spvModuleOp) {
  std::vector<std::string> entryPointNames;
  spvModuleOp.walk([&](spirv::EntryPointOp entryPointOp) {
    entryPointNames.push_back(std::string(entryPointOp.fn()));
  });
  return entryPointNames;
}

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

class VulkanSPIRVTargetBackend : public TargetBackend {
 public:
  VulkanSPIRVTargetBackend(VulkanSPIRVTargetOptions options)
      : options_(std::move(options)) {}

  // NOTE: we could vary this based on the options such as 'vulkan-v1.1'.
  std::string name() const override { return "vulkan*"; }

  void constructTargetOps(IREE::Flow::ExecutableOp sourceOp,
                          IREE::HAL::ExecutableOp executableOp) override {
    // Attach SPIR-V target environment to the hal.executable.target op.
    // If we had multiple target environments we would generate one target op
    // per environment.
    spirv::TargetEnvAttr spvTargetEnv =
        getSPIRVTargetEnv(options_.vulkanTargetEnv, sourceOp.getContext());

    OpBuilder targetBuilder(&executableOp.getBlock().back());
    auto targetOp = targetBuilder.create<IREE::HAL::ExecutableTargetOp>(
        sourceOp.getLoc(), name());
    OpBuilder containerBuilder(&targetOp.getBlock().back());
    auto innerModuleOp =
        containerBuilder.clone(*sourceOp.getInnerModule().getOperation());
    innerModuleOp->setAttr(spirv::getTargetEnvAttrName(), spvTargetEnv);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpPassManager &passManager) override {
    buildSPIRVTransformPassPipeline(passManager, options_.codegenOptions);
  }

  LogicalResult recordDispatch(Location loc, DispatchState dispatchState,
                               DeviceSwitchBuilder &switchBuilder) override {
    // Multiple entry points might be generated for a single dispatch function.
    // Under such circumstances, we will have a special attribute indicating the
    // schedule of the split entry points. Try to see if we can find such
    // schedule attribute first.
    ArrayAttr entryPointScheduleAttr;
    spirv::ModuleOp spvModuleOp;
    IREE::HAL::ExecutableOp executableOp = dispatchState.executableOp;
    for (auto executableTargetOp :
         executableOp.getBlock().getOps<IREE::HAL::ExecutableTargetOp>()) {
      if (matchPattern(executableTargetOp.target_backend(), name())) {
        ModuleOp innerModuleOp = executableTargetOp.getInnerModule();
        auto spvModuleOps = innerModuleOp.getOps<spirv::ModuleOp>();
        assert(llvm::hasSingleElement(spvModuleOps));
        spvModuleOp = *spvModuleOps.begin();
        entryPointScheduleAttr = innerModuleOp.getAttrOfType<ArrayAttr>(
            iree_compiler::getEntryPointScheduleAttrName());
        break;
      }
    }
    if (!spvModuleOp)
      return executableOp.emitError("unable to find spv.module");

    SmallVector<spirv::FuncOp, 2> entryPointFns;
    if (!entryPointScheduleAttr) {
      for (spirv::FuncOp spvFuncOp : spvModuleOp.getOps<spirv::FuncOp>()) {
        if (SymbolTable::getSymbolVisibility(spvFuncOp) ==
            SymbolTable::Visibility::Public)
          entryPointFns.push_back(spvFuncOp);
      }
      if (!llvm::hasSingleElement(entryPointFns)) {
        return spvModuleOp.emitError(
                   "expected a single entry point function, found ")
               << entryPointFns.size();
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
        entryPointFns.push_back(spvFuncOp);
      }
    }

    auto *region = switchBuilder.addConditionRegion(
        IREE::HAL::DeviceMatchIDAttr::get(name(), loc.getContext()),
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
    for (auto it : llvm::enumerate(entryPointFns)) {
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
        workgroupCount = calculateDispatchWorkgroupCount(
            loc, workload, workgroupSize, builder);
      }

      if (llvm::any_of(workgroupCount,
                       [](Value v) -> bool { return v == nullptr; }))
        return spvFuncOp.emitError("unable to find workgroup count");

      builder.create<IREE::HAL::CommandBufferDispatchOp>(
          loc, commandBuffer, executable, /*entryPointOrdinal=*/it.index(),
          workgroupCount[0], workgroupCount[1], workgroupCount[2]);
      if (it.index() + 1 != entryPointFns.size()) {
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
  std::array<Value, 3> calculateDispatchWorkgroupSize(
      Location loc, IREE::HAL::ExecutableOp executableOp,
      IREE::HAL::ExecutableEntryPointOp entryPointOp, Value workload,
      OpBuilder &builder) override {
    // TODO(ravishankarm): possibly emit different recordDispatch logic if the
    // workgroup sizes differ among targets.
    spirv::ModuleOp spvModuleOp;
    for (auto executableTargetOp :
         executableOp.getBlock().getOps<IREE::HAL::ExecutableTargetOp>()) {
      if (matchPattern(executableTargetOp.target_backend(), name())) {
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

  std::array<Value, 3> calculateDispatchWorkgroupSize(
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

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder &executableBuilder) override {
    iree::SpirVExecutableDefT spirvExecutableDef;

    ModuleOp innerModuleOp = targetOp.getInnerModule();
    auto spvModuleOp = *innerModuleOp.getOps<spirv::ModuleOp>().begin();

    // The sequencer and runtime use ordinals instead of names. We provide the
    // list of entry point names here that are then passed in
    // VkShaderModuleCreateInfo.
    if (auto scheduleAttr = innerModuleOp.getAttrOfType<ArrayAttr>(
            iree_compiler::getEntryPointScheduleAttrName())) {
      // We have multiple entry points in this module. Make sure the order
      // specified in the schedule attribute is respected.
      for (Attribute entryPoint : scheduleAttr) {
        spirvExecutableDef.entry_points.emplace_back(
            entryPoint.cast<StringAttr>().getValue().str());
      }
    } else {
      spirvExecutableDef.entry_points = populateEntryPointNames(spvModuleOp);
    }

    // Serialize the spirv::ModuleOp into the binary that we will embed in the
    // final flatbuffer.
    SmallVector<uint32_t, 256> spvBinary;
    if (failed(spirv::serialize(spvModuleOp, spvBinary))) {
      return targetOp.emitError() << "failed to serialize spv.module";
    }
    spirvExecutableDef.code = {spvBinary.begin(), spvBinary.end()};
    if (spirvExecutableDef.code.empty()) {
      return targetOp.emitError()
             << "failed to translate and serialize SPIR-V executable";
    }

    // Pack the executable definition and get the bytes with the proper header.
    // The header is used to verify the contents at runtime.
    ::flatbuffers::FlatBufferBuilder fbb;
    auto executableOffset =
        iree::SpirVExecutableDef::Pack(fbb, &spirvExecutableDef);
    iree::FinishSpirVExecutableDefBuffer(fbb, executableOffset);
    std::vector<uint8_t> bytes;
    bytes.resize(fbb.GetSize());
    std::memcpy(bytes.data(), fbb.GetBufferPointer(), bytes.size());

    // Add the binary data to the target executable.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(),
        static_cast<uint32_t>(IREE::HAL::ExecutableFormat::SpirV),
        std::move(bytes));

    return success();
  }

 protected:
  VulkanSPIRVTargetOptions options_;
};

void registerVulkanSPIRVTargetBackends(
    std::function<VulkanSPIRVTargetOptions()> queryOptions) {
  getVulkanSPIRVTargetOptionsFromFlags();
  static TargetBackendRegistration registration("vulkan-spirv", [=]() {
    return std::make_unique<VulkanSPIRVTargetBackend>(queryOptions());
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
