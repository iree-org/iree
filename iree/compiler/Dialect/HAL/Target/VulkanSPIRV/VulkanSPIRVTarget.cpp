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
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/LegacyUtil.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"
#include "iree/compiler/Dialect/Vulkan/Utils/TargetEnvUtils.h"
#include "iree/compiler/Translation/CodegenPasses/Passes.h"
#include "iree/compiler/Translation/CodegenUtils/CodegenUtils.h"
#include "iree/compiler/Translation/SPIRV/LinalgToSPIRV/LowerToSPIRV.h"
#include "iree/compiler/Translation/SPIRV/XLAToSPIRV/IREEToSPIRVPass.h"
#include "iree/schemas/spirv_executable_def_generated.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

VulkanSPIRVTargetOptions getVulkanSPIRVTargetOptionsFromFlags() {
  // TODO(antiagainst): Enable option categories once the following bug is
  // fixed: https://bugs.llvm.org/show_bug.cgi?id=44223 static
  // llvm::cl::OptionCategory halVulkanSPIRVOptionsCategory(
  //     "IREE Vulkan/SPIR-V backend options");

  // TODO(ravishankarm): Flags to test the Linalg To SPIR-V path. Need a better
  // way to handle these options.
  static llvm::cl::opt<bool> clUseLinalgPath(
      "iree-use-linalg-to-spirv-path",
      llvm::cl::desc("Use the XLA-HLO to Linalg To SPIR-V pass pipeline"),
      llvm::cl::init(false));

  static llvm::cl::list<unsigned> clLinalgPathWorkgroupSize(
      "iree-linalg-to-spirv-workgroup-size",
      llvm::cl::desc(
          "Workgroup size to use for XLA-HLO to Linalg to SPIR-V path"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated);

  static llvm::cl::opt<std::string> clVulkanTargetEnv(
      "iree-vulkan-target-env",
      llvm::cl::desc(
          "Vulkan target environment as #vk.target_env attribute assembly"),
      llvm::cl::init(Vulkan::swiftShaderTargetEnvAssembly));

  VulkanSPIRVTargetOptions targetOptions;
  targetOptions.useLinalgToSPIRVPath = clUseLinalgPath;
  for (unsigned dim : clLinalgPathWorkgroupSize) {
    targetOptions.linalgToSPIRVWorkgroupSize.push_back(dim);
  }
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

/// Returns true if the linalg on tensors path is to be used for
/// compilation.
static bool useLinalgPath(ModuleOp moduleOp,
                          VulkanSPIRVTargetOptions const &targetOptions) {
  if (targetOptions.useLinalgToSPIRVPath) return true;

  // Use linalg path if dispatch function contains any of the following ops.
  auto walkResult = moduleOp.walk([](Operation *op) -> WalkResult {
    if (isa<xla_hlo::ReduceOp>(op) || isa<xla_hlo::ConvOp>(op) ||
        isa<xla_hlo::DotOp>(op) || isa<xla_hlo::ReduceWindowOp>(op)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted();
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

    if (useLinalgPath(sourceOp.getInnerModule(), options_)) {
      targetOp.setAttr("vkspv.use_linalg",
                       UnitAttr::get(sourceOp.getContext()));
    }
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpPassManager &passManager) override {
    passManager.addPass(createHALInterfaceToMemrefPass());
    if (targetOp.getAttr("vkspv.use_linalg")) {
      addHLOToLinalgToSPIRVPasses(passManager,
                                  options_.linalgToSPIRVWorkgroupSize);
    } else {
      addIREEToSPIRVPasses(passManager);
    }
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
      if (matchPattern(executableTargetOp.target_backend(), "vulkan*")) {
        auto spvModuleOps =
            executableTargetOp.getInnerModule().getOps<spirv::ModuleOp>();
        assert(!spvModuleOps.empty());
        spvModuleOp = *spvModuleOps.begin();
        break;
      }
    }

    std::array<Value, 3> workgroupSize;
    for (auto executionModeOp :
         spvModuleOp.getBlock().getOps<spirv::ExecutionModeOp>()) {
      if (executionModeOp.fn() == entryPointOp.sym_name() &&
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

    auto spvModuleOp =
        *targetOp.getInnerModule().getOps<spirv::ModuleOp>().begin();

    // The sequencer and runtime use ordinals instead of names. We provide the
    // list of entry point names here that are then passed in
    // VkShaderModuleCreateInfo.
    spirvExecutableDef.entry_points = populateEntryPointNames(spvModuleOp);

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
