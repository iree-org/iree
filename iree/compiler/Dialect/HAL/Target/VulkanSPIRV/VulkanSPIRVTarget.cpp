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
#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"
#include "iree/compiler/Dialect/Vulkan/Utils/TargetEnvUtils.h"
#include "iree/compiler/Translation/CodegenPasses/Passes.h"
#include "iree/compiler/Translation/CodegenUtils/CodegenUtils.h"
#include "iree/compiler/Translation/SPIRV/EmbeddedKernels/EmbeddedKernels.h"
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

// TODO(antiagainst): Enable option categories once the following bug is fixed:
// https://bugs.llvm.org/show_bug.cgi?id=44223
// static llvm::cl::OptionCategory halVulkanSPIRVOptionsCategory(
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

VulkanSPIRVTargetOptions getVulkanSPIRVTargetOptionsFromFlags() {
  VulkanSPIRVTargetOptions targetOptions;

  targetOptions.useLinalgToSPIRVPath = clUseLinalgPath;
  for (unsigned dim : clLinalgPathWorkgroupSize)
    targetOptions.linalgToSPIRVWorkgroupSize.push_back(dim);
  targetOptions.vulkanTargetEnv = clVulkanTargetEnv;

  return targetOptions;
}

// Returns the Vulkan target environment for conversion.
static spirv::TargetEnvAttr getSPIRVTargetEnv(
    const std::string &vulkanTargetEnv, MLIRContext *context) {
  if (auto attr = mlir::parseAttribute(vulkanTargetEnv, context))
    if (auto vkTargetEnv = attr.dyn_cast<Vulkan::TargetEnvAttr>())
      return convertTargetEnv(vkTargetEnv);

  emitError(Builder(context).getUnknownLoc())
      << "cannot parse vulkan target environment as #vk.target_env attribute";
  return {};
}

// Returns a list of entry point names matching the expected export ordinals.
static std::vector<std::string> populateEntryPointNames(
    IREE::Flow::ExecutableOp executableOp) {
  std::vector<std::string> entryPointNames;
  for (auto &op : executableOp.getBlock().getOperations()) {
    if (auto entryOp = dyn_cast<IREE::Flow::DispatchEntryOp>(op)) {
      entryPointNames.push_back(std::string(entryOp.function_ref()));
    }
  }
  return entryPointNames;
}

/// Returns true if the linalg on tensors path is to be used for
/// compilation.
static bool useLinalgPath(ModuleOp moduleOp,
                          VulkanSPIRVTargetOptions const &targetOptions) {
  if (targetOptions.useLinalgToSPIRVPath) return true;

  // Use linalg path if dispatch function contains any of the following ops.
  auto walkResult = moduleOp.walk([](Operation *op) -> WalkResult {
    if (isa<xla_hlo::ReduceOp>(op) || isa<xla_hlo::ConvOp>(op) ||
        isa<xla_hlo::DotOp>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted();
}

LogicalResult translateToVulkanSPIRVExecutable(
    IREE::HAL::ExecutableOp executableOp,
    ExecutableTargetOptions executableOptions,
    VulkanSPIRVTargetOptions targetOptions) {
  // Clone the module containing the things we want to translate. We do this so
  // that multiple targets can pull from the same source without conflicting.
  auto sourceOp = executableOp.getSourceOp().clone();
  auto sourceOpErase =
      llvm::make_scope_exit([&sourceOp]() { sourceOp.erase(); });
  auto sourceModuleOp = sourceOp.getInnerModule();
  auto flowExecutableOp =
      *sourceModuleOp.getOps<IREE::Flow::ExecutableOp>().begin();
  auto flowModuleOp = flowExecutableOp.getInnerModule();

  // Attach SPIR-V target environment.
  spirv::TargetEnvAttr spvTargetEnv = getSPIRVTargetEnv(
      targetOptions.vulkanTargetEnv, flowModuleOp.getContext());
  flowModuleOp.setAttr(spirv::getTargetEnvAttrName(), spvTargetEnv);

  iree::SpirVExecutableDefT spirvExecutableDef;
  // The sequencer and runtime use ordinals instead of names. We provide the
  // list of entry point names here that are then passed in
  // VkShaderModuleCreateInfo.
  spirvExecutableDef.entry_points = populateEntryPointNames(flowExecutableOp);

  // Lower module to spirv::ModuleOp.
  PassManager conversionPassManager(flowModuleOp.getContext());
  applyPassManagerCLOptions(conversionPassManager);
  conversionPassManager.addPass(createHALInterfaceToMemrefPass());
  OpPassManager &innerModulePassManager =
      conversionPassManager.nest<IREE::Flow::ExecutableOp>().nest<ModuleOp>();
  if (useLinalgPath(flowModuleOp, targetOptions)) {
    addHLOToLinalgToSPIRVPasses(innerModulePassManager,
                                targetOptions.linalgToSPIRVWorkgroupSize);
  } else {
    // Use the Index computation path as fallback.
    addIREEToSPIRVPasses(innerModulePassManager);
  }
  if (failed(conversionPassManager.run(sourceModuleOp))) {
    return sourceModuleOp.emitError() << "failed to run conversion passes";
  }

  auto spvModuleOps = flowModuleOp.getOps<spirv::ModuleOp>();
  if (std::distance(spvModuleOps.begin(), spvModuleOps.end()) != 1) {
    return flowModuleOp.emitError()
           << "Expected a single spv.module for an IREE executable op";
  }
  spirv::ModuleOp spvModuleOp = *spvModuleOps.begin();

  // Find the spv.ExecutionMode operation to get the workgroup size from.
  // TODO(ravishankarm): This might not be the only way this is specified. You
  // could also have a spec constant, but that is not generated in the
  // spv.module right now.
  auto halEntryPointOps =
      executableOp.getBlock().getOps<IREE::HAL::ExecutableEntryPointOp>();
  for (auto executionModeOp :
       spvModuleOp.getBlock().getOps<spirv::ExecutionModeOp>()) {
    if (executionModeOp.execution_mode() == spirv::ExecutionMode::LocalSize) {
      auto workGroupSize = llvm::to_vector<3>(llvm::map_range(
          executionModeOp.values(), [](Attribute attr) -> int64_t {
            return attr.cast<IntegerAttr>().getInt();
          }));
      workGroupSize.resize(3, 1);
      // Find the corresponding hal.executable.entry_point.
      for (auto halEntryPointOp : halEntryPointOps) {
        if (executionModeOp.fn() == halEntryPointOp.sym_name()) {
          OpBuilder builder(halEntryPointOp);
          halEntryPointOp.setAttr("workgroup_size",
                                  builder.getIndexArrayAttr(workGroupSize));
        }
      }
    }
  }

  // Serialize the spirv::ModuleOp into the binary that we will embed in the
  // final flatbuffer.
  SmallVector<uint32_t, 256> spvBinary;
  if (failed(spirv::serialize(spvModuleOp, spvBinary))) {
    return spvModuleOp.emitError() << "failed to serialize spv.module";
  }
  spirvExecutableDef.code = {spvBinary.begin(), spvBinary.end()};
  if (spirvExecutableDef.code.empty()) {
    return spvModuleOp.emitError()
           << "failed to translate and serialize SPIR-V executable";
  }

  // Remove the original functions as we just want to keep the spv.module for
  // debugging.
  for (auto &op :
       llvm::make_early_inc_range(flowModuleOp.getBody()->getOperations())) {
    if (!isa<spirv::ModuleOp>(op) && !isa<ModuleTerminatorOp>(op)) {
      op.erase();
    }
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
  OpBuilder targetBuilder = OpBuilder::atBlockEnd(&executableOp.getBlock());
  targetBuilder.setInsertionPoint(&executableOp.getBlock().back());
  auto binaryOp = targetBuilder.create<IREE::HAL::ExecutableBinaryOp>(
      executableOp.getLoc(),
      static_cast<uint32_t>(IREE::HAL::ExecutableFormat::SpirV),
      std::move(bytes));
  OpBuilder binaryBuilder(&binaryOp.getBlock().back());
  binaryBuilder.clone(*flowModuleOp.getOperation());
  return success();
}

static ExecutableTargetRegistration targetRegistration(
    "vulkan-spirv", +[](IREE::HAL::ExecutableOp executableOp,
                        ExecutableTargetOptions executableOptions) {
      return translateToVulkanSPIRVExecutable(
          executableOp, executableOptions,
          getVulkanSPIRVTargetOptionsFromFlags());
    });

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
