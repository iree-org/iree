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

// Returns a pipeline layout definition based on the bindings required.
static std::unique_ptr<iree::VkPipelineLayoutDefT> populatePipelineLayout(
    spirv::ModuleOp spirvModuleOp) {
  // NOTE: we currently make some assumptions about this based on the expected
  // ABI of the runtime. If we wanted to support more general shaders with more
  // complex I/O we'd need to find a better way to communicate this through the
  // VkPipelineLayoutDef.
  auto pipelineLayoutDef = std::make_unique<iree::VkPipelineLayoutDefT>();
  pipelineLayoutDef->buffer_binding_set = 0;

  // Build a set of descriptor_set -> binding -> variable.
  // This makes it easier to write out the descriptor in a logical order, even
  // though this is not strictly required.
  int64_t maxDescriptorSetOrdinal = -1;
  std::map<int32_t, std::map<int32_t, spirv::GlobalVariableOp>> descriptorSets;
  for (auto globalVar :
       spirvModuleOp.getBlock().getOps<spirv::GlobalVariableOp>()) {
    auto descriptorSetAttr =
        globalVar.getAttrOfType<IntegerAttr>("descriptor_set");
    auto bindingAttr = globalVar.getAttrOfType<IntegerAttr>("binding");
    if (!descriptorSetAttr || !bindingAttr) {
      // Not something the runtime cares about.
      continue;
    }
    maxDescriptorSetOrdinal =
        std::max(descriptorSetAttr.getInt(), maxDescriptorSetOrdinal);
    auto &descriptorSet = descriptorSets[descriptorSetAttr.getInt()];
    descriptorSet[bindingAttr.getInt()] = globalVar;
  }

  // Create the individual layout and binding defs.
  pipelineLayoutDef->descriptor_set_layouts.resize(maxDescriptorSetOrdinal + 1);
  for (auto &descriptorSetBindings : descriptorSets) {
    int32_t descriptorSet = descriptorSetBindings.first;
    auto dsl = std::make_unique<iree::VkDescriptorSetLayoutDefT>();

    for (auto &globalVarBinding : descriptorSetBindings.second) {
      auto binding = std::make_unique<iree::VkDescriptorSetLayoutBindingDefT>();
      binding->binding = globalVarBinding.first;
      binding->descriptor_count = 1;
      // TODO(benvanik): pull from type info.
      binding->descriptor_type = 7;       // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
      binding->stage_flags = 0x00000020;  // VK_SHADER_STAGE_COMPUTE_BIT
      dsl->bindings.push_back(std::move(binding));
    }

    pipelineLayoutDef->descriptor_set_layouts[descriptorSet] = std::move(dsl);
  }

  return pipelineLayoutDef;
}

// Returns an (x,y,z) workgroup size for the given |targetFuncOp|.
// This is pure heuristics until we support dynamic/varying workgroup sizes.
static std::array<int32_t, 3> guessWorkGroupSize(
    IREE::Flow::DispatchEntryOp entryOp, FuncOp targetFuncOp) {
  for (auto &block : targetFuncOp.getBlocks()) {
    if (!block.getOps<xla_hlo::DotOp>().empty()) {
      // A special dot kernel. This has a fixed workgroup size based on the
      // hand-written shader.
      return {16, 16, 1};
    } else if (!block.getOps<xla_hlo::ConvOp>().empty()) {
      // Matches hard-coded assumptions in the conv2d_nhwc hand-written
      // shader.
      return {1, 1, 1};
    }
  }
  return {32, 1, 1};
}

// Applies the guessWorkGroupSize logic to each entry point to ensure we have
// a compatible size set on entry.
// TODO(b/150312935): remove this and just call guessWorkGroupSize (or whatever)
// when you need the workgroup size.
static void guessEntryWorkGroupSizes(IREE::Flow::ExecutableOp sourceOp,
                                     ModuleOp moduleOp) {
  Builder builder(sourceOp.getContext());
  for (auto &op : sourceOp.getBlock()) {
    if (auto entryOp = dyn_cast<IREE::Flow::DispatchEntryOp>(&op)) {
      auto funcOp = moduleOp.lookupSymbol<FuncOp>(entryOp.function_ref());
      auto workGroupSizeAttr = DenseIntElementsAttr::get(
          VectorType::get(3, builder.getIntegerType(32)),
          guessWorkGroupSize(entryOp, funcOp));
      funcOp.setAttr("iree.executable.workgroup_size", workGroupSizeAttr);
    }
  }
}

// Update the workgroup size in the executableOp.
// TODO(b/150312935): remove this and just write the
// IREE::HAL::ExecutableEntryPointOp attributes when you need them.
static void propagateModifiedExecutableABI(
    IREE::Flow::ExecutableOp sourceOp, ModuleOp moduleOp,
    IREE::HAL::ExecutableOp executableOp) {
  for (auto &op : sourceOp.getBlock()) {
    if (auto entryOp = dyn_cast<IREE::Flow::DispatchEntryOp>(&op)) {
      auto targetEntryOp =
          executableOp.lookupSymbol<IREE::HAL::ExecutableEntryPointOp>(
              entryOp.function_ref());
      assert(targetEntryOp && "could not find HAL entry point");
      auto funcOp = moduleOp.lookupSymbol<FuncOp>(entryOp.function_ref());
      assert(funcOp && "could not find target function for HAL entry point");
      auto workGroupSize = funcOp.getAttrOfType<DenseIntElementsAttr>(
          "iree.executable.workgroup_size");
      targetEntryOp.setAttr("workgroup_size", workGroupSize);
    }
  }
}

/// Returns true if the linalg on tensors path is to be used for
/// compilation.
static bool useLinalgPath(ModuleOp moduleOp,
                          VulkanSPIRVTargetOptions const &targetOptions) {
  if (targetOptions.useLinalgToSPIRVPath) return true;

  // Use linalg path if dispatch function contains any of the following ops.
  SmallVector<FuncOp, 1> dispatchFn;
  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    if (isDispatchFunction(funcOp)) dispatchFn.push_back(funcOp);
  }
  if (dispatchFn.size() != 1) return false;
  auto walkResult = dispatchFn[0].walk([](Operation *op) -> WalkResult {
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
  auto flowExecutableOp =
      *sourceOp.getInnerModule().getOps<IREE::Flow::ExecutableOp>().begin();
  auto moduleOp = flowExecutableOp.getInnerModule();
  if (failed(makeLegacyExecutableABI(sourceOp))) {
    return failure();
  }
  guessEntryWorkGroupSizes(flowExecutableOp, moduleOp);

  // Attach SPIR-V target environment.
  spirv::TargetEnvAttr spvTargetEnv =
      getSPIRVTargetEnv(targetOptions.vulkanTargetEnv, moduleOp.getContext());
  moduleOp.setAttr(spirv::getTargetEnvAttrName(), spvTargetEnv);

  // Try first to match against an embedded kernel (such as matmul) and
  // otherwise fall back to generating the kernel.
  iree::SpirVExecutableDefT spirvExecutableDef;
  if (tryEmbeddedKernelRewrite(moduleOp, &spirvExecutableDef)) {
    // Strip out the contents as we don't care (they were manually replaced).
    moduleOp.getBody()->getOperations().erase(
        moduleOp.getBody()->getOperations().begin(),
        --moduleOp.getBody()->getOperations().end());
  } else {
    // The sequencer and runtime use ordinals instead of names. We provide the
    // list of entry point names here that are then passed in
    // VkShaderModuleCreateInfo.
    spirvExecutableDef.entry_points = populateEntryPointNames(flowExecutableOp);

    // Lower module to spirv::ModuleOp.
    PassManager conversionPassManager(moduleOp.getContext());
    if (useLinalgPath(moduleOp, targetOptions)) {
      addHLOToLinalgToSPIRVPasses(conversionPassManager,
                                  targetOptions.linalgToSPIRVWorkgroupSize);
    } else {
      // Use the Index computation path as fallback.
      addIREEToSPIRVPasses(conversionPassManager);
    }
    if (failed(conversionPassManager.run(moduleOp))) {
      return moduleOp.emitError() << "failed to run conversion passes";
    }

    // Drop the gpu.container_module attribute.
    moduleOp.removeAttr("gpu.container_module");
    propagateModifiedExecutableABI(flowExecutableOp, moduleOp, executableOp);
    auto spvModuleOps = moduleOp.getOps<spirv::ModuleOp>();
    if (std::distance(spvModuleOps.begin(), spvModuleOps.end()) != 1) {
      return moduleOp.emitError()
             << "Expected a single spv.module for an IREE executable op";
    }
    spirv::ModuleOp spvModuleOp = *spvModuleOps.begin();

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

    // Reflect against the entry thunk to identify the required pipeline
    // layout based on binding information. This is used by the runtime to
    // create the VkPipelineLayout.
    spirvExecutableDef.pipeline_layout = populatePipelineLayout(spvModuleOp);
    if (!spirvExecutableDef.pipeline_layout) {
      return spvModuleOp.emitError()
             << "failed to generate pipeline for SPIR-V module";
    }

    // Remove the original functions as we just want to keep the spv.module for
    // debugging.
    for (auto &op :
         llvm::make_early_inc_range(moduleOp.getBody()->getOperations())) {
      if (!isa<spirv::ModuleOp>(op) && !isa<ModuleTerminatorOp>(op)) {
        op.erase();
      }
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
  OpBuilder targetBuilder(&executableOp.getBlock());
  targetBuilder.setInsertionPoint(&executableOp.getBlock().back());
  auto binaryOp = targetBuilder.create<IREE::HAL::ExecutableBinaryOp>(
      executableOp.getLoc(),
      static_cast<uint32_t>(IREE::HAL::ExecutableFormat::SpirV),
      std::move(bytes));
  OpBuilder binaryBuilder(&binaryOp.getBlock().back());
  binaryBuilder.clone(*moduleOp.getOperation());
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
