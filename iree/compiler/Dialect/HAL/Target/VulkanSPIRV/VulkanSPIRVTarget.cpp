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

#include "iree/compiler/Conversion/Common/Attributes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/CodeGenOptionUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/SPIRVCommon/SPIRVTarget.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanDialect.h"
#include "iree/compiler/Dialect/Vulkan/Utils/TargetEnvUtils.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/spirv_executable_def_builder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser.h"
#include "mlir/Target/SPIRV/Serialization.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

VulkanSPIRVTargetOptions getVulkanSPIRVTargetOptionsFromFlags() {
  // TODO(antiagainst): Enable option categories once the following bug is
  // fixed: https://bugs.llvm.org/show_bug.cgi?id=44223 static
  // llvm::cl::OptionCategory halVulkanSPIRVOptionsCategory(
  //     "IREE Vulkan/SPIR-V backend options");

  static llvm::cl::opt<std::string> clVulkanTargetTriple(
      "iree-vulkan-target-triple", llvm::cl::desc("Vulkan target triple"),
      llvm::cl::init("swiftshader-unknown-unknown"));

  static llvm::cl::opt<std::string> clVulkanTargetEnv(
      "iree-vulkan-target-env",
      llvm::cl::desc(
          "Vulkan target environment as #vk.target_env attribute assembly"),
      llvm::cl::init(""));

  VulkanSPIRVTargetOptions targetOptions;
  targetOptions.codegenOptions = getSPIRVCodegenOptionsFromClOptions();
  if (!clVulkanTargetEnv.empty()) {
    targetOptions.vulkanTargetEnv = clVulkanTargetEnv;
  } else {
    targetOptions.vulkanTargetEnv =
        Vulkan::getTargetEnvForTriple(clVulkanTargetTriple);
  }

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
      << "cannot parse vulkan target environment as #vk.target_env attribute: '"
      << vulkanTargetEnv << "'";
  return {};
}

class VulkanSPIRVTargetBackend : public SPIRVTargetBackend {
 public:
  VulkanSPIRVTargetBackend(VulkanSPIRVTargetOptions options)
      : SPIRVTargetBackend(options.codegenOptions),
        options_(std::move(options)) {}

  // NOTE: we could vary these based on the options such as 'vulkan-v1.1'.
  std::string name() const override { return "vulkan_spirv"; }
  std::string filter_pattern() const override { return "vulkan*"; }

  BufferConstraintsAttr queryBufferConstraints(MLIRContext *context) override {
    // Picked from here to start:
    // https://vulkan.gpuinfo.org/displaydevicelimit.php?name=minStorageBufferOffsetAlignment&platform=android
    // https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxStorageBufferRange&platform=android
    // We should instead be querying the vulkan environment attributes.
    uint64_t maxAllocationSize = 1 * 1024 * 1024 * 1024ull;
    uint64_t minBufferOffsetAlignment = 256ull;
    uint64_t maxBufferRange = 128 * 1024 * 1024ull;
    uint64_t minBufferRangeAlignment = 16ull;
    Builder b(context);
    return BufferConstraintsAttr::get(b.getIndexAttr(maxAllocationSize),
                                      b.getIndexAttr(minBufferOffsetAlignment),
                                      b.getIndexAttr(maxBufferRange),
                                      b.getIndexAttr(minBufferRangeAlignment));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<Vulkan::VulkanDialect, spirv::SPIRVDialect>();
  }

  void declareTargetOps(IREE::Flow::ExecutableOp sourceOp,
                        IREE::HAL::ExecutableOp executableOp) override {
    spirv::TargetEnvAttr spvTargetEnv =
        getSPIRVTargetEnv(options_.vulkanTargetEnv, sourceOp.getContext());
    declareTargetOpsForEnv(sourceOp, executableOp, spvTargetEnv);
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder &executableBuilder) override {
    ModuleOp innerModuleOp = targetOp.getInnerModule();
    auto spvModuleOp = *innerModuleOp.getOps<spirv::ModuleOp>().begin();

    // Serialize the spirv::ModuleOp into the binary that we will embed in the
    // final flatbuffer.
    FlatbufferBuilder builder;
    SmallVector<uint32_t, 256> spvBinary;
    if (failed(spirv::serialize(spvModuleOp, spvBinary)) || spvBinary.empty()) {
      return targetOp.emitError() << "failed to serialize spv.module";
    }
    auto spvCodeRef = flatbuffers_uint32_vec_create(builder, spvBinary.data(),
                                                    spvBinary.size());

    // The sequencer and runtime use ordinals instead of names. We provide the
    // list of entry point names here that are then passed in
    // VkShaderModuleCreateInfo.
    SmallVector<StringRef, 8> entryPointNames;
    if (auto scheduleAttr = innerModuleOp->getAttrOfType<ArrayAttr>(
            iree_compiler::getEntryPointScheduleAttrName())) {
      // We have multiple entry points in this module. Make sure the order
      // specified in the schedule attribute is respected.
      for (Attribute entryPoint : scheduleAttr) {
        entryPointNames.push_back(
            entryPoint.cast<FlatSymbolRefAttr>().getValue());
      }
    } else {
      spvModuleOp.walk([&](spirv::EntryPointOp entryPointOp) {
        entryPointNames.push_back(entryPointOp.fn());
      });
    }
    auto entryPointsRef = builder.createStringVec(entryPointNames);

    iree_SpirVExecutableDef_start_as_root(builder);
    iree_SpirVExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_SpirVExecutableDef_code_add(builder, spvCodeRef);
    iree_SpirVExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(), targetOp.sym_name(),
        executableBuilder.getStringAttr("SPVE"),
        builder.getBufferAttr(executableBuilder.getContext()));

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
