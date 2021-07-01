// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/VulkanSPIRV/VulkanSPIRVTarget.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/SPIRVCommon/SPIRVTarget.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanDialect.h"
#include "iree/compiler/Dialect/Vulkan/Utils/TargetEnvironment.h"
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
      llvm::cl::init("cpu-swiftshader-unknown"));

  static llvm::cl::opt<std::string> clVulkanTargetEnv(
      "iree-vulkan-target-env",
      llvm::cl::desc(
          "Vulkan target environment as #vk.target_env attribute assembly"),
      llvm::cl::init(""));

  VulkanSPIRVTargetOptions targetOptions;
  targetOptions.codegenOptions = SPIRVCodegenOptions::getFromCLOptions();
  targetOptions.vulkanTargetEnv = clVulkanTargetEnv;
  targetOptions.vulkanTargetTriple = clVulkanTargetTriple;

  return targetOptions;
}

// Returns the Vulkan target environment for conversion.
static spirv::TargetEnvAttr getSPIRVTargetEnv(
    const std::string &vulkanTargetEnv, const std::string &vulkanTargetTriple,
    MLIRContext *context) {
  if (!vulkanTargetEnv.empty()) {
    if (auto attr = mlir::parseAttribute(vulkanTargetEnv, context)) {
      if (auto vkTargetEnv = attr.dyn_cast<Vulkan::TargetEnvAttr>()) {
        return convertTargetEnv(vkTargetEnv);
      }
    }
    emitError(Builder(context).getUnknownLoc())
        << "cannot parse vulkan target environment as #vk.target_env "
           "attribute: '"
        << vulkanTargetEnv << "'";
  } else if (!vulkanTargetTriple.empty()) {
    return convertTargetEnv(
        Vulkan::getTargetEnvForTriple(context, vulkanTargetTriple));
  }

  return {};
}

class VulkanSPIRVTargetBackend : public SPIRVTargetBackend {
 public:
  VulkanSPIRVTargetBackend(VulkanSPIRVTargetOptions options)
      : SPIRVTargetBackend(options.codegenOptions),
        options_(std::move(options)) {}

  // NOTE: we could vary these based on the options such as 'vulkan-v1.1'.
  std::string name() const override { return "vulkan"; }

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

  void declareVariantOps(IREE::Flow::ExecutableOp sourceOp,
                         IREE::HAL::ExecutableOp executableOp) override {
    spirv::TargetEnvAttr spvTargetEnv =
        getSPIRVTargetEnv(options_.vulkanTargetEnv, options_.vulkanTargetTriple,
                          sourceOp.getContext());
    declareVariantOpsForEnv(sourceOp, executableOp, spvTargetEnv);
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    ModuleOp innerModuleOp = variantOp.getInnerModule();
    auto spvModuleOp = *innerModuleOp.getOps<spirv::ModuleOp>().begin();

    FlatbufferBuilder builder;
    iree_SpirVExecutableDef_start_as_root(builder);

    // Serialize the spirv::ModuleOp into the binary that we will embed in the
    // final flatbuffer.
    SmallVector<uint32_t, 256> spvBinary;
    if (failed(spirv::serialize(spvModuleOp, spvBinary)) || spvBinary.empty()) {
      return variantOp.emitError() << "failed to serialize spv.module";
    }
    auto spvCodeRef = flatbuffers_uint32_vec_create(builder, spvBinary.data(),
                                                    spvBinary.size());

    // The sequencer and runtime use ordinals instead of names. We provide the
    // list of entry point names here that are then passed in
    // VkShaderModuleCreateInfo.
    SmallVector<StringRef, 8> entryPointNames;
    spvModuleOp.walk([&](spirv::EntryPointOp entryPointOp) {
      entryPointNames.push_back(entryPointOp.fn());
    });
    auto entryPointsRef = builder.createStringVec(entryPointNames);

    iree_SpirVExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_SpirVExecutableDef_code_add(builder, spvCodeRef);
    iree_SpirVExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.sym_name(),
        executableBuilder.getStringAttr("SPVE"),
        builder.getBufferAttr(executableBuilder.getContext()));
    binaryOp.mime_typeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();
  }

 protected:
  VulkanSPIRVTargetOptions options_;
};

void registerVulkanSPIRVTargetBackends(
    std::function<VulkanSPIRVTargetOptions()> queryOptions) {
  getVulkanSPIRVTargetOptionsFromFlags();
  auto backendFactory = [=]() {
    return std::make_unique<VulkanSPIRVTargetBackend>(queryOptions());
  };
  static TargetBackendRegistration registration0("vulkan", backendFactory);
  static TargetBackendRegistration registration1("vulkan-spirv",
                                                 backendFactory);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
