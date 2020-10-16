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

#include "flatbuffers/flatbuffers.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Attributes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/Target/SPIRVCommon/SPIRVTarget.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanDialect.h"
#include "iree/compiler/Dialect/Vulkan/Utils/TargetEnvUtils.h"
#include "iree/schemas/spirv_executable_def_generated.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

VulkanSPIRVTargetOptions getVulkanSPIRVTargetOptionsFromFlags() {
  // TODO(antiagainst): Enable option categories once the following bug is
  // fixed: https://bugs.llvm.org/show_bug.cgi?id=44223 static
  // llvm::cl::OptionCategory halVulkanSPIRVOptionsCategory(
  //     "IREE Vulkan/SPIR-V backend options");

  static llvm::cl::opt<bool> clUseVectorPass(
      "iree-spirv-use-vector-pass",
      llvm::cl::desc(
          "Enable use of Linalg vectorization in SPIR-V code generation"),
      llvm::cl::init(false));

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

  static llvm::cl::opt<std::string> clVulkanTargetTriple(
      "iree-vulkan-target-triple", llvm::cl::desc("Vulkan target triple"),
      llvm::cl::init("swiftshader-unknown-unknown"));

  static llvm::cl::opt<std::string> clVulkanTargetEnv(
      "iree-vulkan-target-env",
      llvm::cl::desc(
          "Vulkan target environment as #vk.target_env attribute assembly"),
      llvm::cl::init(""));

  VulkanSPIRVTargetOptions targetOptions;
  targetOptions.codegenOptions.workgroupSize.assign(clWorkgroupSize.begin(),
                                                    clWorkgroupSize.end());
  targetOptions.codegenOptions.tileSizes.assign(clTileSizes.begin(),
                                                clTileSizes.end());
  targetOptions.codegenOptions.useWorkgroupMemory = clUseWorkgroupMemory;
  targetOptions.codegenOptions.useVectorPass = clUseVectorPass;
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

// Returns a list of entry point names matching the expected export ordinals.
static std::vector<std::string> populateEntryPointNames(
    spirv::ModuleOp spvModuleOp) {
  std::vector<std::string> entryPointNames;
  spvModuleOp.walk([&](spirv::EntryPointOp entryPointOp) {
    entryPointNames.push_back(std::string(entryPointOp.fn()));
  });
  return entryPointNames;
}

class VulkanSPIRVTargetBackend : public SPIRVTargetBackend {
 public:
  VulkanSPIRVTargetBackend(VulkanSPIRVTargetOptions options)
      : SPIRVTargetBackend(options.codegenOptions),
        options_(std::move(options)) {}

  // NOTE: we could vary these based on the options such as 'vulkan-v1.1'.
  std::string name() const override { return "vulkan_spirv"; }
  std::string filter_pattern() const override { return "vulkan*"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
                    Vulkan::VulkanDialect,
                    gpu::GPUDialect,
                    linalg::LinalgDialect,
                    scf::SCFDialect,
                    spirv::SPIRVDialect,
                    vector::VectorDialect>();
    // clang-format on
  }

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

  void declareTargetOps(IREE::Flow::ExecutableOp sourceOp,
                        IREE::HAL::ExecutableOp executableOp) override {
    spirv::TargetEnvAttr spvTargetEnv =
        getSPIRVTargetEnv(options_.vulkanTargetEnv, sourceOp.getContext());
    declareTargetOpsForEnv(sourceOp, executableOp, spvTargetEnv);
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
