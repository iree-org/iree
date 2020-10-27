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

#include "iree/compiler/Dialect/HAL/Target/MetalSPIRV/MetalSPIRVTarget.h"

#include "flatbuffers/flatbuffers.h"
#include "iree/compiler/Conversion/Common/Attributes.h"
#include "iree/compiler/Dialect/HAL/Target/MetalSPIRV/SPIRVToMSL.h"
#include "iree/compiler/Dialect/HAL/Target/SPIRVCommon/SPIRVTarget.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/schemas/metal_executable_def_generated.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/Vector/VectorOps.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

MetalSPIRVTargetOptions getMetalSPIRVTargetOptionsFromFlags() {
  MetalSPIRVTargetOptions targetOptions;
  return targetOptions;
}

// TODO(antiagainst): provide a proper target environment for Metal.
static spirv::TargetEnvAttr getMetalTargetEnv(MLIRContext *context) {
  auto triple = spirv::VerCapExtAttr::get(
      spirv::Version::V_1_0, {spirv::Capability::Shader},
      {spirv::Extension::SPV_KHR_storage_buffer_storage_class}, context);
  return spirv::TargetEnvAttr::get(triple, spirv::Vendor::Unknown,
                                   spirv::DeviceType::Unknown,
                                   spirv::TargetEnvAttr::kUnknownDeviceID,
                                   spirv::getDefaultResourceLimits(context));
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

class MetalSPIRVTargetBackend : public SPIRVTargetBackend {
 public:
  MetalSPIRVTargetBackend(MetalSPIRVTargetOptions options)
      : SPIRVTargetBackend(SPIRVCodegenOptions()),
        options_(std::move(options)) {}

  // NOTE: we could vary this based on the options such as 'metal-v2'.
  std::string name() const override { return "metal_spirv"; }
  std::string filter_pattern() const override { return "metal*"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
                    gpu::GPUDialect,
                    linalg::LinalgDialect,
                    scf::SCFDialect,
                    spirv::SPIRVDialect,
                    vector::VectorDialect>();
    // clang-format on
  }

  void declareTargetOps(IREE::Flow::ExecutableOp sourceOp,
                        IREE::HAL::ExecutableOp executableOp) override {
    declareTargetOpsForEnv(sourceOp, executableOp,
                           getMetalTargetEnv(sourceOp.getContext()));
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder &executableBuilder) override {
    ModuleOp innerModuleOp = targetOp.getInnerModule();
    auto spvModuleOp = *innerModuleOp.getOps<spirv::ModuleOp>().begin();

    // The runtime use ordinals instead of names but Metal requires function
    // names for constructing pipeline states. Get an ordered list of the entry
    // point names.
    std::vector<std::string> entryPoints;
    if (auto scheduleAttr = innerModuleOp.getAttrOfType<ArrayAttr>(
            iree_compiler::getEntryPointScheduleAttrName())) {
      // We have multiple entry points in this module. Make sure the order
      // specified in the schedule attribute is respected.
      for (Attribute entryPoint : scheduleAttr) {
        entryPoints.emplace_back(
            entryPoint.cast<StringAttr>().getValue().str());
      }
    } else {
      entryPoints = populateEntryPointNames(spvModuleOp);
    }

    // 1. Serialize the spirv::ModuleOp into binary format.
    SmallVector<uint32_t, 0> spvBinary;
    if (failed(spirv::serialize(spvModuleOp, spvBinary))) {
      return targetOp.emitError() << "failed to serialize spv.module";
    }

    // 2. Cross compile SPIR-V to MSL source code.
    llvm::SmallVector<MetalShader, 2> mslShaders;
    for (const std::string &entryPoint : entryPoints) {
      llvm::Optional<MetalShader> mslShader = crossCompileSPIRVToMSL(
          // We can use ArrayRef here given spvBinary reserves 0 bytes on stack.
          llvm::makeArrayRef(spvBinary.data(), spvBinary.size()), entryPoint);
      if (!mslShader) {
        return targetOp.emitError()
               << "failed to cross compile SPIR-V to Metal shader";
      }
      mslShaders.push_back(std::move(*mslShader));
    }

    // 3. Compile MSL to MTLLibrary.
    // TODO(antiagainst): provide the option to compile the shaders into a
    // library and embed in the flatbuffer. Metal provides APIs for compiling
    // shader sources into a MTLLibrary at run-time, but does not provie
    // a way to serialize the generated MTLLibrary. The only way available is
    // to use command-line tools like `metal` and `metallib`. Likely we need
    // to invoke them in C++.

    // 4. Pack the MTLLibrary and metadata into a flatbuffer.
    iree::MetalExecutableDefT metalExecutableDef;
    metalExecutableDef.entry_points = entryPoints;
    for (auto &shader : mslShaders) {
      metalExecutableDef.shader_sources.push_back(std::move(shader.source));
      const auto &sizes = shader.threadgroupSize;
      metalExecutableDef.threadgroup_sizes.push_back(
          {sizes.x, sizes.y, sizes.z});
    }

    // Pack the executable definition and get the bytes with the proper header.
    // The header is used to verify the contents at runtime.
    ::flatbuffers::FlatBufferBuilder fbb;
    auto executableOffset =
        iree::MetalExecutableDef::Pack(fbb, &metalExecutableDef);
    iree::FinishMetalExecutableDefBuffer(fbb, executableOffset);
    std::vector<uint8_t> bytes;
    bytes.resize(fbb.GetSize());
    std::memcpy(bytes.data(), fbb.GetBufferPointer(), bytes.size());

    // 5. Add the binary data to the target executable.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(),
        static_cast<uint32_t>(IREE::HAL::ExecutableFormat::Metal),
        std::move(bytes));

    return success();
  }

 protected:
  MetalSPIRVTargetOptions options_;
};

void registerMetalSPIRVTargetBackends(
    std::function<MetalSPIRVTargetOptions()> queryOptions) {
  static TargetBackendRegistration registration("metal-spirv", [=]() {
    return std::make_unique<MetalSPIRVTargetBackend>(queryOptions());
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
