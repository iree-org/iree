// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/MetalSPIRV/MetalSPIRVTarget.h"

#include "iree/compiler/Dialect/HAL/Target/MetalSPIRV/SPIRVToMSL.h"
#include "iree/compiler/Dialect/HAL/Target/SPIRVCommon/SPIRVTarget.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/metal_executable_def_builder.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Target/SPIRV/Serialization.h"

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

class MetalSPIRVTargetBackend : public SPIRVTargetBackend {
 public:
  MetalSPIRVTargetBackend(MetalSPIRVTargetOptions options)
      : SPIRVTargetBackend(SPIRVCodegenOptions()),
        options_(std::move(options)) {}

  // NOTE: we could vary this based on the options such as 'metal-v2'.
  std::string name() const override { return "metal"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect>();
  }

  void declareVariantOps(IREE::Flow::ExecutableOp sourceOp,
                         IREE::HAL::ExecutableOp executableOp) override {
    declareVariantOpsForEnv(sourceOp, executableOp,
                            getMetalTargetEnv(sourceOp.getContext()));
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    ModuleOp innerModuleOp = variantOp.getInnerModule();
    auto spvModuleOp = *innerModuleOp.getOps<spirv::ModuleOp>().begin();

    // The runtime use ordinals instead of names but Metal requires function
    // names for constructing pipeline states. Get an ordered list of the entry
    // point names.
    SmallVector<StringRef, 8> entryPointNames;
    spvModuleOp.walk([&](spirv::EntryPointOp entryPointOp) {
      entryPointNames.push_back(entryPointOp.fn());
    });

    // 1. Serialize the spirv::ModuleOp into binary format.
    SmallVector<uint32_t, 0> spvBinary;
    if (failed(spirv::serialize(spvModuleOp, spvBinary))) {
      return variantOp.emitError() << "failed to serialize spv.module";
    }

    // 2. Cross compile SPIR-V to MSL source code.
    llvm::SmallVector<MetalShader, 2> mslShaders;
    for (const auto &entryPoint : entryPointNames) {
      llvm::Optional<MetalShader> mslShader = crossCompileSPIRVToMSL(
          // We can use ArrayRef here given spvBinary reserves 0 bytes on stack.
          llvm::makeArrayRef(spvBinary.data(), spvBinary.size()), entryPoint);
      if (!mslShader) {
        return variantOp.emitError()
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
    FlatbufferBuilder builder;
    iree_MetalExecutableDef_start_as_root(builder);

    auto shaderSourcesRef = builder.createStringVec(llvm::map_range(
        mslShaders, [&](const MetalShader &shader) { return shader.source; }));

    iree_MetalThreadgroupSize_vec_start(builder);
    for (auto &shader : mslShaders) {
      iree_MetalThreadgroupSize_vec_push_create(
          builder, shader.threadgroupSize.x, shader.threadgroupSize.y,
          shader.threadgroupSize.z);
    }
    auto threadgroupSizesRef = iree_MetalThreadgroupSize_vec_end(builder);

    auto entryPointNamesRef = builder.createStringVec(entryPointNames);

    iree_MetalExecutableDef_entry_points_add(builder, entryPointNamesRef);
    iree_MetalExecutableDef_threadgroup_sizes_add(builder, threadgroupSizesRef);
    iree_MetalExecutableDef_shader_sources_add(builder, shaderSourcesRef);
    iree_MetalExecutableDef_end_as_root(builder);

    // 5. Add the binary data to the target executable.
    auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.sym_name(),
        executableBuilder.getStringAttr("MTLE"),
        builder.getBufferAttr(executableBuilder.getContext()));
    binaryOp.mime_typeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();
  }

 protected:
  MetalSPIRVTargetOptions options_;
};

void registerMetalSPIRVTargetBackends(
    std::function<MetalSPIRVTargetOptions()> queryOptions) {
  auto backendFactory = [=]() {
    return std::make_unique<MetalSPIRVTargetBackend>(queryOptions());
  };
  static TargetBackendRegistration registration0("metal", backendFactory);
  static TargetBackendRegistration registration1("metal-spirv", backendFactory);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
