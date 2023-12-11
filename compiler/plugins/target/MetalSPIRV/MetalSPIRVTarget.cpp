// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./MetalSPIRVTarget.h"

#include "./MSLToMetalLib.h"
#include "./MetalTargetPlatform.h"
#include "./SPIRVToMSL.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/metal_executable_def_builder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Target/SPIRV/Serialization.h"

namespace mlir::iree_compiler::IREE::HAL {

namespace {
struct MetalSPIRVOptions {
  MetalTargetPlatform targetPlatform = MetalTargetPlatform::macOS;
  bool compileToMetalLib = true;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("MetalSPIRV HAL Target");
    binder.opt<MetalTargetPlatform>(
        "iree-metal-target-platform", targetPlatform, llvm::cl::cat(category),
        llvm::cl::desc("Apple platform to target"),
        llvm::cl::values(
            clEnumValN(MetalTargetPlatform::macOS, "macos", "macOS platform"),
            clEnumValN(MetalTargetPlatform::iOS, "ios", "iOS platform"),
            clEnumValN(MetalTargetPlatform::iOSSimulator, "ios-simulator",
                       "iOS simulator platform")));
    binder.opt<bool>(
        "iree-metal-compile-to-metallib", compileToMetalLib,
        llvm::cl::cat(category),
        llvm::cl::desc("Compile to .metallib and embed in IREE deployable "
                       "flatbuffer if true; "
                       "otherwise stop at and embed MSL source code"));
  }
};
} // namespace

static spirv::TargetEnvAttr getMetalTargetEnv(MLIRContext *context) {
  using spirv::Capability;
  using spirv::Extension;

  // Capabilities and limits according to Metal 3 devices.
  const std::array<Extension, 4> extensions = {
      Extension::SPV_KHR_16bit_storage,
      Extension::SPV_KHR_8bit_storage,
      Extension::SPV_KHR_storage_buffer_storage_class,
      Extension::SPV_KHR_variable_pointers,
  };
  const std::array<Capability, 21> capabilities = {
      Capability::Shader,
      Capability::Int8,
      Capability::Int16,
      Capability::Int64,
      Capability::Float16,
      Capability::UniformAndStorageBuffer8BitAccess,
      Capability::StorageBuffer8BitAccess,
      Capability::StoragePushConstant8,
      Capability::StorageUniform16,
      Capability::StorageBuffer16BitAccess,
      Capability::StoragePushConstant16,
      Capability::GroupNonUniform,
      Capability::GroupNonUniformVote,
      Capability::GroupNonUniformArithmetic,
      Capability::GroupNonUniformBallot,
      Capability::GroupNonUniformShuffle,
      Capability::GroupNonUniformShuffleRelative,
      Capability::GroupNonUniformQuad,
      Capability::StoragePushConstant16,
      Capability::VariablePointers,
      Capability::VariablePointersStorageBuffer,
  };
  auto limits = spirv::ResourceLimitsAttr::get(
      context,
      /*max_compute_shared_memory_size=*/32768,
      /*max_compute_workgroup_invocations=*/1024,
      /*max_compute_workgroup_size=*/
      Builder(context).getI32ArrayAttr({1024, 1024, 1024}),
      /*subgroup_size=*/32,
      /*min_subgroup_size=*/std::nullopt,
      /*max_subgroup_size=*/std::nullopt,
      /*cooperative_matrix_properties_khr=*/ArrayAttr{},
      /*cooperative_matrix_properties_nv=*/ArrayAttr{});

  auto triple = spirv::VerCapExtAttr::get(spirv::Version::V_1_3, capabilities,
                                          extensions, context);
  // Further assuming Apple GPUs.
  return spirv::TargetEnvAttr::get(
      triple, limits, spirv::ClientAPI::Metal, spirv::Vendor::Apple,
      spirv::DeviceType::IntegratedGPU, spirv::TargetEnvAttr::kUnknownDeviceID);
}

class MetalSPIRVTargetBackend : public TargetBackend {
public:
  MetalSPIRVTargetBackend(const MetalSPIRVOptions &options)
      : options(options) {}

  // NOTE: we could vary this based on the options such as 'metal-v2'.
  std::string name() const override { return "metal"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, IREE::Codegen::IREECodegenDialect,
                    IREE::Flow::FlowDialect, spirv::SPIRVDialect>();
  }

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(b.getStringAttr("executable_targets"),
                             getExecutableTargets(context));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }

  void buildConfigurationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                      OpPassManager &passManager) override {
    // For now we disable configuration if the variant has external object
    // files.
    if (variantOp.isExternal())
      return;

    buildSPIRVCodegenConfigurationPassPipeline(passManager);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpPassManager &passManager) override {
    // For now we disable translation if the variant has external object files.
    // We could instead perform linking with those objects (if they're Metal
    // archives, etc).
    if (variantOp.isExternal())
      return;

    buildSPIRVCodegenPassPipeline(passManager, /*enableFastMath=*/false);
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    ModuleOp innerModuleOp = variantOp.getInnerModule();
    auto spvModuleOp = *innerModuleOp.getOps<spirv::ModuleOp>().begin();
    if (!serOptions.dumpIntermediatesPath.empty()) {
      std::string assembly;
      llvm::raw_string_ostream os(assembly);
      spvModuleOp.print(os, OpPrintingFlags().useLocalScope());
      dumpDataToPath(serOptions.dumpIntermediatesPath, serOptions.dumpBaseName,
                     variantOp.getName(), ".mlir", assembly);
    }

    // The runtime use ordinals instead of names but Metal requires function
    // names for constructing pipeline states. Get an ordered list of the entry
    // point names.
    SmallVector<StringRef, 8> spirvEntryPointNames;
    spvModuleOp.walk([&](spirv::EntryPointOp exportOp) {
      spirvEntryPointNames.push_back(exportOp.getFn());
    });

    // 1. Serialize the spirv::ModuleOp into binary format.
    SmallVector<uint32_t, 0> spvBinary;
    if (failed(spirv::serialize(spvModuleOp, spvBinary))) {
      return variantOp.emitError() << "failed to serialize spirv.module";
    }
    if (!serOptions.dumpIntermediatesPath.empty()) {
      dumpDataToPath<uint32_t>(serOptions.dumpIntermediatesPath,
                               serOptions.dumpBaseName, variantOp.getName(),
                               ".spv", spvBinary);
    }

    // 2. Cross compile SPIR-V to MSL source code.
    SmallVector<MetalShader, 2> mslShaders;
    SmallVector<std::string, 2> mslEntryPointNames;
    mslShaders.reserve(spirvEntryPointNames.size());
    mslEntryPointNames.reserve(spirvEntryPointNames.size());
    for (const auto &entryPoint : spirvEntryPointNames) {
      // We can use ArrayRef here given spvBinary reserves 0 bytes on stack.
      ArrayRef spvData(spvBinary.data(), spvBinary.size());
      std::optional<std::pair<MetalShader, std::string>> msl =
          crossCompileSPIRVToMSL(options.targetPlatform, spvData, entryPoint);
      if (!msl) {
        return variantOp.emitError()
               << "failed to cross compile SPIR-V to Metal shader";
      }
      mslShaders.push_back(std::move(msl->first));
      mslEntryPointNames.push_back(std::move(msl->second));
    }

    if (!serOptions.dumpBinariesPath.empty()) {
      for (auto shader : llvm::enumerate(mslShaders)) {
        dumpDataToPath(
            serOptions.dumpBinariesPath, serOptions.dumpBaseName,
            (variantOp.getName() + std::to_string(shader.index())).str(),
            ".metal", shader.value().source);
      }
    }

    // 3. Compile MSL to MTLLibrary.
    SmallVector<std::unique_ptr<llvm::MemoryBuffer>> metalLibs;
    if (options.compileToMetalLib) {
      // We need to use offline Metal shader compilers.
      // TODO(#14048): The toolchain can also exist on other platforms. Probe
      // the PATH instead.
      auto hostTriple = llvm::Triple(llvm::sys::getProcessTriple());
      if (hostTriple.isMacOSX()) {
        for (auto [shader, entryPoint] :
             llvm::zip(mslShaders, mslEntryPointNames)) {
          std::unique_ptr<llvm::MemoryBuffer> lib = compileMSLToMetalLib(
              options.targetPlatform, shader.source, entryPoint);
          if (!lib) {
            return variantOp.emitError()
                   << "failed to compile to MTLLibrary from MSL:\n\n"
                   << shader.source << "\n\n";
          }
          metalLibs.push_back(std::move(lib));
        }
      }
    }

    // 4. Pack the MTLLibrary and metadata into a FlatBuffer.
    FlatbufferBuilder builder;
    iree_hal_metal_ExecutableDef_start_as_root(builder);

    auto entryPointNamesRef = builder.createStringVec(mslEntryPointNames);
    iree_hal_metal_ExecutableDef_entry_points_add(builder, entryPointNamesRef);

    iree_hal_metal_ThreadgroupSize_vec_start(builder);
    for (auto &shader : mslShaders) {
      iree_hal_metal_ThreadgroupSize_vec_push_create(
          builder, shader.threadgroupSize.x, shader.threadgroupSize.y,
          shader.threadgroupSize.z);
    }
    auto threadgroupSizesRef = iree_hal_metal_ThreadgroupSize_vec_end(builder);
    iree_hal_metal_ExecutableDef_threadgroup_sizes_add(builder,
                                                       threadgroupSizesRef);

    if (metalLibs.empty()) {
      auto shaderSourcesRef = builder.createStringVec(
          llvm::map_range(mslShaders, [&](const MetalShader &shader) {
            return shader.source;
          }));
      iree_hal_metal_ExecutableDef_shader_sources_add(builder,
                                                      shaderSourcesRef);
    } else {
      auto refs = llvm::to_vector<8>(llvm::map_range(
          metalLibs, [&](const std::unique_ptr<llvm::MemoryBuffer> &buffer) {
            return flatbuffers_string_create(builder, buffer->getBufferStart(),
                                             buffer->getBufferSize());
          }));
      auto libsRef =
          flatbuffers_string_vec_create(builder, refs.data(), refs.size());
      iree_hal_metal_ExecutableDef_shader_libraries_add(builder, libsRef);
    }

    iree_hal_metal_ExecutableDef_end_as_root(builder);

    // 5. Add the binary data to the target executable.
    auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getBufferAttr(executableBuilder.getContext()));
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();
  }

private:
  ArrayAttr getExecutableTargets(MLIRContext *context) const {
    SmallVector<Attribute> targetAttrs;
    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    targetAttrs.push_back(
        getExecutableTarget(context, getMetalTargetEnv(context)));
    return ArrayAttr::get(context, targetAttrs);
  }

  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(MLIRContext *context,
                      spirv::TargetEnvAttr targetEnv) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(b.getStringAttr(spirv::getTargetEnvAttrName()),
                             targetEnv);

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("metal"), b.getStringAttr("metal-msl-fb"),
        configAttr);
  }

  const MetalSPIRVOptions &options;
};

struct MetalSPIRVSession
    : public PluginSession<MetalSPIRVSession, MetalSPIRVOptions,
                           PluginActivationPolicy::DefaultActivated> {
  void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) {
    auto backendFactory = [=]() {
      return std::make_shared<MetalSPIRVTargetBackend>(options);
    };
    // #hal.device.target<"metal", ...
    targets.add("metal", backendFactory);
    // #hal.executable.target<"metal-spirv", ...
    targets.add("metal-spirv", backendFactory);
  }
};

} // namespace mlir::iree_compiler::IREE::HAL

extern "C" bool iree_register_compiler_plugin_hal_target_metal_spirv(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::iree_compiler::IREE::HAL::MetalSPIRVSession>(
      "hal_target_metal_spirv");
  return true;
}

IREE_DEFINE_COMPILER_OPTION_FLAGS(
    mlir::iree_compiler::IREE::HAL::MetalSPIRVOptions);
