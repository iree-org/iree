// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/MetalSPIRV/MSLToMetalLib.h"
#include "compiler/plugins/target/MetalSPIRV/MetalTargetPlatform.h"
#include "compiler/plugins/target/MetalSPIRV/SPIRVToMSL.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Utils/ExecutableDebugInfoUtils.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/metal_executable_def_builder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
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

// TODO: MetalOptions for choosing the Metal version.
class MetalTargetDevice : public TargetDevice {
public:
  MetalTargetDevice(const MetalSPIRVOptions &options) : options(options) {}

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override {
    Builder b(context);
    auto configAttr = b.getDictionaryAttr({});

    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("metal-spirv")
        ->getDefaultExecutableTargets(context, "metal", configAttr,
                                      executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("metal"),
                                            configAttr, executableTargetAttrs);
  }

  LogicalResult
  setSharedUsageBits(const SetVector<IREE::HAL::DeviceTargetAttr> &targets,
                     IREE::HAL::BufferUsageBitfield &bufferUsage) const override {
    for (auto targetAttr : targets) {
      // if the target is metal, we dont need to add any usage bits
      if (targetAttr.getDeviceID().getValue() == "metal") {
        continue;
      }
      // if the target is local, we need to add the mapping persistent usage bit
      if (targetAttr.getDeviceID().getValue() == "local") {
        bufferUsage = bufferUsage | IREE::HAL::BufferUsageBitfield::MappingPersistent;
      }
      // interop with other targets is not supported yet
      return failure();
    }
    return success();
  }

private:
  const MetalSPIRVOptions &options;
};

class MetalSPIRVTargetBackend : public TargetBackend {
public:
  MetalSPIRVTargetBackend(const MetalSPIRVOptions &options)
      : options(options) {}

  std::string getLegacyDefaultDeviceID() const override { return "metal"; }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    executableTargetAttrs.push_back(getExecutableTarget(context));
  }

  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(MLIRContext *context) const {
    Builder b(context);
    SmallVector<NamedAttribute, 1> configItems;
    if (auto target = GPU::getMetalTargetDetails(context)) {
      configItems.emplace_back(kGPUTargetAttrName, target);
    }

    return b.getAttr<IREE::HAL::ExecutableTargetAttr>(
        b.getStringAttr("metal-spirv"), b.getStringAttr("metal-msl-fb"),
        b.getDictionaryAttr(configItems));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, IREE::Codegen::IREECodegenDialect,
                    IREE::Flow::FlowDialect, spirv::SPIRVDialect,
                    IREE::GPU::IREEGPUDialect>();
  }

  void
  buildConfigurationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                 OpPassManager &passManager) override {
    buildSPIRVCodegenConfigurationPassPipeline(passManager);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                    OpPassManager &passManager) override {
    buildSPIRVCodegenPassPipeline(passManager);
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    ModuleOp innerModuleOp = variantOp.getInnerModule();

    // TODO: rework this to compile all modules into the same metallib and
    // source the entry points from them. Or use a linking tool (metal-ar) to
    // link the compiled metallibs together. If we were not using spirv-cross
    // we'd never do it like this with one module per function.
    //
    // Currently this is _really_ bad because it doesn't support linking like
    // the Vulkan SPIR-V target: that allows multiple spirv::ModuleOps so we
    // at least only have a single HAL executable; this should all be reworked
    // to have multiple SPIR-V modules in a single executable and then even if
    // passing through spirv-cross independently should link the resulting
    // metallibs together.
    auto spvModuleOp = *innerModuleOp.getOps<spirv::ModuleOp>().begin();
    if (!serOptions.dumpIntermediatesPath.empty()) {
      std::string assembly;
      llvm::raw_string_ostream os(assembly);
      spvModuleOp.print(os, OpPrintingFlags().useLocalScope());
      dumpDataToPath(serOptions.dumpIntermediatesPath, serOptions.dumpBaseName,
                     variantOp.getName(), ".mlir", assembly);
    }

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

    // The runtime use ordinals instead of names but Metal requires function
    // names for constructing pipeline states. Get an ordered list of the entry
    // point names.
    SmallVector<StringRef, 8> spirvEntryPointNames;
    spvModuleOp.walk([&](spirv::EntryPointOp exportOp) {
      spirvEntryPointNames.push_back(exportOp.getFn());
    });

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
    SmallVector<std::unique_ptr<llvm::MemoryBuffer>> metallibs;
    metallibs.resize(mslShaders.size());
    if (options.compileToMetalLib) {
      // We need to use offline Metal shader compilers.
      // TODO(#14048): The toolchain can also exist on other platforms. Probe
      // the PATH instead.
      auto hostTriple = llvm::Triple(llvm::sys::getProcessTriple());
      if (hostTriple.isMacOSX()) {
        for (auto [i, shader, entryPoint] :
             llvm::zip_equal(llvm::seq(mslShaders.size()), mslShaders,
                             mslEntryPointNames)) {
          std::unique_ptr<llvm::MemoryBuffer> lib = compileMSLToMetalLib(
              options.targetPlatform, shader.source, entryPoint);
          if (!lib) {
            return variantOp.emitError()
                   << "failed to compile to MTLLibrary from MSL:\n\n"
                   << shader.source << "\n\n";
          }
          metallibs[i] = std::move(lib);
        }
      }
    }

    // 4. Pack the MTLLibrary and metadata into a FlatBuffer.
    FlatbufferBuilder builder;
    iree_hal_metal_ExecutableDef_start_as_root(builder);

    // Attach embedded source file contents.
    auto sourceFilesRef = createSourceFilesVec(
        serOptions.debugLevel, variantOp.getSourcesAttr(), builder);

    // Each library may provide multiple functions so we encode them
    // independently.
    SmallVector<iree_hal_metal_LibraryDef_ref_t> libraryRefs;
    for (auto [shader, metallib] : llvm::zip_equal(mslShaders, metallibs)) {
      const bool embedSource = !metallib || serOptions.debugLevel > 1;
      iree_hal_metal_MSLSourceDef_ref_t sourceRef = 0;
      if (embedSource) {
        // TODO: pull this from an attribute?
        // https://developer.apple.com/documentation/metal/mtllanguageversion
        unsigned version = 196608; // MTLLanguageVersion3_0
        auto sourceStrRef = builder.createString(shader.source);
        sourceRef =
            iree_hal_metal_MSLSourceDef_create(builder, version, sourceStrRef);
      }
      flatbuffers_string_ref_t metallibRef = 0;
      if (metallib) {
        metallibRef = flatbuffers_string_create(
            builder, metallib->getBufferStart(), metallib->getBufferSize());
      }
      iree_hal_metal_LibraryDef_start(builder);
      iree_hal_metal_LibraryDef_source_add(builder, sourceRef);
      iree_hal_metal_LibraryDef_metallib_add(builder, metallibRef);
      libraryRefs.push_back(iree_hal_metal_LibraryDef_end(builder));
    }
    auto librariesRef = builder.createOffsetVecDestructive(libraryRefs);

    // Generate optional per-export debug information.
    // May be empty if no debug information was requested.
    auto exportOps = llvm::to_vector_of<IREE::HAL::ExecutableExportOp>(
        variantOp.getExportOps());
    auto exportDebugInfos =
        createExportDefs(serOptions.debugLevel, exportOps, builder);

    SmallVector<iree_hal_metal_PipelineDef_ref_t> pipelineRefs;
    for (auto [i, shader, entryPoint, exportOp] :
         llvm::zip_equal(llvm::seq(mslShaders.size()), mslShaders,
                         mslEntryPointNames, exportOps)) {
      auto entryPointRef = builder.createString(entryPoint);

      iree_hal_metal_ThreadgroupSize_t threadgroupSize = {
          shader.threadgroupSize.x,
          shader.threadgroupSize.y,
          shader.threadgroupSize.z,
      };

      auto layoutAttr = exportOp.getLayoutAttr();
      uint32_t constantCount = static_cast<uint32_t>(layoutAttr.getConstants());
      SmallVector<iree_hal_metal_BindingBits_enum_t> bindingFlags;
      for (auto bindingAttr : layoutAttr.getBindings()) {
        iree_hal_metal_BindingBits_enum_t flags = 0;
        if (allEnumBitsSet(bindingAttr.getFlags(),
                           IREE::HAL::DescriptorFlags::ReadOnly)) {
          flags |= iree_hal_metal_BindingBits_IMMUTABLE;
        }
        bindingFlags.push_back(flags);
      }
      auto bindingFlagsRef = iree_hal_metal_BindingBits_vec_create(
          builder, bindingFlags.data(), bindingFlags.size());

      iree_hal_metal_PipelineDef_start(builder);
      iree_hal_metal_PipelineDef_library_ordinal_add(builder, i);
      iree_hal_metal_PipelineDef_entry_point_add(builder, entryPointRef);
      iree_hal_metal_PipelineDef_threadgroup_size_add(builder,
                                                      &threadgroupSize);
      // TODO: embed additional metadata on threadgroup info if available.
      // iree_hal_metal_PipelineDef_max_threads_per_threadgroup_add(builder, 0);
      // iree_hal_metal_PipelineDef_threadgroup_size_aligned_add(builder,
      // false);
      iree_hal_metal_PipelineDef_constant_count_add(builder, constantCount);
      iree_hal_metal_PipelineDef_binding_flags_add(builder, bindingFlagsRef);
      iree_hal_metal_PipelineDef_debug_info_add(builder, exportDebugInfos[i]);
      pipelineRefs.push_back(iree_hal_metal_PipelineDef_end(builder));
    }
    auto pipelinesRef = builder.createOffsetVecDestructive(pipelineRefs);

    iree_hal_metal_ExecutableDef_pipelines_add(builder, pipelinesRef);
    iree_hal_metal_ExecutableDef_libraries_add(builder, librariesRef);
    iree_hal_metal_ExecutableDef_source_files_add(builder, sourceFilesRef);

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
  const MetalSPIRVOptions &options;
};

struct MetalSPIRVSession
    : public PluginSession<MetalSPIRVSession, MetalSPIRVOptions,
                           PluginActivationPolicy::DefaultActivated> {
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) {
    // #hal.device.target<"metal", ...
    targets.add("metal",
                [=]() { return std::make_shared<MetalTargetDevice>(options); });
  }
  void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) {
    // #hal.executable.target<"metal-spirv", ...
    targets.add("metal-spirv", [=]() {
      return std::make_shared<MetalSPIRVTargetBackend>(options);
    });
  }
};

} // namespace mlir::iree_compiler::IREE::HAL

IREE_DEFINE_COMPILER_OPTION_FLAGS(
    mlir::iree_compiler::IREE::HAL::MetalSPIRVOptions);

extern "C" bool iree_register_compiler_plugin_hal_target_metal_spirv(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::iree_compiler::IREE::HAL::MetalSPIRVSession>(
      "hal_target_metal_spirv");
  return true;
}
