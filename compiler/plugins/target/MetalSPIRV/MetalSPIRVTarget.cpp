// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/MetalSPIRV/MSLToMetalLib.h"
#include "compiler/plugins/target/MetalSPIRV/MetalTargetPlatform.h"
#include "compiler/plugins/target/MetalSPIRV/SPIRVToMSL.h"
#include "iree/compiler/Codegen/Common/Passes.h"
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

// TODO: MetalOptions for choosing the Metal version.
class MetalTargetDevice final : public TargetDevice {
public:
  MetalTargetDevice(const MetalSPIRVOptions & /*options*/) {}

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const final {
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
};

class MetalSPIRVTargetBackend final : public TargetBackend {
public:
  MetalSPIRVTargetBackend(const MetalSPIRVOptions &options)
      : options(options) {}

  std::string getLegacyDefaultDeviceID() const final { return "metal"; }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const final {
    executableTargetAttrs.push_back(getExecutableTarget(context));
  }

  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(MLIRContext *context) const {
    Builder b(context);
    SmallVector<NamedAttribute, 1> configItems;
    if (auto target = GPU::getMetalTargetDetails(context)) {
      addConfigGPUTarget(context, target, configItems);
    }

    return b.getAttr<IREE::HAL::ExecutableTargetAttr>(
        b.getStringAttr("metal-spirv"), b.getStringAttr("metal-msl-fb"),
        b.getDictionaryAttr(configItems));
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<gpu::GPUDialect, IREE::Codegen::IREECodegenDialect,
                    IREE::Flow::FlowDialect, spirv::SPIRVDialect,
                    IREE::GPU::IREEGPUDialect>();
  }

  void
  buildConfigurationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                 OpPassManager &passManager) final {
    buildCodegenConfigurationPreProcessingPassPipeline(passManager);
    buildSPIRVCodegenConfigurationPassPipeline(passManager.nest<ModuleOp>());
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                    OpPassManager &passManager) final {
    buildSPIRVCodegenPassPipeline(passManager.nest<ModuleOp>());
    buildCodegenTranslationPostProcessingPassPipeline(passManager);
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) final {
    if (variantOp.isExternal()) {
      return serializeExternalExecutable(serOptions, variantOp,
                                         executableBuilder);
    }

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

    // Collect the SPIR-V entry point function names. These are cross-compiled
    // to MSL and used to construct Metal pipeline states; they are the MSL
    // function names (which spirv-cross may rewrite to be MSL-legal) and are
    // distinct from the hal.executable.export symbol names the runtime uses to
    // look up exports by name.
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
    // Each library may provide multiple functions so we encode them
    // independently.
    SmallVector<MetalLibrary> libraries;
    SmallVector<MetalPipeline> pipelines;
    auto exportOps = llvm::to_vector_of<IREE::HAL::ExecutableExportOp>(
        variantOp.getExportOps());
    for (auto [i, shader, metallib, entryPoint, exportOp] :
         llvm::zip_equal(llvm::seq<uint32_t>(mslShaders.size()), mslShaders,
                         metallibs, mslEntryPointNames, exportOps)) {
      pipelines.push_back({exportOp,
                           entryPoint,
                           /*libraryOrdinal=*/i,
                           {
                               shader.threadgroupSize.x,
                               shader.threadgroupSize.y,
                               shader.threadgroupSize.z,
                           }});
      libraries.push_back({std::move(shader.source), std::move(metallib)});
    }
    return serializeLibraries(serOptions, variantOp, libraries, pipelines,
                              executableBuilder);
  }

  // Serializes an external variant, such as one produced by
  // hal.dispatch.extern, whose single object is a precompiled .metallib or an
  // MSL source file. Each hal.executable.export names a kernel function in the
  // object and must declare a workgroup_size, which becomes the Metal
  // threadgroup size.
  //
  // Kernels use the same binding ABI as generated ones: bindings in an
  // argument buffer at [[buffer(0)]] with [[id(N)]] being the binding ordinal,
  // and constants as an array of 32-bit values at [[buffer(3)]].
  LogicalResult
  serializeExternalExecutable(const SerializationOptions &serOptions,
                              IREE::HAL::ExecutableVariantOp variantOp,
                              OpBuilder &executableBuilder) {
    std::optional<ArrayAttr> objects = variantOp.getObjects();
    if (!objects || objects->size() != 1) {
      return variantOp.emitOpError()
             << "external Metal variants need exactly one object (a "
                ".metallib or an MSL source file)";
    }
    auto objectAttr = cast<IREE::HAL::ExecutableObjectAttr>((*objects)[0]);
    std::optional<std::string> data = objectAttr.loadData();
    if (!data) {
      return variantOp.emitOpError()
             << "object file could not be loaded: " << objectAttr;
    }

    // A metallib starts with the "MTLB" magic; anything else is MSL source,
    // compiled here like generated MSL unless metallib compilation is off.
    std::unique_ptr<llvm::MemoryBuffer> metallib;
    std::string source;
    if (StringRef(*data).starts_with("MTLB")) {
      metallib = llvm::MemoryBuffer::getMemBufferCopy(*data);
    } else {
      source = std::move(*data);
      auto hostTriple = llvm::Triple(llvm::sys::getProcessTriple());
      if (options.compileToMetalLib && hostTriple.isMacOSX()) {
        metallib = compileMSLToMetalLib(options.targetPlatform, source,
                                        variantOp.getName());
        if (!metallib) {
          return variantOp.emitOpError()
                 << "failed to compile MSL object to a metallib: "
                 << objectAttr;
        }
      }
    }

    SmallVector<MetalLibrary> libraries;
    libraries.push_back({std::move(source), std::move(metallib)});

    SmallVector<MetalPipeline> pipelines;
    for (auto exportOp : variantOp.getExportOps()) {
      std::optional<ArrayAttr> workgroupSize = exportOp.getWorkgroupSize();
      if (!workgroupSize || workgroupSize->size() != 3) {
        return exportOp.emitOpError()
               << "external Metal exports need a 3D workgroup_size";
      }
      auto dim = [&](unsigned d) {
        return static_cast<uint32_t>(
            cast<IntegerAttr>((*workgroupSize)[d]).getInt());
      };
      // The export symbol is the kernel function name in the object.
      pipelines.push_back({exportOp,
                           exportOp.getName().str(),
                           /*libraryOrdinal=*/0,
                           {dim(0), dim(1), dim(2)}});
    }
    return serializeLibraries(serOptions, variantOp, libraries, pipelines,
                              executableBuilder);
  }

  // An MTLLibrary to embed: a metallib, MSL source or both.
  struct MetalLibrary {
    std::string source;
    std::unique_ptr<llvm::MemoryBuffer> metallib;
  };

  // A pipeline: an export and the kernel function implementing it.
  struct MetalPipeline {
    IREE::HAL::ExecutableExportOp exportOp;
    std::string entryPoint;
    uint32_t libraryOrdinal;
    iree_hal_metal_ThreadgroupSize_t threadgroupSize;
  };

  // Packs libraries and pipelines into a Metal executable FlatBuffer and adds
  // it to the target executable. Pipelines are placed at their export
  // ordinals, which the runtime looks them up by.
  LogicalResult serializeLibraries(const SerializationOptions &serOptions,
                                   IREE::HAL::ExecutableVariantOp variantOp,
                                   ArrayRef<MetalLibrary> libraries,
                                   ArrayRef<MetalPipeline> pipelines,
                                   OpBuilder &executableBuilder) {
    SmallVector<IREE::HAL::ExecutableExportOp> exportOps;
    SmallVector<bool> ordinalUsed(pipelines.size(), false);
    for (const MetalPipeline &pipeline : pipelines) {
      IREE::HAL::ExecutableExportOp exportOp = pipeline.exportOp;
      IntegerAttr ordinalAttr = exportOp.getOrdinalAttr();
      if (!ordinalAttr) {
        return exportOp.emitOpError()
               << "could not compile metal binary: export op is missing "
                  "ordinal";
      }
      int64_t ordinal = ordinalAttr.getInt();
      if (ordinal < 0 || ordinal >= static_cast<int64_t>(pipelines.size()) ||
          ordinalUsed[ordinal]) {
        return exportOp.emitOpError()
               << "could not compile metal binary: export ordinal " << ordinal
               << " is out of range or duplicated";
      }
      ordinalUsed[ordinal] = true;
      exportOps.push_back(exportOp);
    }

    FlatbufferBuilder builder;
    iree_hal_metal_ExecutableDef_start_as_root(builder);

    // Attach embedded source file contents.
    auto sourceFilesRef = createSourceFilesVec(
        serOptions.debugLevel, variantOp.getSourcesAttr(), builder);

    SmallVector<iree_hal_metal_LibraryDef_ref_t> libraryRefs;
    for (const MetalLibrary &library : libraries) {
      const bool embedSource = !library.source.empty() &&
                               (!library.metallib || serOptions.debugLevel > 1);
      iree_hal_metal_MSLSourceDef_ref_t sourceRef = 0;
      if (embedSource) {
        // TODO: pull this from an attribute?
        // https://developer.apple.com/documentation/metal/mtllanguageversion
        unsigned version = 196608; // MTLLanguageVersion3_0
        auto sourceStrRef = builder.createString(library.source);
        sourceRef =
            iree_hal_metal_MSLSourceDef_create(builder, version, sourceStrRef);
      }
      flatbuffers_string_ref_t metallibRef = 0;
      if (library.metallib) {
        metallibRef = flatbuffers_string_create(
            builder, library.metallib->getBufferStart(),
            library.metallib->getBufferSize());
      }
      iree_hal_metal_LibraryDef_start(builder);
      iree_hal_metal_LibraryDef_source_add(builder, sourceRef);
      iree_hal_metal_LibraryDef_metallib_add(builder, metallibRef);
      libraryRefs.push_back(iree_hal_metal_LibraryDef_end(builder));
    }
    auto librariesRef = builder.createOffsetVecDestructive(libraryRefs);

    // Generate optional per-export debug information, indexed by ordinal.
    // May be empty if no debug information was requested.
    auto exportDebugInfos =
        createExportDefs(serOptions.debugLevel, exportOps, builder);

    SmallVector<iree_hal_metal_PipelineDef_ref_t> pipelineRefs(pipelines.size(),
                                                               0);
    for (const MetalPipeline &pipeline : pipelines) {
      IREE::HAL::ExecutableExportOp exportOp = pipeline.exportOp;
      int64_t ordinal = exportOp.getOrdinalAttr().getInt();
      auto entryPointRef = builder.createString(pipeline.entryPoint);
      auto exportNameRef = builder.createString(exportOp.getName());

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
      iree_hal_metal_PipelineDef_library_ordinal_add(builder,
                                                     pipeline.libraryOrdinal);
      iree_hal_metal_PipelineDef_entry_point_add(builder, entryPointRef);
      iree_hal_metal_PipelineDef_name_add(builder, exportNameRef);
      iree_hal_metal_PipelineDef_threadgroup_size_add(
          builder, &pipeline.threadgroupSize);
      // TODO: embed additional metadata on threadgroup info if available.
      // iree_hal_metal_PipelineDef_max_threads_per_threadgroup_add(builder, 0);
      // iree_hal_metal_PipelineDef_threadgroup_size_aligned_add(builder,
      // false);
      iree_hal_metal_PipelineDef_constant_count_add(builder, constantCount);
      iree_hal_metal_PipelineDef_binding_flags_add(builder, bindingFlagsRef);
      iree_hal_metal_PipelineDef_debug_info_add(builder,
                                                exportDebugInfos[ordinal]);
      pipelineRefs[ordinal] = iree_hal_metal_PipelineDef_end(builder);
    }
    auto pipelinesRef = builder.createOffsetVecDestructive(pipelineRefs);

    iree_hal_metal_ExecutableDef_pipelines_add(builder, pipelinesRef);
    iree_hal_metal_ExecutableDef_libraries_add(builder, librariesRef);
    iree_hal_metal_ExecutableDef_source_files_add(builder, sourceFilesRef);

    iree_hal_metal_ExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    auto binaryOp = IREE::HAL::ExecutableBinaryOp::create(
        executableBuilder, variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getHeaderPrefixedBufferAttr(
            executableBuilder.getContext(),
            /*magic=*/iree_hal_metal_ExecutableDef_file_identifier,
            /*version=*/0));
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();
  }

private:
  const MetalSPIRVOptions &options;
};

struct MetalSPIRVSession final
    : PluginSession<MetalSPIRVSession, MetalSPIRVOptions,
                    PluginActivationPolicy::DefaultActivated> {
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) final {
    // #hal.device.target<"metal", ...
    targets.add("metal", [this]() {
      return std::make_shared<MetalTargetDevice>(options);
    });
  }
  void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) final {
    // #hal.executable.target<"metal-spirv", ...
    targets.add("metal-spirv", [this]() {
      return std::make_shared<MetalSPIRVTargetBackend>(options);
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL

IREE_DEFINE_COMPILER_OPTION_FLAGS(
    mlir::iree_compiler::IREE::HAL::MetalSPIRVOptions);

extern "C" bool iree_register_compiler_plugin_hal_target_metal_spirv(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::iree_compiler::IREE::HAL::MetalSPIRVSession>(
      "hal_target_metal_spirv");
  return true;
}
