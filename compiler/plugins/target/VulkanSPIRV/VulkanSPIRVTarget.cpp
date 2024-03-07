// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanDialect.h"
#include "iree/compiler/Dialect/Vulkan/Utils/TargetEnvironment.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "iree/schemas/spirv_executable_def_builder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Linking/ModuleCombiner.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/SPIRV/Serialization.h"

namespace mlir::iree_compiler::IREE::HAL {

namespace {
struct VulkanSPIRVTargetOptions {
  std::string targetTriple = "";
  std::string targetEnv = "";
  bool indirectBindings = false;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("VulkanSPIRV HAL Target");
    binder.opt<std::string>(
        "iree-vulkan-target-triple", targetTriple,
        llvm::cl::desc(
            "Vulkan target triple controlling the SPIR-V environment."));
    binder.opt<std::string>(
        "iree-vulkan-target-env", targetEnv,
        llvm::cl::desc(
            "Vulkan target environment as #vk.target_env attribute assembly."));
    binder.opt<bool>(
        "iree-vulkan-experimental-indirect-bindings", indirectBindings,
        llvm::cl::desc(
            "Force indirect bindings for all generated dispatches."));
  }
};
} // namespace

// Returns the Vulkan target environment for conversion.
static spirv::TargetEnvAttr
getSPIRVTargetEnv(const std::string &vulkanTargetTripleOrEnv,
                  MLIRContext *context) {
  if (!vulkanTargetTripleOrEnv.empty()) {
    if (vulkanTargetTripleOrEnv[0] != '#') {
      // Parse target triple.
      return convertTargetEnv(
          Vulkan::getTargetEnvForTriple(context, vulkanTargetTripleOrEnv));
    }

    // Parse `#vk.target_env<...` attribute assembly.
    if (auto attr = parseAttribute(vulkanTargetTripleOrEnv, context)) {
      if (auto vkTargetEnv = llvm::dyn_cast<Vulkan::TargetEnvAttr>(attr)) {
        return convertTargetEnv(vkTargetEnv);
      }
    }
    emitError(Builder(context).getUnknownLoc())
        << "cannot parse vulkan target environment as #vk.target_env "
           "attribute: '"
        << vulkanTargetTripleOrEnv << "'";
  }
  return {};
}

// TODO: VulkanOptions for choosing the Vulkan version and extensions/features.
class VulkanTargetDevice : public TargetDevice {
public:
  VulkanTargetDevice(const VulkanSPIRVTargetOptions &options)
      : options_(options) {}

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    auto configAttr = b.getDictionaryAttr(configItems);

    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("vulkan-spirv")
        ->getDefaultExecutableTargets(context, "vulkan", configAttr,
                                      executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("vulkan"),
                                            configAttr, executableTargetAttrs);
  }

private:
  const VulkanSPIRVTargetOptions &options_;
};

class VulkanSPIRVTargetBackend : public TargetBackend {
public:
  VulkanSPIRVTargetBackend(const VulkanSPIRVTargetOptions &options)
      : options_(options) {}

  std::string getLegacyDefaultDeviceID() const override { return "vulkan"; }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    std::string targetTripleOrEnv;
    if (!options_.targetEnv.empty()) {
      // TODO(scotttodd): assert if triple is set too? (mutually exclusive)
      targetTripleOrEnv = options_.targetEnv;
    } else if (!options_.targetTriple.empty()) {
      targetTripleOrEnv = options_.targetTriple;
    } else {
      targetTripleOrEnv = "unknown-unknown-unknown";
    }

    executableTargetAttrs.push_back(getExecutableTarget(
        context, getSPIRVTargetEnv(targetTripleOrEnv, context),
        options_.indirectBindings));
  }

  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(MLIRContext *context, spirv::TargetEnvAttr targetEnv,
                      bool indirectBindings) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;
    auto addConfig = [&](StringRef name, Attribute value) {
      configItems.emplace_back(b.getStringAttr(name), value);
    };

    addConfig(spirv::getTargetEnvAttrName(), targetEnv);
    if (indirectBindings) {
      addConfig("hal.bindings.indirect", b.getUnitAttr());
    }

    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("vulkan-spirv"),
        indirectBindings ? b.getStringAttr("vulkan-spirv-fb-ptr")
                         : b.getStringAttr("vulkan-spirv-fb"),
        b.getDictionaryAttr(configItems));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect, Vulkan::VulkanDialect,
                    spirv::SPIRVDialect, gpu::GPUDialect>();
  }

  void buildConfigurationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                      OpPassManager &passManager) override {
    // For now we disable translation if the variant has external object files.
    // We could instead perform linking with those objects (if they're .spv
    // files we could use spirv-link or import them into MLIR and merge here).
    if (variantOp.isExternal())
      return;

    buildSPIRVCodegenConfigurationPassPipeline(passManager);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpPassManager &passManager) override {
    // For now we disable translation if the variant has external object files.
    // We could instead perform linking with those objects (if they're .spv
    // files we could use spirv-link or import them into MLIR and merge here).
    if (variantOp.isExternal())
      return;

    buildSPIRVCodegenPassPipeline(passManager, /*enableFastMath=*/false);
  }

  void buildLinkingPassPipeline(OpPassManager &passManager) override {
    buildSPIRVLinkingPassPipeline(passManager);
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    // Today we special-case external variants but in the future we could allow
    // for a linking approach allowing both code generation and external .spv
    // files to be combined together.
    if (variantOp.isExternal()) {
      return serializeExternalExecutable(options, variantOp, executableBuilder);
    }

    ModuleOp innerModuleOp = variantOp.getInnerModule();
    auto spirvModuleOps = innerModuleOp.getOps<spirv::ModuleOp>();
    if (spirvModuleOps.empty()) {
      return variantOp.emitError() << "should contain some spirv.module ops";
    }

    DenseMap<StringRef, uint64_t> entryPointOrdinals;

    SmallVector<IREE::HAL::ExecutableExportOp> exportOps =
        llvm::to_vector(variantOp.getOps<IREE::HAL::ExecutableExportOp>());
    for (auto exportOp : exportOps) {
      uint64_t ordinal = 0;
      if (std::optional<APInt> optionalOrdinal = exportOp.getOrdinal()) {
        ordinal = optionalOrdinal->getZExtValue();
      } else {
        // For executables with only one entry point, linking doesn't kick in at
        // all. So the ordinal can be missing for this case.
        if (!llvm::hasSingleElement(exportOps)) {
          return exportOp.emitError() << "should have ordinal attribute";
        }
      }
      entryPointOrdinals[exportOp.getSymName()] = ordinal;
    }
    uint64_t ordinalCount = entryPointOrdinals.size();

    FlatbufferBuilder builder;
    iree_hal_spirv_ExecutableDef_start_as_root(builder);

    // The list of shader modules.
    SmallVector<iree_hal_spirv_ShaderModuleDef_ref_t> shaderModuleRefs;

    // Per entry-point data.
    // Note that the following vectors should all be of the same size and
    // element at index #i is for entry point with ordinal #i!
    SmallVector<StringRef> entryPointNames;
    SmallVector<uint32_t> subgroupSizes;
    SmallVector<uint32_t> shaderModuleIndices;
    SmallVector<iree_hal_spirv_FileLineLocDef_ref_t> sourceLocationRefs;
    entryPointNames.resize(ordinalCount);
    subgroupSizes.resize(ordinalCount);
    shaderModuleIndices.resize(ordinalCount);

    bool hasAnySubgroupSizes = false;

    // Iterate over all spirv.module ops and encode them into the FlatBuffer
    // data structure.
    for (spirv::ModuleOp spvModuleOp : spirvModuleOps) {
      // Currently the spirv.module op should only have one entry point. Get it.
      auto spirvEntryPoints = spvModuleOp.getOps<spirv::EntryPointOp>();
      if (!llvm::hasSingleElement(spirvEntryPoints)) {
        return spvModuleOp.emitError()
               << "expected to contain exactly one entry point";
      }
      spirv::EntryPointOp spvEntryPoint = *spirvEntryPoints.begin();
      uint64_t ordinal = entryPointOrdinals.at(spvEntryPoint.getFn());

      if (!options.dumpIntermediatesPath.empty()) {
        std::string assembly;
        llvm::raw_string_ostream os(assembly);
        spvModuleOp.print(os, OpPrintingFlags().useLocalScope());
        dumpDataToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                       spvEntryPoint.getFn(), ".spirv.mlir", assembly);
      }

      // Serialize the spirv::ModuleOp into the binary blob.
      SmallVector<uint32_t, 0> spvBinary;
      if (failed(spirv::serialize(spvModuleOp, spvBinary)) ||
          spvBinary.empty()) {
        return spvModuleOp.emitError() << "failed to serialize";
      }
      if (!options.dumpBinariesPath.empty()) {
        dumpDataToPath<uint32_t>(options.dumpBinariesPath, options.dumpBaseName,
                                 spvEntryPoint.getFn(), ".spv", spvBinary);
      }
      auto spvCodeRef = flatbuffers_uint32_vec_create(builder, spvBinary.data(),
                                                      spvBinary.size());
      shaderModuleIndices[ordinal] = shaderModuleRefs.size();
      shaderModuleRefs.push_back(
          iree_hal_spirv_ShaderModuleDef_create(builder, spvCodeRef));

      // The IREE runtime uses ordinals instead of names. We need to attach the
      // entry point name for VkShaderModuleCreateInfo.
      entryPointNames[ordinal] = spvEntryPoint.getFn();

      // If there are subgroup size requests, we need to pick up too.
      auto fn = spvModuleOp.lookupSymbol<spirv::FuncOp>(spvEntryPoint.getFn());
      auto abi = fn->getAttrOfType<spirv::EntryPointABIAttr>(
          spirv::getEntryPointABIAttrName());
      if (abi && abi.getSubgroupSize()) {
        subgroupSizes[ordinal] = *abi.getSubgroupSize();
        hasAnySubgroupSizes = true;
      } else {
        subgroupSizes[ordinal] = 0;
      }

      // Optional source location information for debugging/profiling.
      if (options.debugLevel >= 1) {
        if (auto loc = findFirstFileLoc(spvEntryPoint.getLoc())) {
          // We only ever resize to the maximum -- so all previous data will be
          // kept as-is.
          sourceLocationRefs.resize(ordinalCount);
          auto filenameRef = builder.createString(loc->getFilename());
          sourceLocationRefs[ordinal] = iree_hal_spirv_FileLineLocDef_create(
              builder, filenameRef, loc->getLine());
        }
      };
    }

    // Add top-level executable fields following their order of definition.
    auto entryPointsRef = builder.createStringVec(entryPointNames);
    flatbuffers_int32_vec_ref_t subgroupSizesRef =
        hasAnySubgroupSizes ? builder.createInt32Vec(subgroupSizes) : 0;
    flatbuffers_int32_vec_ref_t shaderModuleIndicesRef =
        builder.createInt32Vec(shaderModuleIndices);
    iree_hal_spirv_ExecutableDef_entry_points_add(builder, entryPointsRef);
    if (subgroupSizesRef) {
      iree_hal_spirv_ExecutableDef_subgroup_sizes_add(builder,
                                                      subgroupSizesRef);
    }
    iree_hal_spirv_ExecutableDef_shader_module_indices_add(
        builder, shaderModuleIndicesRef);
    auto shaderModulesRef =
        builder.createOffsetVecDestructive(shaderModuleRefs);
    iree_hal_spirv_ExecutableDef_shader_modules_add(builder, shaderModulesRef);
    if (!sourceLocationRefs.empty()) {
      auto sourceLocationsRef =
          builder.createOffsetVecDestructive(sourceLocationRefs);
      iree_hal_spirv_ExecutableDef_source_locations_add(builder,
                                                        sourceLocationsRef);
    }

    iree_hal_spirv_ExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getBufferAttr(executableBuilder.getContext()));
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();
  }

  LogicalResult
  serializeExternalExecutable(const SerializationOptions &options,
                              IREE::HAL::ExecutableVariantOp variantOp,
                              OpBuilder &executableBuilder) {
    if (!variantOp.getObjects().has_value()) {
      return variantOp.emitOpError()
             << "no objects defined for external variant";
    } else if (variantOp.getObjects()->getValue().size() != 1) {
      // For now we assume there will be exactly one object file.
      // TODO(#7824): support multiple .spv files in a single flatbuffer archive
      // so that we can combine executables.
      return variantOp.emitOpError() << "only one object reference is "
                                        "supported for external variants";
    }

    // Take exported names verbatim for passing into VkShaderModuleCreateInfo.
    SmallVector<StringRef, 8> entryPointNames;
    for (auto exportOp : variantOp.getExportOps()) {
      entryPointNames.emplace_back(exportOp.getSymName());
    }
    // We only have one object file for now. So all entry points have shader
    // module index 0.
    SmallVector<uint32_t, 8> shaderModuleIndices(entryPointNames.size(), 0);

    // Load .spv object file.
    auto objectAttr = llvm::cast<IREE::HAL::ExecutableObjectAttr>(
        variantOp.getObjects()->getValue().front());
    std::string spvBinary;
    if (auto data = objectAttr.loadData()) {
      spvBinary = data.value();
    } else {
      return variantOp.emitOpError()
             << "object file could not be loaded: " << objectAttr;
    }
    if (spvBinary.size() % 4 != 0) {
      return variantOp.emitOpError()
             << "object file is not 4-byte aligned as expected for SPIR-V";
    }

    FlatbufferBuilder builder;
    iree_hal_spirv_ExecutableDef_start_as_root(builder);

    auto spvCodeRef = flatbuffers_uint32_vec_create(
        builder, reinterpret_cast<const uint32_t *>(spvBinary.data()),
        spvBinary.size() / sizeof(uint32_t));
    SmallVector<iree_hal_spirv_ShaderModuleDef_ref_t> shaderModuleRefs;
    shaderModuleRefs.push_back(
        iree_hal_spirv_ShaderModuleDef_create(builder, spvCodeRef));

    // Add top-level executable fields following their order of definition.
    auto entryPointsRef = builder.createStringVec(entryPointNames);
    auto shaderModuleIndicesRef = builder.createInt32Vec(shaderModuleIndices);
    iree_hal_spirv_ExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_hal_spirv_ExecutableDef_shader_module_indices_add(
        builder, shaderModuleIndicesRef);
    auto shaderModulesRef =
        builder.createOffsetVecDestructive(shaderModuleRefs);
    iree_hal_spirv_ExecutableDef_shader_modules_add(builder, shaderModulesRef);

    iree_hal_spirv_ExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getBufferAttr(executableBuilder.getContext()));
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();
  }

private:
  const VulkanSPIRVTargetOptions &options_;
};

namespace {
struct VulkanSPIRVSession
    : public PluginSession<VulkanSPIRVSession, VulkanSPIRVTargetOptions,
                           PluginActivationPolicy::DefaultActivated> {
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) {
    // #hal.device.target<"vulkan", ...
    targets.add("vulkan", [&]() {
      return std::make_shared<VulkanTargetDevice>(options);
    });
  }
  void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) {
    // #hal.executable.target<"vulkan-spirv", ...
    targets.add("vulkan-spirv", [&]() {
      return std::make_shared<VulkanSPIRVTargetBackend>(options);
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL

extern "C" bool iree_register_compiler_plugin_hal_target_vulkan_spirv(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::iree_compiler::IREE::HAL::VulkanSPIRVSession>(
      "hal_target_vulkan_spirv");
  return true;
}

IREE_DEFINE_COMPILER_OPTION_FLAGS(
    mlir::iree_compiler::IREE::HAL::VulkanSPIRVTargetOptions);
