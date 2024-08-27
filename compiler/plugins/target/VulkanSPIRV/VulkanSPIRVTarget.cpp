// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Utils/ExecutableDebugInfoUtils.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "iree/schemas/vulkan_executable_def_builder.h"
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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Target/SPIRV/Serialization.h"

namespace mlir::iree_compiler::IREE::HAL {

namespace {
struct VulkanSPIRVTargetOptions {
  // Use vp_android_baseline_2022 profile as the default target--it's a good
  // lowest common denominator to guarantee the generated SPIR-V is widely
  // accepted for now. Eventually we want to use a list for multi-targeting.
  std::string target = "vp_android_baseline_2022";
  bool indirectBindings = false;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("VulkanSPIRV HAL Target");
    binder.opt<std::string>(
        "iree-vulkan-target", target,
        llvm::cl::desc(
            "Vulkan target controlling the SPIR-V environment. Given the wide "
            "support of Vulkan, this option supports a few schemes: 1) LLVM "
            "CodeGen backend style: e.g., 'gfx*' for AMD GPUs and 'sm_*' for "
            "NVIDIA GPUs; 2) architecture code name style: e.g., "
            "'rdna3'/'valhall4'/'ampere'/'adreno' for AMD/ARM/NVIDIA/Qualcomm "
            "GPUs; 3) product name style: 'rx7900xtx'/'rtx4090' for AMD/NVIDIA "
            "GPUs. See "
            "https://iree.dev/guides/deployment-configurations/gpu-vulkan/ for "
            "more details."));
    binder.opt<bool>(
        "iree-vulkan-experimental-indirect-bindings", indirectBindings,
        llvm::cl::desc(
            "Force indirect bindings for all generated dispatches."));
  }
};
} // namespace

using DescriptorSetLayout = std::pair<unsigned, ArrayRef<PipelineBindingAttr>>;

static std::tuple<iree_hal_vulkan_DescriptorSetLayoutDef_vec_ref_t,
                  iree_hal_vulkan_PipelineLayoutDef_vec_ref_t,
                  DenseMap<IREE::HAL::PipelineLayoutAttr, uint32_t>>
createPipelineLayoutDefs(ArrayRef<IREE::HAL::ExecutableExportOp> exportOps,
                         FlatbufferBuilder &fbb) {
  DenseMap<DescriptorSetLayout, size_t> descriptorSetLayoutMap;
  DenseMap<IREE::HAL::PipelineLayoutAttr, uint32_t> pipelineLayoutMap;
  SmallVector<iree_hal_vulkan_DescriptorSetLayoutDef_ref_t>
      descriptorSetLayoutRefs;
  SmallVector<iree_hal_vulkan_PipelineLayoutDef_ref_t> pipelineLayoutRefs;
  for (auto exportOp : exportOps) {
    auto pipelineLayoutAttr = exportOp.getLayout();
    if (pipelineLayoutMap.contains(pipelineLayoutAttr)) {
      continue; // already present
    }

    // Currently only one descriptor set on the compiler side. We could
    // partition it by binding type (direct vs indirect, etc).
    SmallVector<uint32_t> descriptorSetLayoutOrdinals;
    auto descriptorSetLayout =
        DescriptorSetLayout(0, pipelineLayoutAttr.getBindings());
    auto it = descriptorSetLayoutMap.find(descriptorSetLayout);
    if (it != descriptorSetLayoutMap.end()) {
      descriptorSetLayoutOrdinals.push_back(it->second);
    } else {
      SmallVector<iree_hal_vulkan_DescriptorSetLayoutBindingDef_ref_t>
          bindingRefs;
      for (auto [i, bindingAttr] :
           llvm::enumerate(pipelineLayoutAttr.getBindings())) {
        uint32_t ordinal = static_cast<uint32_t>(i);
        iree_hal_vulkan_VkDescriptorType_enum_t descriptorType = 0;
        switch (bindingAttr.getType()) {
        case IREE::HAL::DescriptorType::UniformBuffer:
          descriptorType = iree_hal_vulkan_VkDescriptorType_UNIFORM_BUFFER;
          break;
        case IREE::HAL::DescriptorType::StorageBuffer:
          descriptorType = iree_hal_vulkan_VkDescriptorType_STORAGE_BUFFER;
          break;
        }
        uint32_t descriptorCount = 1;
        uint32_t stageFlags = 0x00000020u; // VK_SHADER_STAGE_COMPUTE_BIT
        bindingRefs.push_back(
            iree_hal_vulkan_DescriptorSetLayoutBindingDef_create(
                fbb, ordinal, descriptorType, descriptorCount, stageFlags));
      }
      auto bindingsRef = fbb.createOffsetVecDestructive(bindingRefs);

      descriptorSetLayoutOrdinals.push_back(descriptorSetLayoutRefs.size());
      descriptorSetLayoutMap[descriptorSetLayout] =
          descriptorSetLayoutRefs.size();
      descriptorSetLayoutRefs.push_back(
          iree_hal_vulkan_DescriptorSetLayoutDef_create(fbb, bindingsRef));
    }
    auto descriptorSetLayoutOrdinalsRef =
        fbb.createInt32Vec(descriptorSetLayoutOrdinals);

    iree_hal_vulkan_PushConstantRange_vec_ref_t pushConstantRangesRef = 0;
    if (int64_t pushConstantCount = pipelineLayoutAttr.getConstants()) {
      SmallVector<iree_hal_vulkan_PushConstantRange> pushConstantRanges;
      iree_hal_vulkan_PushConstantRange range0;
      range0.stage_flags = 0x00000020u; // VK_SHADER_STAGE_COMPUTE_BIT
      range0.offset = 0;
      range0.size = pushConstantCount * sizeof(uint32_t);
      pushConstantRanges.push_back(range0);
      pushConstantRangesRef = iree_hal_vulkan_PushConstantRange_vec_create(
          fbb, pushConstantRanges.data(), pushConstantRanges.size());
    }

    pipelineLayoutMap[pipelineLayoutAttr] =
        static_cast<uint32_t>(pipelineLayoutRefs.size());
    iree_hal_vulkan_PipelineLayoutDef_start(fbb);
    iree_hal_vulkan_PipelineLayoutDef_descriptor_set_layout_ordinals_add(
        fbb, descriptorSetLayoutOrdinalsRef);
    if (pushConstantRangesRef) {
      iree_hal_vulkan_PipelineLayoutDef_push_constant_ranges_add(
          fbb, pushConstantRangesRef);
    }
    pipelineLayoutRefs.push_back(iree_hal_vulkan_PipelineLayoutDef_end(fbb));
  }

  auto descriptorSetLayoutsRef =
      fbb.createOffsetVecDestructive(descriptorSetLayoutRefs);
  auto pipelineLayoutsRef = fbb.createOffsetVecDestructive(pipelineLayoutRefs);
  return std::make_tuple(descriptorSetLayoutsRef, pipelineLayoutsRef,
                         pipelineLayoutMap);
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

    SmallVector<NamedAttribute> deviceConfigAttrs;
    auto deviceConfigAttr = b.getDictionaryAttr(deviceConfigAttrs);

    SmallVector<NamedAttribute> executableConfigAttrs;
    auto executableConfigAttr = b.getDictionaryAttr(executableConfigAttrs);

    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("vulkan-spirv")
        ->getDefaultExecutableTargets(context, "vulkan", executableConfigAttr,
                                      executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("vulkan"),
                                            deviceConfigAttr,
                                            executableTargetAttrs);
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
    executableTargetAttrs.push_back(
        getExecutableTarget(context, options_.indirectBindings));
  }

  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(MLIRContext *context, bool indirectBindings) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;
    auto addConfig = [&](StringRef name, Attribute value) {
      configItems.emplace_back(b.getStringAttr(name), value);
    };

    if (auto target = GPU::getVulkanTargetDetails(options_.target, context)) {
      addConfig("iree.gpu.target", target);
    } else {
      emitError(b.getUnknownLoc(), "Unknown Vulkan target '")
          << options_.target << "'";
      return nullptr;
    }

    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("vulkan-spirv"),
        indirectBindings ? b.getStringAttr("vulkan-spirv-fb-ptr")
                         : b.getStringAttr("vulkan-spirv-fb"),
        b.getDictionaryAttr(configItems));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect, spirv::SPIRVDialect,
                    gpu::GPUDialect, IREE::GPU::IREEGPUDialect>();
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

    // Create a list of executable exports (by ordinal) to the SPIR-V module and
    // entry point defining them.
    auto unsortedExportOps =
        llvm::to_vector(variantOp.getOps<IREE::HAL::ExecutableExportOp>());
    DenseMap<StringRef, std::tuple<IREE::HAL::ExecutableExportOp, uint64_t>>
        exportOrdinalMap;
    for (auto exportOp : variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
      uint64_t ordinal = 0;
      if (std::optional<APInt> optionalOrdinal = exportOp.getOrdinal()) {
        ordinal = optionalOrdinal->getZExtValue();
      } else {
        // For executables with only one entry point linking doesn't kick in at
        // all. So the ordinal can be missing for this case.
        if (!llvm::hasSingleElement(unsortedExportOps)) {
          return exportOp.emitError() << "should have ordinal attribute";
        }
      }
      exportOrdinalMap[exportOp.getSymName()] =
          std::make_tuple(exportOp, ordinal);
    }
    SmallVector<IREE::HAL::ExecutableExportOp> sortedExportOps;
    sortedExportOps.resize(unsortedExportOps.size());
    SmallVector<std::tuple<IREE::HAL::ExecutableExportOp, spirv::ModuleOp,
                           spirv::EntryPointOp>>
        exportOps;
    exportOps.resize(unsortedExportOps.size());
    for (spirv::ModuleOp spirvModuleOp : spirvModuleOps) {
      // Currently the spirv.module op should only have one entry point.
      auto spirvEntryPoints = spirvModuleOp.getOps<spirv::EntryPointOp>();
      if (!llvm::hasSingleElement(spirvEntryPoints)) {
        // TODO: support multiple entry points. We only need them here to get
        // the module name for dumping files.
        return spirvModuleOp.emitError()
               << "expected to contain exactly one entry point";
      }
      spirv::EntryPointOp spirvEntryPointOp = *spirvEntryPoints.begin();
      auto [exportOp, ordinal] = exportOrdinalMap.at(spirvEntryPointOp.getFn());
      sortedExportOps[ordinal] = exportOp;
      exportOps[ordinal] =
          std::make_tuple(exportOp, spirvModuleOp, spirvEntryPointOp);
    }

    FlatbufferBuilder builder;
    iree_hal_vulkan_ExecutableDef_start_as_root(builder);

    // Attach embedded source file contents.
    auto sourceFilesRef = createSourceFilesVec(
        options.debugLevel, variantOp.getSourcesAttr(), builder);

    // Generate optional per-export debug information.
    // May be empty if no debug information was requested.
    auto exportDebugInfos =
        createExportDefs(options.debugLevel, sortedExportOps, builder);

    // Create a list of all serialized SPIR-V modules.
    // TODO: unique the modules when each contains multiple entry points.
    DenseMap<spirv::EntryPointOp, uint32_t> entryPointToModuleMap;
    SmallVector<iree_hal_vulkan_ShaderModuleDef_ref_t> shaderModuleRefs;
    for (auto [exportOp, spirvModuleOp, spirvEntryPointOp] : exportOps) {
      if (!options.dumpIntermediatesPath.empty()) {
        std::string assembly;
        llvm::raw_string_ostream os(assembly);
        spirvModuleOp.print(os, OpPrintingFlags().useLocalScope());
        dumpDataToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                       spirvEntryPointOp.getFn(), ".spirv.mlir", assembly);
      }

      // Serialize the spirv::ModuleOp into the binary blob.
      SmallVector<uint32_t, 0> spirvBinary;
      if (failed(spirv::serialize(spirvModuleOp, spirvBinary)) ||
          spirvBinary.empty()) {
        return spirvModuleOp.emitError() << "failed to serialize";
      }
      if (!options.dumpBinariesPath.empty()) {
        dumpDataToPath<uint32_t>(options.dumpBinariesPath, options.dumpBaseName,
                                 spirvEntryPointOp.getFn(), ".spv",
                                 spirvBinary);
      }
      auto spirvCodeRef = flatbuffers_uint32_vec_create(
          builder, spirvBinary.data(), spirvBinary.size());
      entryPointToModuleMap[spirvEntryPointOp] =
          static_cast<uint32_t>(shaderModuleRefs.size());
      shaderModuleRefs.push_back(
          iree_hal_vulkan_ShaderModuleDef_create(builder, spirvCodeRef));
    }
    auto shaderModulesRef =
        builder.createOffsetVecDestructive(shaderModuleRefs);

    // Create unique descriptor and pipeline layouts for each entry point.
    auto [descriptorSetLayoutsRef, pipelineLayoutsRef, pipelineLayoutMap] =
        createPipelineLayoutDefs(sortedExportOps, builder);

    // Create pipelines representing entry points.
    // Note that the element at index #i is for entry point with ordinal #i.
    SmallVector<iree_hal_vulkan_PipelineDef_ref_t> pipelineRefs;
    for (auto [exportOp, spirvModuleOp, spirvEntryPointOp] : exportOps) {
      int64_t ordinal = exportOp.getOrdinalAttr().getInt();

      uint32_t shaderModuleOrdinal =
          entryPointToModuleMap.at(spirvEntryPointOp);
      uint32_t pipelineLayoutOrdinal =
          pipelineLayoutMap.at(exportOp.getLayout());

      // Subgroup size requests are optional.
      auto spirvFuncOp =
          spirvModuleOp.lookupSymbol<spirv::FuncOp>(spirvEntryPointOp.getFn());
      auto abiAttr = spirvFuncOp->getAttrOfType<spirv::EntryPointABIAttr>(
          spirv::getEntryPointABIAttrName());
      uint32_t subgroupSize =
          abiAttr ? abiAttr.getSubgroupSize().value_or(0) : 0;

      auto entryPointRef = builder.createString(spirvEntryPointOp.getFn());
      iree_hal_vulkan_PipelineDef_start(builder);
      iree_hal_vulkan_PipelineDef_shader_module_ordinal_add(
          builder, shaderModuleOrdinal);
      iree_hal_vulkan_PipelineDef_entry_point_add(builder, entryPointRef);
      iree_hal_vulkan_PipelineDef_pipeline_layout_ordinal_add(
          builder, pipelineLayoutOrdinal);
      iree_hal_vulkan_PipelineDef_subgroup_size_add(builder, subgroupSize);
      iree_hal_vulkan_PipelineDef_debug_info_add(builder,
                                                 exportDebugInfos[ordinal]);
      pipelineRefs.push_back(iree_hal_vulkan_PipelineDef_end(builder));
    }
    auto pipelinesRef = builder.createOffsetVecDestructive(pipelineRefs);

    // Add top-level executable fields following their order of definition.
    iree_hal_vulkan_ExecutableDef_pipelines_add(builder, pipelinesRef);
    iree_hal_vulkan_ExecutableDef_descriptor_set_layouts_add(
        builder, descriptorSetLayoutsRef);
    iree_hal_vulkan_ExecutableDef_pipeline_layouts_add(builder,
                                                       pipelineLayoutsRef);
    iree_hal_vulkan_ExecutableDef_shader_modules_add(builder, shaderModulesRef);
    iree_hal_vulkan_ExecutableDef_source_files_add(builder, sourceFilesRef);

    iree_hal_vulkan_ExecutableDef_end_as_root(builder);

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

    // Load .spv object file.
    auto objectAttr = llvm::cast<IREE::HAL::ExecutableObjectAttr>(
        variantOp.getObjects()->getValue().front());
    std::string spirvBinary;
    if (auto data = objectAttr.loadData()) {
      spirvBinary = data.value();
    } else {
      return variantOp.emitOpError()
             << "object file could not be loaded: " << objectAttr;
    }
    if (spirvBinary.size() % 4 != 0) {
      return variantOp.emitOpError()
             << "object file is not 4-byte aligned as expected for SPIR-V";
    }

    FlatbufferBuilder builder;
    iree_hal_vulkan_ExecutableDef_start_as_root(builder);

    // Wrap and embed shader module binary.
    auto spirvCodeRef = flatbuffers_uint32_vec_create(
        builder, reinterpret_cast<const uint32_t *>(spirvBinary.data()),
        spirvBinary.size() / sizeof(uint32_t));
    SmallVector<iree_hal_vulkan_ShaderModuleDef_ref_t> shaderModuleRefs;
    shaderModuleRefs.push_back(
        iree_hal_vulkan_ShaderModuleDef_create(builder, spirvCodeRef));
    auto shaderModulesRef =
        builder.createOffsetVecDestructive(shaderModuleRefs);

    // Generate descriptor set and pipeline layouts from export ops.
    auto exportOps = llvm::to_vector(variantOp.getExportOps());
    auto [descriptorSetLayoutsRef, pipelineLayoutsRef, pipelineLayoutMap] =
        createPipelineLayoutDefs(exportOps, builder);

    // Create a pipeline for each export.
    SmallVector<iree_hal_vulkan_PipelineDef_ref_t> pipelineRefs;
    for (auto exportOp : exportOps) {
      uint32_t shaderModuleOrdinal = 0; // only one today
      uint32_t pipelineLayoutOrdinal =
          pipelineLayoutMap.at(exportOp.getLayout());

      // Subgroup size requests are optional.
      // TODO: support annotation on an attribute to allow users to specify.
      uint32_t subgroupSize = 0;

      auto entryPointRef = builder.createString(exportOp.getName());
      iree_hal_vulkan_PipelineDef_start(builder);
      iree_hal_vulkan_PipelineDef_shader_module_ordinal_add(
          builder, shaderModuleOrdinal);
      iree_hal_vulkan_PipelineDef_entry_point_add(builder, entryPointRef);
      iree_hal_vulkan_PipelineDef_pipeline_layout_ordinal_add(
          builder, pipelineLayoutOrdinal);
      iree_hal_vulkan_PipelineDef_subgroup_size_add(builder, subgroupSize);
      pipelineRefs.push_back(iree_hal_vulkan_PipelineDef_end(builder));
    }
    auto pipelinesRef = builder.createOffsetVecDestructive(pipelineRefs);

    // Add top-level executable fields following their order of definition.
    iree_hal_vulkan_ExecutableDef_pipelines_add(builder, pipelinesRef);
    iree_hal_vulkan_ExecutableDef_descriptor_set_layouts_add(
        builder, descriptorSetLayoutsRef);
    iree_hal_vulkan_ExecutableDef_pipeline_layouts_add(builder,
                                                       pipelineLayoutsRef);
    iree_hal_vulkan_ExecutableDef_shader_modules_add(builder, shaderModulesRef);

    iree_hal_vulkan_ExecutableDef_end_as_root(builder);

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
