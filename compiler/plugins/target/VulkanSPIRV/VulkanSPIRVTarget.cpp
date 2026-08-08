// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Utils/ExecutableDebugInfoUtils.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "iree/schemas/vulkan_executable_def_builder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Linking/ModuleCombiner.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/SPIRV/Serialization.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {
constexpr unsigned kBdaDispatchRootDwordCount = 8;
constexpr uint32_t kBdaDispatchRootLength =
    kBdaDispatchRootDwordCount * sizeof(uint32_t);
constexpr uint32_t kBdaDispatchRootOffset = 0;
constexpr uint32_t kBdaDispatchConstantOffset = kBdaDispatchRootLength;

constexpr char kVulkanSpirvFlatbufferFormat[] = "vulkan-spirv-fb";
constexpr char kVulkanSpirvBdaFormat[] = "vulkan-spirv-bda-v1";

enum class VulkanDispatchAbi {
  Descriptors,
  Bda,
  All,
};

class PropagateExecutableTargetPass final
    : public PassWrapper<PropagateExecutableTargetPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PropagateExecutableTargetPass)

  explicit PropagateExecutableTargetPass(
      IREE::HAL::ExecutableTargetAttr targetAttr)
      : targetAttr(targetAttr) {}

  StringRef getArgument() const final {
    return "iree-vulkan-spirv-propagate-executable-target";
  }

  StringRef getDescription() const final {
    return "Propagates the selected Vulkan executable target into the inner "
           "module.";
  }

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    if (failed(setOrVerifyTarget(moduleOp))) {
      return signalPassFailure();
    }
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      if (failed(setOrVerifyTarget(funcOp))) {
        return signalPassFailure();
      }
    }
  }

private:
  LogicalResult setOrVerifyTarget(Operation *op) {
    Attribute existingAttr = op->getAttr(IREE::HAL::ExecutableTargetAttr::name);
    if (!existingAttr) {
      op->setAttr(IREE::HAL::ExecutableTargetAttr::name, targetAttr);
      return success();
    }
    if (existingAttr == targetAttr) {
      return success();
    }
    return op->emitError() << "conflicting Vulkan executable target attribute";
  }

  // Target selected by the HAL target backend for this translation pipeline.
  IREE::HAL::ExecutableTargetAttr targetAttr;
};

struct VulkanSPIRVTargetOptions {
  std::string target;
  std::string targetFeatures;
  VulkanDispatchAbi dispatchAbi = VulkanDispatchAbi::Descriptors;

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
    binder.opt<std::string>(
        "iree-vulkan-target-features", targetFeatures,
        llvm::cl::desc(
            "Vulkan target features (SPIR-V version, capabilities, "
            "extensions, etc.) to use. If provided, replaces the default "
            "features for the target."));
    binder.opt<VulkanDispatchAbi>(
        "iree-vulkan-dispatch-abi", dispatchAbi,
        llvm::cl::desc("Selects the Vulkan dispatch ABI emitted for generated "
                       "executables."),
        llvm::cl::values(
            clEnumValN(VulkanDispatchAbi::Descriptors, "descriptors",
                       "Emit descriptor-set executables."),
            clEnumValN(VulkanDispatchAbi::Bda, "bda",
                       "Emit BDA root binding table executables."),
            clEnumValN(VulkanDispatchAbi::All, "all",
                       "Emit all supported executable ABI variants ordered by "
                       "runtime preference.")));
  }
};

using DescriptorSetLayout = std::pair<unsigned, ArrayRef<PipelineBindingAttr>>;

static IREE::GPU::TargetAttr
getTargetAttrWithBdaRootAbiFeatures(IREE::GPU::TargetAttr target) {
  StringRef features = target.getFeatures();
  std::string bdaFeatures = features.str();
  auto appendFeatureIfMissing = [&](StringRef feature) {
    if (features.contains(feature)) {
      return;
    }
    if (!bdaFeatures.empty()) {
      bdaFeatures += ",";
    }
    bdaFeatures += feature;
  };
  appendFeatureIfMissing("cap:Int64");
  appendFeatureIfMissing("cap:PhysicalStorageBufferAddresses");
  appendFeatureIfMissing("ext:SPV_KHR_physical_storage_buffer");

  if (features == StringRef(bdaFeatures)) {
    return target;
  }
  return IREE::GPU::TargetAttr::get(target.getContext(), target.getArch(),
                                    bdaFeatures, target.getWgp(),
                                    target.getChip());
}

static iree_hal_vulkan_BdaDispatchLayoutDef_ref_t
createBdaDispatchLayoutDef(IREE::HAL::ExecutableExportOp exportOp,
                           FlatbufferBuilder &fbb) {
  iree_hal_vulkan_BdaDispatchLayoutDef_start(fbb);
  iree_hal_vulkan_BdaDispatchLayoutDef_abi_version_add(fbb, 1);
  iree_hal_vulkan_BdaDispatchLayoutDef_root_push_constant_offset_add(
      fbb, kBdaDispatchRootOffset);
  iree_hal_vulkan_BdaDispatchLayoutDef_root_push_constant_length_add(
      fbb, kBdaDispatchRootLength);
  iree_hal_vulkan_BdaDispatchLayoutDef_constant_push_constant_offset_add(
      fbb, kBdaDispatchConstantOffset);
  iree_hal_vulkan_BdaDispatchLayoutDef_constant_count_add(
      fbb, static_cast<uint32_t>(exportOp.getLayout().getConstants()));
  iree_hal_vulkan_BdaDispatchLayoutDef_binding_table_entry_type_add(
      fbb, iree_hal_vulkan_BdaBindingTableEntryType_ADDRESS64);
  iree_hal_vulkan_BdaDispatchLayoutDef_binding_count_add(
      fbb, static_cast<uint32_t>(exportOp.getLayout().getBindings().size()));
  return iree_hal_vulkan_BdaDispatchLayoutDef_end(fbb);
}

static std::tuple<iree_hal_vulkan_DescriptorSetLayoutDef_vec_ref_t,
                  iree_hal_vulkan_PipelineLayoutDef_vec_ref_t,
                  DenseMap<IREE::HAL::PipelineLayoutAttr, uint32_t>>
createPipelineLayoutDefs(ArrayRef<IREE::HAL::ExecutableExportOp> exportOps,
                         bool useBdaRootAbi, FlatbufferBuilder &fbb) {
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

    SmallVector<uint32_t> descriptorSetLayoutOrdinals;
    if (!useBdaRootAbi) {
      // Currently only one descriptor set on the compiler side. We could
      // partition it by binding type (direct vs indirect, etc).
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
    }
    auto descriptorSetLayoutOrdinalsRef =
        fbb.createInt32Vec(descriptorSetLayoutOrdinals);

    iree_hal_vulkan_PushConstantRange_vec_ref_t pushConstantRangesRef = 0;
    int64_t pushConstantCount = pipelineLayoutAttr.getConstants();
    if (useBdaRootAbi) {
      pushConstantCount += kBdaDispatchRootDwordCount;
    }
    if (pushConstantCount) {
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
class VulkanTargetDevice final : public TargetDevice {
public:
  VulkanTargetDevice(const VulkanSPIRVTargetOptions & /*options*/) {}

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const final {
    Builder b(context);
    auto deviceConfigAttr = b.getDictionaryAttr({});
    auto executableConfigAttr = b.getDictionaryAttr({});

    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("vulkan-spirv")
        ->getDefaultExecutableTargets(context, "vulkan", executableConfigAttr,
                                      executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("vulkan"),
                                            deviceConfigAttr,
                                            executableTargetAttrs);
  }
};

class VulkanSPIRVTargetBackend final : public TargetBackend {
public:
  VulkanSPIRVTargetBackend(const VulkanSPIRVTargetOptions &options)
      : options_(options) {}

  std::string getLegacyDefaultDeviceID() const final { return "vulkan"; }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const final {
    auto addTarget = [&](bool useBdaRootAbi) {
      if (auto target = getExecutableTarget(context, useBdaRootAbi)) {
        executableTargetAttrs.push_back(target);
      }
    };

    switch (options_.dispatchAbi) {
    case VulkanDispatchAbi::Descriptors:
      addTarget(/*useBdaRootAbi=*/false);
      break;
    case VulkanDispatchAbi::Bda:
      addTarget(/*useBdaRootAbi=*/true);
      break;
    case VulkanDispatchAbi::All:
      addTarget(/*useBdaRootAbi=*/true);
      addTarget(/*useBdaRootAbi=*/false);
      break;
    }
  }

  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(MLIRContext *context, bool useBdaRootAbi) const {
    Builder b(context);
    SmallVector<NamedAttribute, 1> configItems;
    if (options_.target.empty()) {
      emitError(b.getUnknownLoc(), "Vulkan target not specified");
      return nullptr;
    }
    if (auto target = GPU::getVulkanTargetDetails(options_.target, context)) {
      if (!options_.targetFeatures.empty()) {
        target = IREE::GPU::TargetAttr::get(context, target.getArch(),
                                            options_.targetFeatures,
                                            target.getWgp(), target.getChip());
      }
      if (useBdaRootAbi) {
        target = getTargetAttrWithBdaRootAbiFeatures(target);
      }
      addConfigGPUTarget(context, target, configItems);
    } else {
      emitError(b.getUnknownLoc(), "Unknown Vulkan target '")
          << options_.target << "'";
      return nullptr;
    }

    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("vulkan-spirv"),
        useBdaRootAbi ? b.getStringAttr(kVulkanSpirvBdaFormat)
                      : b.getStringAttr(kVulkanSpirvFlatbufferFormat),
        b.getDictionaryAttr(configItems));
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<IREE::Codegen::IREECodegenDialect,
                    IREE::Encoding::IREEEncodingDialect, spirv::SPIRVDialect,
                    gpu::GPUDialect, IREE::GPU::IREEGPUDialect>();
  }

  void
  buildConfigurationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                 OpPassManager &passManager) final {
    buildCodegenConfigurationPreProcessingPassPipeline(passManager);
    buildSPIRVCodegenConfigurationPassPipeline(passManager.nest<ModuleOp>());
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                    OpPassManager &passManager) final {
    OpPassManager &modulePassManager = passManager.nest<ModuleOp>();
    modulePassManager.addPass(
        std::make_unique<PropagateExecutableTargetPass>(targetAttr));
    buildSPIRVCodegenPassPipeline(modulePassManager);
    buildCodegenTranslationPostProcessingPassPipeline(passManager);
  }

  void buildLinkingPassPipeline(OpPassManager &passManager) final {
    buildSPIRVLinkingPassPipeline(passManager);
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) final {
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
      for (spirv::EntryPointOp spirvEntryPointOp :
           spirvModuleOp.getOps<spirv::EntryPointOp>()) {
        auto it = exportOrdinalMap.find(spirvEntryPointOp.getFn());
        if (it == exportOrdinalMap.end()) {
          continue;
        }
        auto [exportOp, ordinal] = it->second;
        sortedExportOps[ordinal] = exportOp;
        exportOps[ordinal] =
            std::make_tuple(exportOp, spirvModuleOp, spirvEntryPointOp);
      }
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

    // Create a serialized SPIR-V module for each entry point. When a
    // spirv.module contains multiple entry points each gets its own copy of
    // the binary — deduplicating shared modules is left as a future
    // optimization (N:M mapping via specialization constants).
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

    const bool useBdaRootAbi =
        variantOp.getTarget().getFormat() == kVulkanSpirvBdaFormat;

    // Create unique descriptor and pipeline layouts for each entry point.
    auto [descriptorSetLayoutsRef, pipelineLayoutsRef, pipelineLayoutMap] =
        createPipelineLayoutDefs(sortedExportOps, useBdaRootAbi, builder);

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

      iree_hal_vulkan_BdaDispatchLayoutDef_ref_t bdaDispatchLayoutRef =
          useBdaRootAbi ? createBdaDispatchLayoutDef(exportOp, builder) : 0;

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
      if (useBdaRootAbi) {
        iree_hal_vulkan_PipelineDef_dispatch_abi_add(
            builder, iree_hal_vulkan_DispatchAbi_BDA_V1);
        iree_hal_vulkan_PipelineDef_bda_dispatch_layout_add(
            builder, bdaDispatchLayoutRef);
      }
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
    auto binaryOp = IREE::HAL::ExecutableBinaryOp::create(
        executableBuilder, variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getHeaderPrefixedBufferAttr(
            executableBuilder.getContext(),
            /*magic=*/iree_hal_vulkan_ExecutableDef_file_identifier,
            /*version=*/0));
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
    auto objectAttr = cast<IREE::HAL::ExecutableObjectAttr>(
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

    const bool useBdaRootAbi =
        variantOp.getTarget().getFormat() == kVulkanSpirvBdaFormat;

    // Generate descriptor set and pipeline layouts from export ops.
    auto exportOps = llvm::to_vector(variantOp.getExportOps());
    auto [descriptorSetLayoutsRef, pipelineLayoutsRef, pipelineLayoutMap] =
        createPipelineLayoutDefs(exportOps, useBdaRootAbi, builder);

    // Create a pipeline for each export.
    SmallVector<iree_hal_vulkan_PipelineDef_ref_t> pipelineRefs;
    for (auto exportOp : exportOps) {
      uint32_t shaderModuleOrdinal = 0; // only one today
      uint32_t pipelineLayoutOrdinal =
          pipelineLayoutMap.at(exportOp.getLayout());

      // Subgroup size requests are optional.
      // TODO: support annotation on an attribute to allow users to specify.
      uint32_t subgroupSize = 0;

      iree_hal_vulkan_BdaDispatchLayoutDef_ref_t bdaDispatchLayoutRef =
          useBdaRootAbi ? createBdaDispatchLayoutDef(exportOp, builder) : 0;

      auto entryPointRef = builder.createString(exportOp.getName());
      iree_hal_vulkan_PipelineDef_start(builder);
      iree_hal_vulkan_PipelineDef_shader_module_ordinal_add(
          builder, shaderModuleOrdinal);
      iree_hal_vulkan_PipelineDef_entry_point_add(builder, entryPointRef);
      iree_hal_vulkan_PipelineDef_pipeline_layout_ordinal_add(
          builder, pipelineLayoutOrdinal);
      iree_hal_vulkan_PipelineDef_subgroup_size_add(builder, subgroupSize);
      if (useBdaRootAbi) {
        iree_hal_vulkan_PipelineDef_dispatch_abi_add(
            builder, iree_hal_vulkan_DispatchAbi_BDA_V1);
        iree_hal_vulkan_PipelineDef_bda_dispatch_layout_add(
            builder, bdaDispatchLayoutRef);
      }
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
    auto binaryOp = IREE::HAL::ExecutableBinaryOp::create(
        executableBuilder, variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getHeaderPrefixedBufferAttr(
            executableBuilder.getContext(),
            /*magic=*/iree_hal_vulkan_ExecutableDef_file_identifier,
            /*version=*/0));
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();
  }

private:
  const VulkanSPIRVTargetOptions &options_;
};

struct VulkanSPIRVSession final
    : PluginSession<VulkanSPIRVSession, VulkanSPIRVTargetOptions,
                    PluginActivationPolicy::DefaultActivated> {
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) final {
    // #hal.device.target<"vulkan", ...
    targets.add("vulkan", [&]() {
      return std::make_shared<VulkanTargetDevice>(options);
    });
  }
  void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) final {
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
