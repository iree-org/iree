// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/VulkanSPIRV/VulkanSPIRVTarget.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanAttributes.h"
#include "iree/compiler/Dialect/Vulkan/IR/VulkanDialect.h"
#include "iree/compiler/Dialect/Vulkan/Utils/TargetEnvironment.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "iree/schemas/spirv_executable_def_builder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
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
#include "mlir/IR/SymbolTable.h"
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

  static llvm::cl::list<std::string> clVulkanTargetTriples{
      "iree-vulkan-target-triple",
      llvm::cl::desc(
          "Vulkan target triple controlling the SPIR-V environment."),
  };

  static llvm::cl::list<std::string> clVulkanTargetEnvs{
      "iree-vulkan-target-env",
      llvm::cl::desc(
          "Vulkan target environment as #vk.target_env attribute assembly."),
  };

  static llvm::cl::opt<bool> clVulkanIndirectBindings(
      "iree-vulkan-experimental-indirect-bindings",
      llvm::cl::desc("Force indirect bindings for all generated dispatches."),
      llvm::cl::init(false));

  VulkanSPIRVTargetOptions targetOptions;

  int tripleCount = clVulkanTargetTriples.getNumOccurrences();
  int envCount = clVulkanTargetEnvs.getNumOccurrences();
  int tripleIdx = 0;
  int envIdx = 0;

  // Use the relative positions in the argument list to get the flat list of
  // target environments.
  SmallVector<std::string> vulkanTargetTriplesAndEnvs;
  for (int i = 0, e = tripleCount + envCount; i < e; ++i) {
    if (tripleIdx >= tripleCount) {
      vulkanTargetTriplesAndEnvs.push_back(clVulkanTargetEnvs[envIdx++]);
      continue;
    }
    if (envIdx >= envCount) {
      vulkanTargetTriplesAndEnvs.push_back(clVulkanTargetTriples[tripleIdx++]);
      continue;
    }
    if (clVulkanTargetTriples.getPosition(tripleIdx) >
        clVulkanTargetEnvs.getPosition(envIdx)) {
      vulkanTargetTriplesAndEnvs.push_back(clVulkanTargetEnvs[envIdx++]);
    } else {
      vulkanTargetTriplesAndEnvs.push_back(clVulkanTargetTriples[tripleIdx++]);
    }
  }
  targetOptions.targetTriplesAndEnvs = vulkanTargetTriplesAndEnvs;

  targetOptions.indirectBindings = clVulkanIndirectBindings;

  return targetOptions;
}

// Returns the Vulkan target environment for conversion.
static spirv::TargetEnvAttr
getSPIRVTargetEnv(const std::string &vulkanTargetTripleOrEnv,
                  MLIRContext *context) {
  if (!vulkanTargetTripleOrEnv.empty()) {
    if (vulkanTargetTripleOrEnv[0] == '#') {
      if (auto attr = parseAttribute(vulkanTargetTripleOrEnv, context)) {
        if (auto vkTargetEnv = llvm::dyn_cast<Vulkan::TargetEnvAttr>(attr)) {
          return convertTargetEnv(vkTargetEnv);
        }
      }
      emitError(Builder(context).getUnknownLoc())
          << "cannot parse vulkan target environment as #vk.target_env "
             "attribute: '"
          << vulkanTargetTripleOrEnv << "'";
    } else {
      return convertTargetEnv(
          Vulkan::getTargetEnvForTriple(context, vulkanTargetTripleOrEnv));
    }
  }
  return {};
}

class VulkanSPIRVTargetBackend : public TargetBackend {
public:
  VulkanSPIRVTargetBackend(VulkanSPIRVTargetOptions options)
      : options_(std::move(options)) {}

  // NOTE: we could vary these based on the options such as 'vulkan-v1.1'.
  std::string name() const override { return "vulkan"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect, Vulkan::VulkanDialect,
                    spirv::SPIRVDialect, gpu::GPUDialect>();
  }

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    // Indicates that the runtime HAL driver operates only in the legacy
    // synchronous mode.
    configItems.emplace_back(b.getStringAttr("legacy_sync"), b.getUnitAttr());

    configItems.emplace_back(b.getStringAttr("executable_targets"),
                             getExecutableTargets(context));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
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
    if (!llvm::hasSingleElement(spirvModuleOps)) {
      return variantOp.emitError()
             << "should only contain exactly one spirv.module op";
    }
    auto spvModuleOp = *spirvModuleOps.begin();
    if (!options.dumpIntermediatesPath.empty()) {
      std::string assembly;
      llvm::raw_string_ostream os(assembly);
      spvModuleOp.print(os, OpPrintingFlags().useLocalScope());
      dumpDataToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                     variantOp.getName(), ".mlir", assembly);
    }

    FlatbufferBuilder builder;
    iree_hal_spirv_ExecutableDef_start_as_root(builder);

    // Serialize the spirv::ModuleOp into the binary that we will embed in the
    // final FlatBuffer.
    SmallVector<uint32_t, 256> spvBinary;
    if (failed(spirv::serialize(spvModuleOp, spvBinary)) || spvBinary.empty()) {
      return variantOp.emitError() << "failed to serialize spirv.module";
    }
    if (!options.dumpBinariesPath.empty()) {
      dumpDataToPath<uint32_t>(options.dumpBinariesPath, options.dumpBaseName,
                               variantOp.getName(), ".spv", spvBinary);
    }

    auto spvCodeRef = flatbuffers_uint32_vec_create(builder, spvBinary.data(),
                                                    spvBinary.size());

    // The runtime uses ordinals instead of names. We provide the list of entry
    // point names here that are then passed in VkShaderModuleCreateInfo.
    SmallVector<StringRef> entryPointNames;
    SmallVector<uint32_t> subgroupSizes;
    SmallVector<iree_hal_spirv_FileLineLocDef_ref_t> sourceLocationRefs;
    bool hasAnySubgroupSizes = false;
    spvModuleOp.walk([&](spirv::EntryPointOp exportOp) {
      entryPointNames.push_back(exportOp.getFn());

      auto fn = spvModuleOp.lookupSymbol<spirv::FuncOp>(exportOp.getFn());
      auto abi = fn->getAttrOfType<spirv::EntryPointABIAttr>(
          spirv::getEntryPointABIAttrName());
      if (abi && abi.getSubgroupSize()) {
        subgroupSizes.push_back(*abi.getSubgroupSize());
        hasAnySubgroupSizes = true;
      } else {
        subgroupSizes.push_back(0);
      }

      // Optional source location information for debugging/profiling.
      if (options.debugLevel >= 1) {
        if (auto loc = findFirstFileLoc(exportOp.getLoc())) {
          auto filenameRef = builder.createString(loc->getFilename());
          sourceLocationRefs.push_back(iree_hal_spirv_FileLineLocDef_create(
              builder, filenameRef, loc->getLine()));
        }
      }
    });
    auto entryPointsRef = builder.createStringVec(entryPointNames);
    flatbuffers_int32_vec_ref_t subgroupSizesRef =
        hasAnySubgroupSizes ? builder.createInt32Vec(subgroupSizes) : 0;

    iree_hal_spirv_ExecutableDef_entry_points_add(builder, entryPointsRef);
    if (subgroupSizesRef) {
      iree_hal_spirv_ExecutableDef_subgroup_sizes_add(builder,
                                                      subgroupSizesRef);
    }
    iree_hal_spirv_ExecutableDef_code_add(builder, spvCodeRef);
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

    auto entryPointsRef = builder.createStringVec(entryPointNames);

    iree_hal_spirv_ExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_hal_spirv_ExecutableDef_code_add(builder, spvCodeRef);
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
  ArrayAttr getExecutableTargets(MLIRContext *context) const {
    SmallVector<Attribute> targetAttrs;

    for (std::string targetTripleOrEnv : options_.targetTriplesAndEnvs) {
      targetAttrs.push_back(getExecutableTarget(
          context, getSPIRVTargetEnv(targetTripleOrEnv, context),
          options_.indirectBindings));
    }

    // If no environment specified, populate with a minimal target.
    if (targetAttrs.empty()) {
      targetAttrs.push_back(getExecutableTarget(
          context, getSPIRVTargetEnv("unknown-unknown-unknown", context),
          options_.indirectBindings));
    }
    return ArrayAttr::get(context, targetAttrs);
  }

  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(MLIRContext *context, spirv::TargetEnvAttr targetEnv,
                      bool indirectBindings) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(b.getStringAttr(spirv::getTargetEnvAttrName()),
                             targetEnv);
    if (indirectBindings) {
      configItems.emplace_back(b.getStringAttr("hal.bindings.indirect"),
                               UnitAttr::get(context));
    }

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("vulkan"),
        indirectBindings ? b.getStringAttr("vulkan-spirv-fb-ptr")
                         : b.getStringAttr("vulkan-spirv-fb"),
        configAttr);
  }

  VulkanSPIRVTargetOptions options_;
};

void registerVulkanSPIRVTargetBackends(
    std::function<VulkanSPIRVTargetOptions()> queryOptions) {
  getVulkanSPIRVTargetOptionsFromFlags();
  auto backendFactory = [=]() {
    return std::make_shared<VulkanSPIRVTargetBackend>(queryOptions());
  };
  // #hal.device.target<"vulkan", ...
  static TargetBackendRegistration registration0("vulkan", backendFactory);
  // #hal.executable.target<"vulkan-spirv", ...
  static TargetBackendRegistration registration1("vulkan-spirv",
                                                 backendFactory);
}

} // namespace HAL
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
