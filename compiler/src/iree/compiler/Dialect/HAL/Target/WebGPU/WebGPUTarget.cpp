// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/WebGPU/WebGPUTarget.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/WGSL/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Target/WebGPU/SPIRVToWGSL.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/wgsl_executable_def_builder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "spirv-tools/libspirv.hpp"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

WebGPUTargetOptions getWebGPUTargetOptionsFromFlags() {
  static llvm::cl::opt<bool> clDebugSymbols(
      "iree-webgpu-debug-symbols",
      llvm::cl::desc(
          "Include debug information like variable names in outputs"),
      llvm::cl::init(true));

  WebGPUTargetOptions targetOptions;
  targetOptions.debugSymbols = clDebugSymbols;

  return targetOptions;
}

// TODO(scotttodd): provide a proper target environment for WebGPU.
static spirv::TargetEnvAttr getWebGPUTargetEnv(MLIRContext *context) {
  // TODO(scotttodd): find list of SPIR-V extensions supported by WebGPU/WGSL
  auto triple = spirv::VerCapExtAttr::get(
      spirv::Version::V_1_0, {spirv::Capability::Shader},
      {spirv::Extension::SPV_KHR_storage_buffer_storage_class}, context);
  return spirv::TargetEnvAttr::get(
      triple, spirv::getDefaultResourceLimits(context),
      spirv::ClientAPI::WebGPU, spirv::Vendor::Unknown,
      spirv::DeviceType::Unknown, spirv::TargetEnvAttr::kUnknownDeviceID);
}

class WebGPUTargetBackend : public TargetBackend {
public:
  WebGPUTargetBackend(WebGPUTargetOptions options)
      : options_(std::move(options)) {}

  // NOTE: we could vary this based on the options such as 'webgpu-v2'.
  std::string name() const override { return "webgpu"; }

  // TODO(scotttodd): Prune FlowDialect dep when WGSLReplacePushConstantsPass
  //     does not use the Flow dialect (TranslateExecutables calls this
  //     function and _does not_ query which passes are used by the dynamic
  //     pipeline created by buildTranslationPassPipeline)
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect, IREE::Flow::FlowDialect,
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
    if (variantOp.isExternal())
      return;

    // WebGPU does not support push constants (yet?), so replace loads from
    // push constants with loads from uniform buffers.
    // The corresponding runtime code must perform similar emulation, based
    // on the push constant count listed in the executable layout.
    passManager.nest<ModuleOp>().nest<func::FuncOp>().addPass(
        createWGSLReplacePushConstantsPass());

    // From WGSL spec, "Floating Point Evaluation"
    // (https://www.w3.org/TR/WGSL/#floating-point-evaluation):
    // - Implementations may assume that NaNs and infinities are not present at
    //   runtime.
    //   - In such an implementation, when an evaluation would produce an
    //     infinity or a NaN, an undefined value of the target type is produced
    //     instead.
    // So WebGPU effectively assumes fast math mode. We also don't have reliable
    // ways to check whether a floating point number is NaN or infinity.
    // Therefore, just let the SPIR-V CodeGen to avoid generating guards w.r.t.
    // NaN and infinity.
    buildSPIRVCodegenPassPipeline(passManager, /*enableFastMath=*/true);

    // WGSL does not support extended multiplication:
    // https://github.com/gpuweb/gpuweb/issues/1565. Make sure to lower it to
    // regular multiplication before we convert SPIR-V to WGSL.
    passManager.nest<ModuleOp>().nest<spirv::ModuleOp>().addPass(
        spirv::createSPIRVWebGPUPreparePass());
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    ModuleOp innerModuleOp = variantOp.getInnerModule();
    auto spirvModuleOps = innerModuleOp.getOps<spirv::ModuleOp>();
    if (!llvm::hasSingleElement(spirvModuleOps)) {
      // TODO(#7824): Implement linking / shader module combining and relax this
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

    // The schema expects each shader module to have entry points named "dN",
    // where N is the entry point ordinal.
    // For each executable entry point op, rename the entry point symbol using
    // that convention and keep track of the mapping between entry point
    // ordinals to which shader module they reference.
    auto exportOps = llvm::to_vector(variantOp.getExportOps());
    llvm::SmallVector<uint32_t> entryPointOrdinals(exportOps.size());
    SymbolTableCollection symbolTable;
    SymbolUserMap symbolUsers(symbolTable, variantOp);
    for (auto exportOp : exportOps) {
      auto entryPointFunc = dyn_cast<spirv::FuncOp>(
          SymbolTable::lookupSymbolIn(spvModuleOp, exportOp.getSymName()));

      std::string symbolName = llvm::formatv("d{0}", exportOp.getOrdinal());
      mlir::StringAttr nameAttr =
          mlir::StringAttr::get(variantOp->getContext(), symbolName);

      symbolUsers.replaceAllUsesWith(entryPointFunc, nameAttr);
      exportOp.setName(symbolName); // Same symbol reference? Not in table?
      SymbolTable::setSymbolName(entryPointFunc, symbolName);

      // We only have one shader module right now, so all point to index 0.
      // TODO(#7824): Support multiple shader modules per executable.
      uint64_t ordinal =
          exportOp.getOrdinal().value_or(APInt(64, 0)).getZExtValue();
      entryPointOrdinals[ordinal] = 0;
    }

    // Serialize the spirv::ModuleOp into binary format.
    SmallVector<uint32_t, 0> spvBinary;
    spirv::SerializationOptions serializationOptions;
    serializationOptions.emitSymbolName = options_.debugSymbols;
    serializationOptions.emitDebugInfo = options_.debugSymbols;
    if (failed(
            spirv::serialize(spvModuleOp, spvBinary, serializationOptions)) ||
        spvBinary.empty()) {
      return variantOp.emitError() << "failed to serialize spirv.module";
    }
    if (!options.dumpIntermediatesPath.empty()) {
      dumpDataToPath<uint32_t>(options.dumpIntermediatesPath,
                               options.dumpBaseName, variantOp.getName(),
                               ".spv", spvBinary);

      // Disassemble the shader and save that too.
      // Note: this should match what getWebGPUTargetEnv used.
      // TODO(scotttodd): Query spirv env from the executable variant?
      spvtools::SpirvTools spirvTools(SPV_ENV_VULKAN_1_0);
      std::string spvDisassembled;
      if (spirvTools.Disassemble(
              spvBinary.data(), spvBinary.size(), &spvDisassembled,
              SPV_BINARY_TO_TEXT_OPTION_INDENT |
                  SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)) {
        dumpDataToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                       variantOp.getName(), ".spvasm", spvDisassembled);
      } else {
        llvm::errs() << "Failed to disassemble SPIR-V binary\n";
      }
    }

    // Compile SPIR-V to WGSL source code.
    auto wgsl = compileSPIRVToWGSL(spvBinary);
    if (!wgsl.has_value()) {
      // TODO(scotttodd): restructure branching and write disassembled SPIR-V
      //                  to stderr / an error diagnostic (don't want to
      //                  disassemble if successful + option not set, also
      //                  don't want to disassemble twice :P)
      return variantOp.emitError()
             << "failed to compile SPIR-V to WGSL. Consider inspecting the "
                "shader program using -iree-hal-dump-executable-intermediates.";
    }
    if (!options.dumpBinariesPath.empty()) {
      dumpDataToPath(options.dumpBinariesPath, options.dumpBaseName,
                     variantOp.getName(), ".wgsl", wgsl.value());
    }

    // Pack the WGSL and metadata into a FlatBuffer.
    FlatbufferBuilder builder;
    iree_hal_wgsl_ExecutableDef_start_as_root(builder);

    iree_hal_wgsl_ShaderModuleDef_start(builder);
    auto wgslRef = builder.createString(wgsl.value());
    iree_hal_wgsl_ShaderModuleDef_code_add(builder, wgslRef);
    // TODO(scotttodd): populate source map
    auto shaderModuleRef = iree_hal_wgsl_ShaderModuleDef_end(builder);

    auto shaderModulesVec = iree_hal_wgsl_ShaderModuleDef_vec_create(
        builder, &shaderModuleRef, /*len=*/1);
    iree_hal_wgsl_ExecutableDef_shader_modules_add(builder, shaderModulesVec);

    auto entryPointsRef = flatbuffers_uint32_vec_create(
        builder, entryPointOrdinals.data(), entryPointOrdinals.size());
    iree_hal_wgsl_ExecutableDef_entry_points_add(builder, entryPointsRef);

    iree_hal_wgsl_ExecutableDef_end_as_root(builder);

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
    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    targetAttrs.push_back(
        getExecutableTarget(context, getWebGPUTargetEnv(context)));
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
        context, b.getStringAttr("webgpu"), b.getStringAttr("webgpu-wgsl-fb"),
        configAttr);
  }

  WebGPUTargetOptions options_;
};

void registerWebGPUTargetBackends(
    std::function<WebGPUTargetOptions()> queryOptions) {
  getWebGPUTargetOptionsFromFlags();
  auto backendFactory = [=]() {
    return std::make_shared<WebGPUTargetBackend>(queryOptions());
  };
  // #hal.device.target<"webgpu", ...
  static TargetBackendRegistration registration0("webgpu", backendFactory);
  // #hal.executable.target<"webgpu-wgsl", ...
  static TargetBackendRegistration registration1("webgpu-wgsl", backendFactory);
}

} // namespace HAL
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
