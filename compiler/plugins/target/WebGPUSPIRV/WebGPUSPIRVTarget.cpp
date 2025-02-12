// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/WebGPUSPIRV/SPIRVToWGSL.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/WGSL/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Utils/ExecutableDebugInfoUtils.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/webgpu_executable_def_builder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "spirv-tools/libspirv.hpp"

namespace mlir::iree_compiler::IREE::HAL {

namespace {

struct WebGPUSPIRVOptions {
  bool debugSymbols = true;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("WebGPU HAL Target");
    binder.opt<bool>(
        "iree-webgpu-debug-symbols", debugSymbols, llvm::cl::cat(category),
        llvm::cl::desc(
            "Include debug information like variable names in outputs."));
  }
};

// TODO: WebGPUOptions for choosing the version/extensions/etc.
class WebGPUTargetDevice : public TargetDevice {
public:
  WebGPUTargetDevice(const WebGPUSPIRVOptions &options) : options(options) {}

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override {
    Builder b(context);
    auto configAttr = b.getDictionaryAttr({});

    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("webgpu-spirv")
        ->getDefaultExecutableTargets(context, "webgpu", configAttr,
                                      executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("webgpu"),
                                            configAttr, executableTargetAttrs);
  }

private:
  const WebGPUSPIRVOptions &options;
};

class WebGPUSPIRVTargetBackend : public TargetBackend {
public:
  WebGPUSPIRVTargetBackend(const WebGPUSPIRVOptions &options)
      : options(options) {}

  std::string getLegacyDefaultDeviceID() const override { return "webgpu"; }

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
    if (auto target = GPU::getWebGPUTargetDetails(context)) {
      configItems.emplace_back("iree.gpu.target", target);
    }

    return b.getAttr<IREE::HAL::ExecutableTargetAttr>(
        b.getStringAttr("webgpu-spirv"), b.getStringAttr("webgpu-wgsl-fb"),
        b.getDictionaryAttr(configItems));
  }

  // TODO(scotttodd): Prune FlowDialect dep when WGSLReplacePushConstantsPass
  //     does not use the Flow dialect (TranslateExecutables calls this
  //     function and _does not_ query which passes are used by the dynamic
  //     pipeline created by buildTranslationPassPipeline)
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect, IREE::Flow::FlowDialect,
                    spirv::SPIRVDialect, gpu::GPUDialect,
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

    // Prepare SPIR-V for WebGPU by expanding or removing unsupported ops.
    // For example,
    //   * WGSL does not support extended multiplication:
    //     https://github.com/gpuweb/gpuweb/issues/1565, so we lower to
    //     regular multiplication
    //   * WGSL does not support NaN or infinities:
    //     https://www.w3.org/TR/WGSL/#floating-point-evaluation
    passManager.nest<ModuleOp>().nest<spirv::ModuleOp>().addPass(
        spirv::createSPIRVWebGPUPreparePass());
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
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
    if (!serOptions.dumpIntermediatesPath.empty()) {
      std::string assembly;
      llvm::raw_string_ostream os(assembly);
      spvModuleOp.print(os, OpPrintingFlags().useLocalScope());
      dumpDataToPath(serOptions.dumpIntermediatesPath, serOptions.dumpBaseName,
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

      std::string symbolName = llvm::formatv("d{}", exportOp.getOrdinal());
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
    spirv::SerializationOptions spirvSerializationOptions;
    spirvSerializationOptions.emitSymbolName = options.debugSymbols;
    spirvSerializationOptions.emitDebugInfo = options.debugSymbols;
    if (failed(spirv::serialize(spvModuleOp, spvBinary,
                                spirvSerializationOptions)) ||
        spvBinary.empty()) {
      return variantOp.emitError() << "failed to serialize spirv.module";
    }
    if (!serOptions.dumpIntermediatesPath.empty()) {
      dumpDataToPath<uint32_t>(serOptions.dumpIntermediatesPath,
                               serOptions.dumpBaseName, variantOp.getName(),
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
        dumpDataToPath(serOptions.dumpIntermediatesPath,
                       serOptions.dumpBaseName, variantOp.getName(), ".spvasm",
                       spvDisassembled);
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
    if (!serOptions.dumpBinariesPath.empty()) {
      dumpDataToPath(serOptions.dumpBinariesPath, serOptions.dumpBaseName,
                     variantOp.getName(), ".wgsl", wgsl.value());
    }

    // Pack the WGSL and metadata into a FlatBuffer.
    FlatbufferBuilder builder;
    iree_hal_webgpu_ExecutableDef_start_as_root(builder);

    // Attach embedded source file contents.
    auto sourceFilesRef = createSourceFilesVec(
        serOptions.debugLevel, variantOp.getSourcesAttr(), builder);

    iree_hal_webgpu_ShaderModuleDef_start(builder);
    auto wgslRef = builder.createString(wgsl.value());
    iree_hal_webgpu_ShaderModuleDef_wgsl_source_add(builder, wgslRef);
    // TODO(scotttodd): populate source map
    auto shaderModuleRef = iree_hal_webgpu_ShaderModuleDef_end(builder);

    auto shaderModulesVec = iree_hal_webgpu_ShaderModuleDef_vec_create(
        builder, &shaderModuleRef, /*len=*/1);
    iree_hal_webgpu_ExecutableDef_shader_modules_add(builder, shaderModulesVec);

    auto entryPointsRef = flatbuffers_uint32_vec_create(
        builder, entryPointOrdinals.data(), entryPointOrdinals.size());
    iree_hal_webgpu_ExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_hal_webgpu_ExecutableDef_source_files_add(builder, sourceFilesRef);

    iree_hal_webgpu_ExecutableDef_end_as_root(builder);

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
  const WebGPUSPIRVOptions &options;
};

struct WebGPUSPIRVSession
    : public PluginSession<WebGPUSPIRVSession, WebGPUSPIRVOptions,
                           PluginActivationPolicy::DefaultActivated> {
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) {
    // #hal.device.target<"webgpu", ...
    targets.add("webgpu", [=]() {
      return std::make_shared<WebGPUTargetDevice>(options);
    });
  }
  void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) {
    // #hal.executable.target<"webgpu-spirv", ...
    targets.add("webgpu-spirv", [=]() {
      return std::make_shared<WebGPUSPIRVTargetBackend>(options);
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL

IREE_DEFINE_COMPILER_OPTION_FLAGS(
    mlir::iree_compiler::IREE::HAL::WebGPUSPIRVOptions);

extern "C" bool iree_register_compiler_plugin_hal_target_webgpu_spirv(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::iree_compiler::IREE::HAL::WebGPUSPIRVSession>(
      "hal_target_webgpu_spirv");
  return true;
}
