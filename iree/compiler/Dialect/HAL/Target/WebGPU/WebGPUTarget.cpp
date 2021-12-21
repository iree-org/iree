// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/WebGPU/WebGPUTarget.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Target/WebGPU/SPIRVToWGSL.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/wgsl_executable_def_builder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
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

  static llvm::cl::opt<bool> clWebGPUKeepShaderModules(
      "iree-webgpu-keep-shader-modules",
      llvm::cl::desc("Save shader modules to disk separately"),
      llvm::cl::init(false));

  WebGPUTargetOptions targetOptions;
  targetOptions.keepShaderModules = clWebGPUKeepShaderModules;

  return targetOptions;
}

// TODO(scotttodd): provide a proper target environment for WebGPU.
static spirv::TargetEnvAttr getWebGPUTargetEnv(MLIRContext *context) {
  // TODO(scotttodd): find list of SPIR-V extensions supported by WebGPU/WGSL
  auto triple = spirv::VerCapExtAttr::get(
      spirv::Version::V_1_0, {spirv::Capability::Shader},
      {spirv::Extension::SPV_KHR_storage_buffer_storage_class}, context);
  return spirv::TargetEnvAttr::get(triple, spirv::Vendor::Unknown,
                                   spirv::DeviceType::Unknown,
                                   spirv::TargetEnvAttr::kUnknownDeviceID,
                                   spirv::getDefaultResourceLimits(context));
}

class WebGPUTargetBackend : public TargetBackend {
 public:
  WebGPUTargetBackend(WebGPUTargetOptions options)
      : options_(std::move(options)) {}

  // NOTE: we could vary this based on the options such as 'webgpu-v2'.
  std::string name() const override { return "webgpu"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect, spirv::SPIRVDialect,
                    gpu::GPUDialect>();
  }

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(b.getIdentifier("executable_targets"),
                             getExecutableTargets(context));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    buildSPIRVCodegenPassPipeline(passManager);
    // TODO(scotttodd): additional passes for WebGPU/WGSL
    //                  (here or during serialization?)
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    ModuleOp innerModuleOp = variantOp.getInnerModule();
    auto spirvModuleOps = innerModuleOp.getOps<spirv::ModuleOp>();
    if (!llvm::hasSingleElement(spirvModuleOps)) {
      // TODO(#7824): Implement linking / shader module combining and relax this
      return variantOp.emitError()
             << "should only contain exactly one spv.module op";
    }
    auto spvModuleOp = *spirvModuleOps.begin();

    // The schema expects each shader module to have entry points named "dN",
    // where N is the entry point ordinal.
    // For each executable entry point op, rename the entry point symbol using
    // that convention and keep track of the mapping between entry point
    // ordinals to which shader module they reference.
    auto entryPointOps = llvm::to_vector<4>(
        variantOp.getOps<IREE::HAL::ExecutableEntryPointOp>());
    llvm::SmallVector<uint32_t, 4> entryPointOrdinals(entryPointOps.size());
    SymbolTableCollection symbolTable;
    SymbolUserMap symbolUsers(symbolTable, variantOp);
    for (auto entryPointOp : entryPointOps) {
      auto entryPointFunc = dyn_cast<spirv::FuncOp>(
          SymbolTable::lookupSymbolIn(spvModuleOp, entryPointOp.sym_name()));

      std::string symbolName = llvm::formatv("d{0}", entryPointOp.ordinal());
      mlir::StringAttr nameAttr =
          mlir::StringAttr::get(variantOp->getContext(), symbolName);

      symbolUsers.replaceAllUsesWith(entryPointFunc, nameAttr);
      entryPointOp.setName(symbolName);  // Same symbol reference? Not in table?
      SymbolTable::setSymbolName(entryPointFunc, symbolName);

      // We only have one shader module right now, so all point to index 0.
      // TODO(#7824): Support multiple shader modules per executable
      entryPointOrdinals[entryPointOp.ordinal().getZExtValue()] = 0;
    }

    // Serialize the spirv::ModuleOp into binary format.
    SmallVector<uint32_t, 0> spvBinary;
    spirv::SerializationOptions serializationOptions;
    serializationOptions.emitSymbolName = options_.debugSymbols;
    serializationOptions.emitDebugInfo = options_.debugSymbols;
    if (failed(
            spirv::serialize(spvModuleOp, spvBinary, serializationOptions)) ||
        spvBinary.empty()) {
      return variantOp.emitError() << "failed to serialize spv.module";
    }
    if (options_.keepShaderModules) {
      saveShaderToTempFile(variantOp, "spv",
                           reinterpret_cast<const char *>(spvBinary.data()),
                           spvBinary.size_in_bytes());

      // Disassemble the shader and save that too.
      // Note: this should match what getWebGPUTargetEnv used.
      // TODO(scotttodd): Query spirv env from the executable variant?
      spvtools::SpirvTools spirvTools(SPV_ENV_VULKAN_1_0);
      std::string spvDisassembled;
      if (spirvTools.Disassemble(
              spvBinary.data(), spvBinary.size(), &spvDisassembled,
              SPV_BINARY_TO_TEXT_OPTION_INDENT |
                  SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES)) {
        saveShaderToTempFile(variantOp, "spvasm", spvDisassembled.data(),
                             spvDisassembled.size());
      } else {
        llvm::errs() << "Failed to disassemble SPIR-V binary\n";
      }
    }

    // Compile SPIR-V to WGSL source code.
    auto wgsl = compileSPIRVToWGSL(spvBinary);
    if (!wgsl.hasValue()) {
      // TODO(scotttodd): restructure branching and write disassembled SPIR-V
      //                  to stderr / an error diagnostic (don't want to
      //                  disassemble if successful + option not set, also
      //                  don't want to disassemble twice :P)
      return variantOp.emitError()
             << "failed to compile SPIR-V to WGSL. Consider inspecting the "
                "shader program using -iree-webgpu-keep-shader-modules";
    }
    if (options_.keepShaderModules) {
      saveShaderToTempFile(variantOp, "wgsl", wgsl.getValue().data(),
                           wgsl.getValue().length());
    }

    // Pack the WGSL and metadata into a flatbuffer.
    FlatbufferBuilder builder;
    iree_WGSLExecutableDef_start_as_root(builder);

    iree_WGSLShaderModuleDef_start(builder);
    auto wgslRef = builder.createString(wgsl.getValue());
    iree_WGSLShaderModuleDef_code_add(builder, wgslRef);
    // TODO(scotttodd): populate source map
    auto shaderModuleRef = iree_WGSLShaderModuleDef_end(builder);

    auto shaderModulesVec = iree_WGSLShaderModuleDef_vec_create(
        builder, &shaderModuleRef, /*len=*/1);
    iree_WGSLExecutableDef_shader_modules_add(builder, shaderModulesVec);

    auto entryPointsRef = flatbuffers_uint32_vec_create(
        builder, entryPointOrdinals.data(), entryPointOrdinals.size());
    iree_WGSLExecutableDef_entry_points_add(builder, entryPointsRef);

    iree_WGSLExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.sym_name(),
        variantOp.target().getFormat(),
        builder.getBufferAttr(executableBuilder.getContext()));
    binaryOp.mime_typeAttr(
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

  IREE::HAL::ExecutableTargetAttr getExecutableTarget(
      MLIRContext *context, spirv::TargetEnvAttr targetEnv) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(b.getIdentifier(spirv::getTargetEnvAttrName()),
                             targetEnv);

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("webgpu"), b.getStringAttr("webgpu-wgsl-fb"),
        configAttr);
  }

  void saveShaderToTempFile(IREE::HAL::ExecutableVariantOp variantOp,
                            llvm::StringRef suffix, const char *data,
                            size_t size) {
    llvm::SmallString<32> filePath;
    if (std::error_code error = llvm::sys::fs::createTemporaryFile(
            variantOp.getName(), suffix, filePath)) {
      llvm::errs() << "failed to generate temp file for shader: "
                   << error.message();
      return;
    }
    std::error_code error;
    auto file = std::make_unique<llvm::ToolOutputFile>(filePath, error,
                                                       llvm::sys::fs::OF_None);
    if (error) {
      llvm::errs() << "failed to open temp file for shader '" << filePath
                   << "': " << error.message();
      return;
    }

    // TODO(scotttodd): refactor to group these messages
    mlir::emitRemark(variantOp.getLoc())
        << "Shader file for " << variantOp.getName() << " preserved:\n"
        << "    " << filePath;
    file->os().write(data, size);
    file->keep();
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

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
