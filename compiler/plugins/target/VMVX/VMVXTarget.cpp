// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/VMVX/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/Dialect/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/VMVX/Transforms/Passes.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::HAL {

namespace {
struct VMVXOptions {
  bool enableMicrokernels = false;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("VMVX HAL Target");
    binder.opt<bool>(
        "iree-vmvx-enable-microkernels", enableMicrokernels,
        llvm::cl::cat(category),
        llvm::cl::desc("Enables microkernel lowering for vmvx (experimental)"));
  }
};
} // namespace

static IREE::HAL::ExecutableTargetAttr
getVMVXExecutableTarget(bool enableMicrokernels, MLIRContext *context,
                        StringRef backend, StringRef format) {
  Builder b(context);
  SmallVector<NamedAttribute> configItems;

  configItems.emplace_back(
      b.getStringAttr("ukernels"),
      b.getStringAttr(enableMicrokernels ? "all" : "none"));

  return b.getAttr<IREE::HAL::ExecutableTargetAttr>(
      b.getStringAttr(backend), b.getStringAttr(format),
      b.getDictionaryAttr(configItems));
}

// TODO(benvanik): move to a CPU device registration outside of VMVX.
class VMVXTargetDevice final : public TargetDevice {
public:
  VMVXTargetDevice() = default;

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    auto configAttr = b.getDictionaryAttr(configItems);

    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("vmvx")->getDefaultExecutableTargets(
        context, "vmvx", configAttr, executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("vmvx"),
                                            configAttr, executableTargetAttrs);
  }

  std::optional<IREE::HAL::DeviceTargetAttr>
  getHostDeviceTarget(MLIRContext *context,
                      const TargetRegistry &targetRegistry) const override {
    return getDefaultDeviceTarget(context, targetRegistry);
  }
};

class VMVXTargetBackend final : public TargetBackend {
public:
  VMVXTargetBackend(const VMVXOptions &options) : options(options) {}

  std::string getLegacyDefaultDeviceID() const override { return "vmvx"; }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    executableTargetAttrs.push_back(getVMVXExecutableTarget(
        options.enableMicrokernels, context, "vmvx", "vmvx-bytecode-fb"));
  }

  void getHostExecutableTargets(MLIRContext *context, StringRef deviceID,
                                DictionaryAttr deviceConfigAttr,
                                SmallVectorImpl<IREE::HAL::ExecutableTargetAttr>
                                    &executableTargetAttrs) const override {
    executableTargetAttrs.push_back(getVMVXExecutableTarget(
        options.enableMicrokernels, context, "vmvx", "vmvx-bytecode-fb"));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect, IREE::VM::VMDialect,
                    IREE::VMVX::VMVXDialect,
                    IREE::LinalgExt::IREELinalgExtDialect>();
  }

  IREE::VM::TargetOptions
  getTargetOptions(IREE::HAL::ExecutableTargetAttr targetAttr) {
    // TODO(benvanik): derive these from a vm target triple.
    auto vmOptions = IREE::VM::TargetOptions::FromFlags::get();
    vmOptions.f32Extension = true;
    vmOptions.optimizeForStackSize = false;
    return vmOptions;
  }

  void buildConfigurationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                      OpPassManager &passManager) override {
    IREE::VMVX::buildVMVXConfigurationPassPipeline(passManager);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpPassManager &passManager) override {
    IREE::VMVX::buildVMVXTransformPassPipeline(passManager);

    OpPassManager &nestedModulePM = passManager.nest<ModuleOp>();

    auto vmOptions = getTargetOptions(variantOp.getTargetAttr());
    IREE::VM::buildVMTransformPassPipeline(nestedModulePM, vmOptions);
  }

  void buildLinkingPassPipeline(OpPassManager &passManager) override {
    buildVMVXLinkingPassPipeline(passManager);
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    // Add reflection information used at runtime specific to the HAL interface.
    SymbolTable symbolTable(variantOp.getInnerModule());
    for (auto exportOp : variantOp.getBlock().getOps<ExecutableExportOp>()) {
      auto funcOp = symbolTable.lookup<IREE::VM::FuncOp>(exportOp.getName());

      // Optionally entry points may specify that they require workgroup local
      // memory. We fetch that value here and plumb it through so the runtime
      // knows how much memory to reserve and pass in.
      auto localMemorySizeAttr = exportOp.getWorkgroupLocalMemoryAttr();
      if (localMemorySizeAttr) {
        funcOp.setReflectionAttr("local_memory", localMemorySizeAttr);
      }
    }

    // Serialize the VM module to bytes and embed it directly.
    SmallVector<char> moduleData;
    {
      auto vmOptions = getTargetOptions(variantOp.getTargetAttr());
      // TODO(benvanik): plumb this through somewhere? these options are mostly
      // about output format stuff such as debug information so it's probably
      // fine to share.
      auto bytecodeOptions = IREE::VM::BytecodeTargetOptions::FromFlags::get();
      llvm::raw_svector_ostream stream(moduleData);
      if (failed(translateModuleToBytecode(variantOp.getInnerModule(),
                                           vmOptions, bytecodeOptions,
                                           stream))) {
        return variantOp.emitOpError()
               << "failed to serialize VM bytecode module";
      }
    }
    if (!serOptions.dumpBinariesPath.empty()) {
      dumpDataToPath<char>(serOptions.dumpBinariesPath, serOptions.dumpBaseName,
                           variantOp.getName(), ".vmfb", moduleData);
    }

    auto bufferAttr = DenseIntElementsAttr::get(
        VectorType::get({static_cast<int64_t>(moduleData.size())},
                        IntegerType::get(executableBuilder.getContext(), 8)),
        std::move(moduleData));

    // Add the binary data to the target executable.
    // NOTE: this snapshots the FlatBuffer builder data at the time it is called
    // and future changes to the target op will not be observed.
    auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(), bufferAttr);
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();
  }

private:
  const VMVXOptions &options;
};

class VMVXInlineTargetDevice final : public TargetDevice {
public:
  VMVXInlineTargetDevice() = default;

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    auto configAttr = b.getDictionaryAttr(configItems);

    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("vmvx-inline")
        ->getDefaultExecutableTargets(context, "vmvx-inline", configAttr,
                                      executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context,
                                            b.getStringAttr("vmvx-inline"),
                                            configAttr, executableTargetAttrs);
  }
};

class VMVXInlineTargetBackend final : public TargetBackend {
public:
  VMVXInlineTargetBackend(const VMVXOptions &options) : options(options) {}

  std::string getLegacyDefaultDeviceID() const override {
    return "vmvx-inline";
  }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    executableTargetAttrs.push_back(getVMVXExecutableTarget(
        options.enableMicrokernels, context, "vmvx-inline", "vmvx-ir"));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::Codegen::IREECodegenDialect, IREE::VMVX::VMVXDialect>();
  }

  void buildConfigurationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                      OpPassManager &passManager) override {
    IREE::VMVX::buildVMVXConfigurationPassPipeline(passManager);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpPassManager &passManager) override {
    IREE::VMVX::buildVMVXTransformPassPipeline(passManager);
  }

private:
  const VMVXOptions &options;
};

namespace {
struct VMVXSession
    : public PluginSession<VMVXSession, VMVXOptions,
                           PluginActivationPolicy::DefaultActivated> {
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) {
    // TODO(benvanik): move to a CPU device registration outside of VMVX. Note
    // that the inline device does need to be special.
    // #hal.device.target<"vmvx", ...
    targets.add("vmvx", [&]() { return std::make_shared<VMVXTargetDevice>(); });
    // #hal.device.target<"vmvx-inline", ...
    targets.add("vmvx-inline",
                [&]() { return std::make_shared<VMVXInlineTargetDevice>(); });
  }
  void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) {
    // #hal.executable.target<"vmvx", ...
    targets.add("vmvx",
                [&]() { return std::make_shared<VMVXTargetBackend>(options); });
    // #hal.executable.target<"vmvx-inline", ...
    targets.add("vmvx-inline", [&]() {
      return std::make_shared<VMVXInlineTargetBackend>(options);
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL

extern "C" bool iree_register_compiler_plugin_hal_target_vmvx(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::iree_compiler::IREE::HAL::VMVXSession>(
      "hal_target_vmvx");
  return true;
}

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::IREE::HAL::VMVXOptions);
