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
  SmallVector<NamedAttribute> config;
  config.emplace_back(
      StringAttr::get(context, "ukernels"),
      StringAttr::get(context, enableMicrokernels ? "all" : "none"));
  return IREE::HAL::ExecutableTargetAttr::get(
      context, StringAttr::get(context, backend),
      StringAttr::get(context, format), DictionaryAttr::get(context, config));
}

class VMVXTargetBackend final : public TargetBackend {
public:
  VMVXTargetBackend(const VMVXOptions &options) : options(options) {}

  std::string name() const override { return "vmvx"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect, IREE::VM::VMDialect,
                    IREE::VMVX::VMVXDialect,
                    IREE::LinalgExt::IREELinalgExtDialect>();
  }

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(b.getStringAttr("executable_targets"),
                             getExecutableTargets(context));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }

  std::optional<IREE::HAL::DeviceTargetAttr>
  getHostDeviceTarget(MLIRContext *context) const override {
    return getDefaultDeviceTarget(context);
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
  ArrayAttr getExecutableTargets(MLIRContext *context) const {
    SmallVector<Attribute> targetAttrs;
    // This is where we would multiversion.
    targetAttrs.push_back(getVMVXExecutableTarget(
        options.enableMicrokernels, context, "vmvx", "vmvx-bytecode-fb"));
    return ArrayAttr::get(context, targetAttrs);
  }

  const VMVXOptions &options;
};

class VMVXInlineTargetBackend final : public TargetBackend {
public:
  VMVXInlineTargetBackend(const VMVXOptions &options) : options(options) {}

  std::string name() const override { return "vmvx-inline"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::Codegen::IREECodegenDialect, IREE::VMVX::VMVXDialect>();
  }

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(b.getStringAttr("executable_targets"),
                             getExecutableTargets(context));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
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
  ArrayAttr getExecutableTargets(MLIRContext *context) const {
    SmallVector<Attribute> targetAttrs;
    // This is where we would multiversion.
    targetAttrs.push_back(getVMVXExecutableTarget(
        options.enableMicrokernels, context, "vmvx-inline", "vmvx-ir"));
    return ArrayAttr::get(context, targetAttrs);
  }

  const VMVXOptions &options;
};

namespace {
struct VMVXSession
    : public PluginSession<VMVXSession, VMVXOptions,
                           PluginActivationPolicy::DefaultActivated> {
  void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) {
    // #hal.device.target<"vmvx", ...
    // #hal.executable.target<"vmvx", ...
    targets.add("vmvx",
                [&]() { return std::make_shared<VMVXTargetBackend>(options); });
    // #hal.device.target<"vmvx-inline", ...
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
