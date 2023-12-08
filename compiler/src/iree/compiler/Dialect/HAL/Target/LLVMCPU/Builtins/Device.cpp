// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/Builtins/Device.h"

#include "iree/builtins/device/libdevice_bitcode.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::HAL {

static const iree_file_toc_t *lookupDeviceFile(StringRef filename) {
  for (size_t i = 0; i < iree_builtins_libdevice_bitcode_size(); ++i) {
    const auto &file_toc = iree_builtins_libdevice_bitcode_create()[i];
    if (filename == file_toc.name)
      return &file_toc;
  }
  return nullptr;
}

static const iree_file_toc_t *
lookupDeviceFile(llvm::TargetMachine *targetMachine) {
  const auto &triple = targetMachine->getTargetTriple();

  // NOTE: other arch-specific checks go here.

  if (triple.isWasm()) {
    // TODO(benvanik): feature detect simd and such.
    // auto features = targetMachine->getTargetFeatureString();
    if (triple.isArch32Bit()) {
      return lookupDeviceFile("libdevice_wasm32_generic.bc");
    } else if (triple.isArch64Bit()) {
      return lookupDeviceFile("libdevice_wasm64_generic.bc");
    }
  }

  // Fallback path using the generic wasm variants as they are largely
  // machine-agnostic.
  if (triple.isArch32Bit()) {
    return lookupDeviceFile("libdevice_wasm32_generic.bc");
  } else if (triple.isArch64Bit() &&
             targetMachine->getTargetFeatureString().contains("+sme")) {
    return lookupDeviceFile("libdevice_device_aarch64_sme.bc");
  } else if (triple.isArch64Bit()) {
    return lookupDeviceFile("libdevice_wasm64_generic.bc");
  } else {
    return nullptr;
  }
}

llvm::Expected<std::unique_ptr<llvm::Module>>
loadDeviceBitcode(llvm::TargetMachine *targetMachine,
                  llvm::LLVMContext &context) {
  // Find a bitcode file for the current architecture.
  const auto *file = lookupDeviceFile(targetMachine);
  if (!file) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no matching architecture bitcode file");
  }

  // Load the generic bitcode file contents.
  llvm::MemoryBufferRef bitcodeBufferRef(
      llvm::StringRef(file->data, file->size), file->name);
  auto bitcodeModuleValue = llvm::parseBitcodeFile(bitcodeBufferRef, context);
  if (!bitcodeModuleValue)
    return bitcodeModuleValue;
  auto bitcodeModule = std::move(bitcodeModuleValue.get());

  // Clang adds its own per-function attributes that we need to strip so that
  // our current executable variant target is used instead.
  for (auto &func : bitcodeModule->functions()) {
    func.removeFnAttr("target-cpu");
    func.removeFnAttr("tune-cpu");
    func.removeFnAttr("target-features");
  }

  return std::move(bitcodeModule);
}

static void overridePlatformGlobal(llvm::Module &module, StringRef globalName,
                                   uint32_t newValue) {
  // NOTE: the global will not be defined if it is not used in the module.
  auto *globalValue = module.getNamedGlobal(globalName);
  if (!globalValue)
    return;
  globalValue->setLinkage(llvm::GlobalValue::PrivateLinkage);
  globalValue->setDSOLocal(true);
  globalValue->setConstant(true);
  globalValue->setInitializer(
      llvm::ConstantInt::get(globalValue->getValueType(), APInt(32, newValue)));
}

void specializeDeviceModule(IREE::HAL::ExecutableVariantOp variantOp,
                            llvm::Module &module,
                            llvm::TargetMachine &targetMachine) {
  overridePlatformGlobal(module, "libdevice_platform_example_flag", 0u);
}

} // namespace mlir::iree_compiler::IREE::HAL
