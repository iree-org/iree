// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVM/Builtins/LibMMT4D.h"

#include "iree/builtins/libmmt4d/bin/libmmt4d.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

static const iree_file_toc_t *lookupMMT4DFile(StringRef filename) {
  for (size_t i = 0; i < iree_builtins_libmmt4d_size(); ++i) {
    const auto &file_toc = iree_builtins_libmmt4d_create()[i];
    if (filename == file_toc.name) return &file_toc;
  }
  return nullptr;
}

static const iree_file_toc_t *lookupMMT4DFile(
    llvm::TargetMachine *targetMachine) {
  const auto &triple = targetMachine->getTargetTriple();
  auto features = targetMachine->getTargetFeatureString();

  if (triple.isAArch64()) {
    if (features.contains_insensitive("dotprod")) {
      return lookupMMT4DFile("libmmt4d_aarch64_dotprod.bc");
    } else {
      return lookupMMT4DFile("libmmt4d_aarch64_generic.bc");
    }
  } else if (triple.isWasm()) {
    // TODO(benvanik): feature detect simd.
    if (triple.isArch32Bit()) {
      return lookupMMT4DFile("libmmt4d_wasm32_generic.bc");
    } else if (triple.isArch64Bit()) {
      return lookupMMT4DFile("libmmt4d_wasm64_generic.bc");
    }
  }

  // Fallback path:
  if (triple.isArch32Bit()) {
    return lookupMMT4DFile("libmmt4d_wasm32_generic.bc");
  } else if (triple.isArch64Bit()) {
    return lookupMMT4DFile("libmmt4d_wasm64_generic.bc");
  } else {
    return nullptr;
  }
}

llvm::Expected<std::unique_ptr<llvm::Module>> loadMMT4DBitcode(
    llvm::TargetMachine *targetMachine, llvm::LLVMContext &context) {
  // Find a bitcode file for the current architecture.
  const auto *file = lookupMMT4DFile(targetMachine);
  if (!file) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no matching architecture bitcode file");
  }

  // Load the generic bitcode file contents.
  llvm::MemoryBufferRef bitcodeBufferRef(
      llvm::StringRef(file->data, file->size), file->name);
  auto bitcodeModuleValue = llvm::parseBitcodeFile(bitcodeBufferRef, context);
  if (!bitcodeModuleValue) return bitcodeModuleValue;
  auto bitcodeModule = std::move(bitcodeModuleValue.get());
  bitcodeModule->setDataLayout(targetMachine->createDataLayout());
  bitcodeModule->setTargetTriple(targetMachine->getTargetTriple().str());
  return bitcodeModule;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
