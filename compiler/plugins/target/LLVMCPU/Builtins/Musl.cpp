// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/LLVMCPU/Builtins/Musl.h"

#include "iree/builtins/musl/bin/libmusl.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::HAL {

static const iree_file_toc_t *lookupMuslFile(StringRef filename) {
  for (size_t i = 0; i < iree_builtins_libmusl_size(); ++i) {
    const auto &file_toc = iree_builtins_libmusl_create()[i];
    if (filename == file_toc.name)
      return &file_toc;
  }
  return nullptr;
}

static const iree_file_toc_t *
lookupMuslFile(llvm::TargetMachine *targetMachine) {
  const auto &triple = targetMachine->getTargetTriple();

  // NOTE: other arch-specific checks go here.

  // Fallback path using the generic wasm variants as they are largely
  // machine-agnostic.
  if (triple.isArch32Bit()) {
    return lookupMuslFile("libmusl_wasm32_generic.bc");
  } else if (triple.isArch64Bit()) {
    return lookupMuslFile("libmusl_wasm64_generic.bc");
  } else {
    return nullptr;
  }
}

llvm::Expected<std::unique_ptr<llvm::Module>>
loadMuslBitcode(llvm::TargetMachine *targetMachine,
                llvm::LLVMContext &context) {
  // Find a bitcode file for the current architecture.
  const auto *file = lookupMuslFile(targetMachine);
  if (!file) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no matching architecture bitcode file");
  }

  // Load the generic bitcode file contents.
  llvm::MemoryBufferRef bitcodeBufferRef(
      llvm::StringRef(file->data, file->size), file->name);
  return llvm::parseBitcodeFile(bitcodeBufferRef, context);
}

} // namespace mlir::iree_compiler::IREE::HAL
