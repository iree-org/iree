// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/Builtins/UKernel.h"

#include "iree/builtins/ukernel/libukernel.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

static const iree_file_toc_t *lookupUKernelFile(StringRef filename) {
  for (size_t i = 0; i < iree_builtins_libukernel_size(); ++i) {
    const auto &file_toc = iree_builtins_libukernel_create()[i];
    if (filename == file_toc.name) return &file_toc;
  }
  return nullptr;
}

static const iree_file_toc_t *lookupUKernelFile(
    llvm::TargetMachine *targetMachine) {
  const auto &triple = targetMachine->getTargetTriple();

  // NOTE: other arch-specific checks go here.

  // Fallback path using the generic wasm variants as they are largely
  // machine-agnostic.
  if (triple.isX86()) {
    return lookupUKernelFile("ukernel_bitcode.bc");
  } else {
    return nullptr;
  }
}

std::optional<std::unique_ptr<llvm::Module>> loadUKernelBitcode(
    llvm::TargetMachine *targetMachine, llvm::LLVMContext &context) {
  // Find a bitcode file for the current architecture.
  const auto *file = lookupUKernelFile(targetMachine);
  if (!file) {
    return std::nullopt;
  }

  // Load the generic bitcode file contents.
  llvm::MemoryBufferRef bitcodeBufferRef(
      llvm::StringRef(file->data, file->size), file->name);
  auto bitcodeFile = llvm::parseBitcodeFile(bitcodeBufferRef, context);
  if (!bitcodeFile) {
    // TODO: Do we want to error out here or silently proceed.
    return std::nullopt;
  }
  return std::move(*bitcodeFile);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
