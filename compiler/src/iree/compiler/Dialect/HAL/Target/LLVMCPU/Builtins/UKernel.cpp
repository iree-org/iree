// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/Builtins/UKernel.h"

#include "iree/builtins/ukernel/ukernel_bitcode.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

static std::unique_ptr<llvm::Module> loadUKernelBitcodeFile(
    StringRef filename, llvm::LLVMContext& context) {
  const iree_file_toc_t* file_start = iree_ukernel_bitcode_create();
  const iree_file_toc_t* file_end = file_start + iree_ukernel_bitcode_size();
  for (const iree_file_toc_t* file = file_start; file < file_end; ++file) {
    if (filename == file->name) {
      llvm::MemoryBufferRef bitcodeBufferRef(
          llvm::StringRef(file->data, file->size), file->name);
      auto bitcodeFile = llvm::parseBitcodeFile(bitcodeBufferRef, context);
      if (!bitcodeFile) return nullptr;
      return std::move(*bitcodeFile);
    }
  }
  return nullptr;
}

std::unique_ptr<llvm::Module> loadUKernelBaseBitcode(
    llvm::LLVMContext& context) {
  std::unique_ptr<llvm::Module> baseBitcode =
      loadUKernelBitcodeFile("ukernel_bitcode_base.bc", context);
  assert(baseBitcode && "base ukernel bitcode file not found");
  return baseBitcode;
}

std::unique_ptr<llvm::Module> loadUKernelArchBitcode(
    llvm::TargetMachine* targetMachine, llvm::LLVMContext& context) {
  const char* archName =
      getIreeArchNameForTargetTriple(targetMachine->getTargetTriple());
  char archBitcodeFilename[64];
  snprintf(archBitcodeFilename, sizeof archBitcodeFilename,
           "ukernel_bitcode_%s.bc", archName);
  std::unique_ptr<llvm::Module> archBitcode =
      loadUKernelBitcodeFile(archBitcodeFilename, context);
  // archBitcode is optional: we don't have arch-specific ukernel code for all
  // architectures. So it's normal to be returning nullptr here.
  return archBitcode;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
