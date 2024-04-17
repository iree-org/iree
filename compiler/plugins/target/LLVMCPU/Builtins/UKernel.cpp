// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/LLVMCPU/Builtins/UKernel.h"

#include "iree/builtins/ukernel/ukernel_bitcode.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::HAL {

// Note: even if the llvm::Expected status is successful, the enclosed pointer
// may still be null, indicating that the file was not found.
static llvm::Expected<std::unique_ptr<llvm::Module>>
loadUKernelBitcodeFile(StringRef filename, llvm::LLVMContext &context) {
  const iree_file_toc_t *file_start = iree_ukernel_bitcode_create();
  const iree_file_toc_t *file_end = file_start + iree_ukernel_bitcode_size();
  for (const iree_file_toc_t *file = file_start; file < file_end; ++file) {
    if (filename == file->name) {
      llvm::MemoryBufferRef bitcodeBufferRef(
          llvm::StringRef(file->data, file->size), file->name);
      return llvm::parseBitcodeFile(bitcodeBufferRef, context);
    }
  }

  // Some bitcode files are optional: we don't have arch-specific ukernel code
  // for all architectures. So it's normal to be returning nullptr here.
  return nullptr;
}

llvm::Expected<std::unique_ptr<llvm::Module>>
loadUKernelBitcode(llvm::TargetMachine *targetMachine,
                   llvm::LLVMContext &context) {
  const char *archName =
      getIreeArchNameForTargetTriple(targetMachine->getTargetTriple());
  std::string filename = std::string("ukernel_bitcode_") + archName + ".bc";
  llvm::Expected<std::unique_ptr<llvm::Module>> module =
      loadUKernelBitcodeFile(filename, context);
  if (!module) {
    // Error. Propagate to the caller.
    return module;
  }
  if (!module.get()) {
    // File not found. Just means that we don't have bitcode for that
    // architecture. Return the null module as a success case.
    return module;
  }
  // Ukernels rely fundamentally on always getting inlined, for their logic
  // to specialize at compile time, including specialization for a specific
  // combination of data types, a specific SIMD ISA variant, etc. Then all the
  // unused code paths can get DCE'd. That's why failure to inline a ukernel
  // can result in a large penalty in both performance and code size.
  for (auto &func : module.get()->functions()) {
    func.addFnAttr(llvm::Attribute::AlwaysInline);
  }
  return module;
}

} // namespace mlir::iree_compiler::IREE::HAL
