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

static void removeTargetAttributes(llvm::Module &module) {
  // Copied from Device.cpp - TODO: move this to a shared utility.
  // Clang adds its own per-function attributes that we need to strip so that
  // our current executable variant target is used instead.
  for (auto &func : module.functions()) {
    func.removeFnAttr("target-cpu");
    func.removeFnAttr("tune-cpu");
    func.removeFnAttr("target-features");
  }
}

llvm::Expected<std::unique_ptr<llvm::Module>>
loadUKernelBaseBitcode(llvm::TargetMachine *targetMachine,
                       llvm::LLVMContext &context) {
  llvm::Triple triple = targetMachine->getTargetTriple();
  StringRef filename;
  if (triple.isArch64Bit()) {
    filename = "ukernel_bitcode_64bit_base.bc";
  } else if (triple.isArch32Bit()) {
    filename = "ukernel_bitcode_32bit_base.bc";
  } else {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Don't know what ukernel bitcode file to load.");
  }
  llvm::Expected<std::unique_ptr<llvm::Module>> bitcode =
      loadUKernelBitcodeFile(filename, context);
  if (!bitcode) {
    // Propagate the error to the caller.
    return bitcode;
  }

  if (!bitcode.get()) {
    // File not found. For base bitcode, this shouldn't happen.
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Base ukernel bitcode file not found: %s",
                                   filename.str().c_str());
  }

  // Base bitcode is compiled for any reasonable architecture of the right
  // bitness, as we don't care about anything else than bitness here.
  removeTargetAttributes(*bitcode.get());
  return bitcode;
}

llvm::Expected<std::unique_ptr<llvm::Module>>
loadUKernelArchEntryPointsBitcode(llvm::TargetMachine *targetMachine,
                                  llvm::LLVMContext &context) {
  const char *archName =
      getIreeArchNameForTargetTriple(targetMachine->getTargetTriple());
  char filename[64];
  snprintf(filename, sizeof filename, "ukernel_bitcode_%s_entry_points.bc",
           archName);
  llvm::Expected<std::unique_ptr<llvm::Module>> bitcode =
      loadUKernelBitcodeFile(filename, context);
  if (!bitcode) {
    // Propagate the error to the caller.
    return bitcode;
  }

  if (!bitcode.get()) {
    // File not found. This is normal: arch-specific bitcode is optional.
    return bitcode;
  }

  // Architecture entry-point functions should be inlinable into base (non-arch)
  // functions, so that their logic selecting specific "tile functions" can
  // evaluate at compile time based on constant argument values in the caller,
  // so that unused tile functions (e.g. for other data types, other CPU feature
  // variants, etc) get DCE'd. In order for these entry points to be inlinable,
  // they must have matching target attributes, so, just like we call
  // removeTargetAttributes in loadUKernelBaseBitcode, we need to do that also
  // here.
  removeTargetAttributes(*bitcode.get());
  return bitcode;
}

llvm::Expected<std::unique_ptr<llvm::Module>>
loadUKernelArchBitcode(llvm::TargetMachine *targetMachine,
                       llvm::LLVMContext &context) {
  const char *archName =
      getIreeArchNameForTargetTriple(targetMachine->getTargetTriple());
  char filename[64];
  snprintf(filename, sizeof filename, "ukernel_bitcode_%s.bc", archName);
  return loadUKernelBitcodeFile(filename, context);
}

} // namespace mlir::iree_compiler::IREE::HAL
