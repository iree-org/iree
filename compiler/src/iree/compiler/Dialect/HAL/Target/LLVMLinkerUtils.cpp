// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVMLinkerUtils.h"

#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Transforms/IPO/Internalize.h"

namespace mlir::iree_compiler::IREE::HAL {

static llvm::cl::opt<std::string> clBitcodeFiles(
    "iree-link-bitcode",
    llvm::cl::desc(
        "Paths of additional bitcode files to load and link. Comma-separated. "
        "Any list entry that contains an equals (=) is parsed as `arch=path` "
        "and is only linked if `arch` matches the target triple."),
    llvm::cl::init(""));

bool anyRequiredSymbols(const llvm::Module &module, StringRef prefix) {
  for (const auto &function : module.functions()) {
    if (!function.isIntrinsic() && function.isDeclaration() &&
        (function.getName().starts_with(prefix))) {
      return true;
    }
  }
  return false;
}

LogicalResult linkBitcodeModule(
    Location loc, llvm::Linker &linker, unsigned linkerFlags,
    llvm::TargetMachine &targetMachine, StringRef name,
    llvm::Expected<std::unique_ptr<llvm::Module>> bitcodeModuleValue,
    ModuleSpecializationCallback specializationCallback) {
  // Ensure the bitcode loaded correctly. It may fail if the LLVM version is
  // incompatible.
  if (!bitcodeModuleValue) {
    return mlir::emitError(loc)
           << "failed to parse " << name
           << " bitcode: " << llvm::toString(bitcodeModuleValue.takeError())
           << " (possible LLVM bitcode incompatibility?)";
  }

  // Override the data layout and target triple with the final one we expect.
  // This is at the module level and if functions have their own specified
  // target attributes they won't be modified.
  auto bitcodeModule = std::move(bitcodeModuleValue.get());
  bitcodeModule->setDataLayout(targetMachine.createDataLayout());
  bitcodeModule->setTargetTriple(targetMachine.getTargetTriple().str());

  // Inject target-specific flags to specialize the bitcode prior to linking.
  if (specializationCallback) {
    specializationCallback(*bitcodeModule);
  }

  // Link the bitcode into the base module. This will merge in any required
  // symbols and override declarations that may exist.
  if (linker.linkInModule(
          std::move(bitcodeModule), linkerFlags,
          [&](llvm::Module &m, const StringSet<> &gvs) {
            if (linkerFlags & llvm::Linker::LinkOnlyNeeded) {
              llvm::internalizeModule(m, [&gvs](const llvm::GlobalValue &gv) {
                return !gv.hasName() || (gvs.count(gv.getName()) == 0);
              });
            }
          })) {
    return mlir::emitError(loc) << "failed to link " << name << " bitcode";
  }

  return success();
}

llvm::Expected<std::unique_ptr<llvm::Module>>
loadBitcodeObject(IREE::HAL::ExecutableObjectAttr objectAttr,
                  llvm::LLVMContext &context) {
  // Load the object data into memory.
  auto objectData = objectAttr.loadData();
  if (!objectData) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to load bitcode object file");
  }

  // Load the generic bitcode file contents.
  llvm::MemoryBufferRef bitcodeBufferRef(objectData.value(),
                                         objectAttr.getPath());
  auto bitcodeModuleValue = llvm::parseBitcodeFile(bitcodeBufferRef, context);
  if (!bitcodeModuleValue)
    return bitcodeModuleValue;
  // NOTE: at this point the bitcode may not have the expected data layout!
  return std::move(bitcodeModuleValue.get());
}

LogicalResult
linkBitcodeObjects(Location loc, llvm::Linker &linker, unsigned linkerFlags,
                   llvm::TargetMachine &targetMachine, ArrayAttr objectAttrs,
                   llvm::LLVMContext &context,
                   ModuleSpecializationCallback specializationCallback) {
  // Gather only the bitcode objects.
  SmallVector<IREE::HAL::ExecutableObjectAttr> bitcodeObjectAttrs;
  IREE::HAL::ExecutableObjectAttr::filterObjects(objectAttrs, {".bc"},
                                                 bitcodeObjectAttrs);

  // Load and link each object in the order declared.
  for (auto objectAttr : bitcodeObjectAttrs) {
    if (failed(linkBitcodeModule(
            loc, linker, linkerFlags, targetMachine, objectAttr.getPath(),
            loadBitcodeObject(objectAttr, context), specializationCallback))) {
      return mlir::emitError(loc)
             << "failed linking in user object bitcode `"
             << objectAttr.getPath() << "` for target triple '"
             << targetMachine.getTargetTriple().str() << "'";
    }
  }

  return success();
}

LogicalResult linkPathBitcodeFiles(Location loc, llvm::Linker &linker,
                                      unsigned linkerFlags,
                                      StringRef path,
                                      llvm::TargetMachine &targetMachine,
                                      llvm::LLVMContext &context) {
    auto bitcodeBufferRef = llvm::MemoryBuffer::getFile(path);
    if (auto ec = bitcodeBufferRef.getError()) {
      return mlir::emitError(loc) << "failed reading user bitcode file `"
                                  << path << "`: " << ec.message();
    }
    auto setAlwaysInline = [&](llvm::Module &module) {
      // ROCM/HIP builtin functions are non-inlinable.
      // if (targetMachine.getTargetTriple().isAMDGCN() || targetMachine.getTargetTriple().isAMDGPU()) return;
      for (auto &func : module.getFunctionList()) {
        if (func.hasFnAttribute(llvm::Attribute::NoInline)) {
          func.removeFnAttr(llvm::Attribute::NoInline);
        }
        func.addFnAttr(llvm::Attribute::AlwaysInline);
      }
    };
    if (failed(linkBitcodeModule(
            loc, linker, linkerFlags, targetMachine, path,
            llvm::parseBitcodeFile(*bitcodeBufferRef->get(), context),
            setAlwaysInline))) {
      return mlir::emitError(loc)
             << "failed linking in user bitcode file `" << path
             << "` for target triple '" << targetMachine.getTargetTriple().str()
             << "'";
    }

  return success();
}

LogicalResult linkCmdlineBitcodeFiles(Location loc, llvm::Linker &linker,
                                      unsigned linkerFlags,
                                      llvm::TargetMachine &targetMachine,
                                      llvm::LLVMContext &context) {
  if (clBitcodeFiles.empty()) {
    return success();
  }
  SmallVector<StringRef> entries;
  StringRef(clBitcodeFiles.getValue()).split(entries, ',');
  for (StringRef entry : entries) {
    StringRef path = entry;
    if (entry.contains('=')) {
      std::pair<StringRef, StringRef> components = entry.split('=');
      StringRef filterArch = components.first;
      const char *archName =
          getIreeArchNameForTargetTriple(targetMachine.getTargetTriple());
      if (filterArch != archName) {
        continue;
      }
      path = components.second;
    }
    if (failed(linkPathBitcodeFiles(loc, linker, linkerFlags, path, targetMachine, context))) {
      return mlir::emitError(loc) << "failed linking cmd line bit code.";
    }
  }
  return success();
}

} // namespace mlir::iree_compiler::IREE::HAL
