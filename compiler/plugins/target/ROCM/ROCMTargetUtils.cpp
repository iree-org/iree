// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/ROCMTargetUtils.h"

#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/Utils/LLVMLinkerUtils.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::HAL {

static std::unique_ptr<llvm::Module>
loadIRModule(Location loc, const std::string &filename,
             llvm::LLVMContext *llvm_context) {
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module(
      llvm::parseIRFile(llvm::StringRef(filename.data(), filename.size()),
                        diagnostic, *llvm_context));

  if (!module) {
    mlir::emitError(loc) << "error loading HIP LLVM module: "
                         << diagnostic.getFilename().str() << ":"
                         << diagnostic.getLineNo() << ":"
                         << diagnostic.getColumnNo() << ": "
                         << diagnostic.getMessage().str();
    return {};
  }

  return module;
}

static LogicalResult linkWithBitcodeFiles(Location loc, llvm::Module *module,
                                          ArrayRef<std::string> bitcodePaths) {
  if (bitcodePaths.empty())
    return success();
  llvm::Linker linker(*module);
  for (auto &bitcodePath : bitcodePaths) {
    if (!llvm::sys::fs::exists(bitcodePath)) {
      return mlir::emitError(loc)
             << "AMD bitcode module is required by this module but was "
                "not found at "
             << bitcodePath;
    }
    std::unique_ptr<llvm::Module> bitcodeModule =
        loadIRModule(loc, bitcodePath, &module->getContext());
    if (!bitcodeModule)
      return failure();
    // Ignore the data layout of the module we're importing. This avoids a
    // warning from the linker.
    bitcodeModule->setDataLayout(module->getDataLayout());
    if (linker.linkInModule(
            std::move(bitcodeModule), llvm::Linker::Flags::LinkOnlyNeeded,
            [](llvm::Module &M, const llvm::StringSet<> &GVS) {
              llvm::internalizeModule(M, [&GVS](const llvm::GlobalValue &GV) {
                return !GV.hasName() || (GVS.count(GV.getName()) == 0);
              });
            })) {
      return mlir::emitError(loc) << "llvm link of AMD bitcode failed";
    }
  }
  return success();
}

static LogicalResult linkBitcodeFile(Location loc, llvm::Linker &linker,
                                     unsigned linkerFlags, StringRef path,
                                     llvm::TargetMachine &targetMachine,
                                     llvm::LLVMContext &context) {
  auto bitcodeBufferRef = llvm::MemoryBuffer::getFile(path);
  if (auto ec = bitcodeBufferRef.getError()) {
    return mlir::emitError(loc) << "failed reading user bitcode file `" << path
                                << "`: " << ec.message();
  }
  auto setAlwaysInline = [&](llvm::Module &module) {
    if (targetMachine.getTargetCPU().contains("gfx10") ||
        targetMachine.getTargetCPU().contains("gfx11")) {
      // some ROCM/HIP functions for gfx10 or gfx11 has accuracy issue if
      // inlined.
      return;
    }
    for (auto &func : module.getFunctionList()) {
      // Some ROCM/HIP builtin functions have Optnone and NoInline for default.
      if (targetMachine.getTargetTriple().isAMDGCN()) {
        if (func.hasFnAttribute(llvm::Attribute::OptimizeNone)) {
          func.removeFnAttr(llvm::Attribute::OptimizeNone);
        }
        if (targetMachine.getTargetTriple().isAMDGCN() &&
            func.hasFnAttribute(llvm::Attribute::NoInline)) {
          func.removeFnAttr(llvm::Attribute::NoInline);
        }
      }
      func.addFnAttr(llvm::Attribute::AlwaysInline);
    }
  };
  if (failed(linkBitcodeModule(
          loc, linker, linkerFlags, targetMachine, path,
          llvm::parseBitcodeFile(*bitcodeBufferRef->get(), context),
          setAlwaysInline))) {
    return mlir::emitError(loc) << "failed linking in user bitcode file `"
                                << path << "` for target triple '"
                                << targetMachine.getTargetTriple().str() << "'";
  }

  return success();
}

static std::vector<std::string> getUkernelPaths(StringRef enabledUkernelsStr,
                                                StringRef targetChip,
                                                StringRef bitcodePath) {
  std::vector<std::string> selectedUkernelNames;
  if (enabledUkernelsStr == "all") {
    const char *allUkernelNames[] = {"argmax"};
    size_t numUkernels = sizeof(allUkernelNames) / sizeof(allUkernelNames[0]);
    for (int i = 0; i < numUkernels; i++) {
      selectedUkernelNames.push_back(allUkernelNames[i]);
    }
  } else {
    while (!enabledUkernelsStr.empty()) {
      auto split = enabledUkernelsStr.split(',');
      selectedUkernelNames.push_back(split.first.str());
      enabledUkernelsStr = split.second;
    }
  }

  // Construct full path to ROCDL bitcode libraries.
  std::vector<std::string> result;
  std::string app = "/";
  for (auto &kernelName : selectedUkernelNames) {
    std::string filename =
        "rocm_" + kernelName + "_ukernel_" + targetChip.str();
    result.push_back(bitcodePath.str() + app + filename + ".bc");
  }
  return result;
}

static void overridePlatformGlobal(llvm::Module *module, StringRef globalName,
                                   uint32_t newValue, llvm::Type *globalTy) {
  // NOTE: the global will not be defined if it is not used in the module.
  auto *globalValue = module->getNamedGlobal(globalName);
  if (!globalValue)
    return;
  globalValue->setDSOLocal(true);
  globalValue->setConstant(true);
  globalValue->setInitializer(llvm::ConstantInt::get(
      globalValue->getValueType(),
      APInt(globalValue->getValueType()->getIntegerBitWidth(), newValue)));
}

LogicalResult setHIPGlobals(Location loc, llvm::Module *module,
                            StringRef targetChip) {
  // Link target chip ISA version as global.
  const int kLenOfChipPrefix = 3;
  StringRef chipId = targetChip.substr(kLenOfChipPrefix);
  int major = 0;
  int minor = 0;
  if (chipId.drop_back(2).getAsInteger(10, major))
    return failure();
  if (chipId.take_back(2).getAsInteger(16, minor))
    return failure();
  // Oldest GFX arch supported is gfx60x.
  if (major < 6)
    return failure();
  // Latest GFX arch supported is gfx115x.
  if (major > 11 || (major == 11 && minor > 0x5f))
    return failure();
  int chipCode = major * 1000 + minor;
  auto *int32Type = llvm::Type::getInt32Ty(module->getContext());
  overridePlatformGlobal(module, "__oclc_ISA_version", chipCode, int32Type);

  // Link oclc configurations as globals.
  auto *boolType = llvm::Type::getInt8Ty(module->getContext());
  static const std::vector<std::pair<std::string, bool>> rocdlGlobalParams(
      {{"__oclc_finite_only_opt", false},
       {"__oclc_daz_opt", false},
       {"__oclc_correctly_rounded_sqrt32", true},
       {"__oclc_unsafe_math_opt", false},
       {"__oclc_wavefrontsize64", true}});
  for (auto &globalParam : rocdlGlobalParams) {
    overridePlatformGlobal(module, globalParam.first, globalParam.second,
                           boolType);
  }

  return success();
}

LogicalResult linkHIPBitcodeIfNeeded(Location loc, llvm::Module *module,
                                     StringRef targetChip,
                                     StringRef bitcodePath) {
  bool usesOCML = false;
  bool usesOCKL = false;
  for (const llvm::Function &function : module->functions()) {
    if (!function.isIntrinsic() && function.isDeclaration()) {
      auto functionName = function.getName();
      if (functionName.starts_with("__ocml_"))
        usesOCML = true;
      else if (functionName.starts_with("__ockl_"))
        usesOCKL = true;
    }
  }

  // Link externally-provided bitcode files when used.
  SmallVector<std::string> bitcodePaths;
  if (usesOCML) {
    bitcodePaths.push_back(
        (bitcodePath + llvm::sys::path::get_separator() + "ocml.bc").str());
  }
  if (usesOCKL) {
    bitcodePaths.push_back(
        (bitcodePath + llvm::sys::path::get_separator() + "ockl.bc").str());
  }
  return linkWithBitcodeFiles(loc, module, bitcodePaths);
}

// Links optimized Ukernel bitcode into the given module if the module needs it.
LogicalResult linkUkernelBitcodeFiles(Location loc, llvm::Module *module,
                                      StringRef enabledUkernelsStr,
                                      StringRef targetChip,
                                      StringRef bitcodePath,
                                      unsigned linkerFlags,
                                      llvm::TargetMachine &targetMachine) {
  // Early exit if Ukernel not supported on target chip.
  if (!iree_compiler::hasUkernelSupportedRocmArch(targetChip)) {
    return mlir::emitError(loc)
           << "ukernel '" << enabledUkernelsStr
           << "' not supported on target chip: " << targetChip;
  }
  std::vector<std::string> ukernelPaths =
      getUkernelPaths(enabledUkernelsStr, targetChip, bitcodePath);
  llvm::Linker linker(*module);
  for (auto &path : ukernelPaths) {
    if (failed(linkBitcodeFile(loc, linker, linkerFlags, StringRef(path),
                               targetMachine, module->getContext())))
      return failure();
  }

  return success();
}

// Link object file using lld lnker to generate code object
// Inspiration from this section comes from LLVM-PROJECT-MLIR by
// ROCmSoftwarePlatform
// https://github.com/ROCmSoftwarePlatform/rocMLIR/blob/0ec7b2176308229ac05f1594f5b5019d58cd9e15/mlir/lib/ExecutionEngine/ROCm/BackendUtils.cpp
std::string createHsaco(Location loc, StringRef isa, StringRef name) {
  // Save the ISA binary to a temp file.
  int tempIsaBinaryFd = -1;
  SmallString<128> tempIsaBinaryFilename;
  std::error_code ec = llvm::sys::fs::createTemporaryFile(
      "kernel", "o", tempIsaBinaryFd, tempIsaBinaryFilename);
  if (ec) {
    emitError(loc) << "temporary file for ISA binary creation error";
    return {};
  }
  llvm::FileRemover cleanupIsaBinary(tempIsaBinaryFilename);
  llvm::raw_fd_ostream tempIsaBinaryOs(tempIsaBinaryFd, true);
  tempIsaBinaryOs << isa;
  tempIsaBinaryOs.close();

  // Create a temp file for HSA code object.
  int tempHsacoFD = -1;
  SmallString<128> tempHsacoFilename;
  ec = llvm::sys::fs::createTemporaryFile("kernel", "hsaco", tempHsacoFD,
                                          tempHsacoFilename);
  if (ec) {
    emitError(loc) << "temporary file for HSA code object creation error";
    return {};
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  // Invoke lld. Expect a true return value from lld.
  const SmallVector<std::string> toolNames = {"iree-lld", "lld"};
  std::string lldProgram = findTool(toolNames);
  if (lldProgram.empty()) {
    emitError(loc) << "unable to find iree-lld";
    return {};
  }
  SmallVector<StringRef> lldArgs{
      lldProgram,
      "-flavor",
      "gnu",
      "-shared",
      tempIsaBinaryFilename.str(),
      "-o",
      tempHsacoFilename.str(),
  };

  // Execute LLD.
  std::string errorMessage;
  int lldResult = llvm::sys::ExecuteAndWait(
      unescapeCommandLineComponent(lldProgram),
      ArrayRef<llvm::StringRef>(lldArgs), StringRef("LLD_VERSION=IREE"), {}, 0,
      0, &errorMessage);
  if (lldResult) {
    emitError(loc) << "iree-lld execute fail:" << errorMessage
                   << "Error Code:" << lldResult;
    return {};
  }

  // Load the HSA code object.
  std::unique_ptr<llvm::MemoryBuffer> hsacoFile =
      mlir::openInputFile(tempHsacoFilename);
  if (!hsacoFile) {
    emitError(loc) << "read HSA code object from temp file error";
    return {};
  }
  return hsacoFile->getBuffer().str();
}

} // namespace mlir::iree_compiler::IREE::HAL
