// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/ROCMTargetUtils.h"

#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/Utils/LLVMLinkerUtils.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "llvm/ADT/StringSwitch.h"
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
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
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
    mlir::emitError(loc) << "error loading ROCM LLVM module: "
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
                                     unsigned linkerFlags, StringRef filename,
                                     StringRef contents,
                                     llvm::TargetMachine &targetMachine,
                                     llvm::LLVMContext &context) {
  llvm::MemoryBufferRef bitcodeBufferRef(contents, filename);
  auto setAlwaysInline = [&](llvm::Module &module) {
    for (auto &func : module.getFunctionList()) {
      func.addFnAttr(llvm::Attribute::AlwaysInline);
    }
  };
  if (failed(
          linkBitcodeModule(loc, linker, linkerFlags, targetMachine, filename,
                            llvm::parseBitcodeFile(bitcodeBufferRef, context),
                            setAlwaysInline))) {
    return mlir::emitError(loc) << "failed linking in user bitcode file `"
                                << filename << "` for target triple '"
                                << targetMachine.getTargetTriple().str() << "'";
  }

  return success();
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
                            const amdgpu::Chipset &chipset, bool isWave64,
                            uint32_t abiVersion) {
  // Oldest GFX arch supported is gfx60x.
  if (chipset.majorVersion < 6) {
    return emitError(loc, "pre-gfx6 chipsets are not supported");
  }
  // Latest GFX arch supported is gfx120x.
  if (chipset.majorVersion > 12 ||
      (chipset.majorVersion == 12 && chipset.minorVersion > 0)) {
    return emitError(loc)
           << "a chipset with major version = " << chipset.majorVersion
           << " and minor version = " << chipset.minorVersion
           << " was not known to exist at the time this IREE was built";
  }
  int chipCode = chipset.majorVersion * 1000 + chipset.minorVersion * 16 +
                 chipset.steppingVersion;
  auto *int32Type = llvm::Type::getInt32Ty(module->getContext());
  overridePlatformGlobal(module, "__oclc_ISA_version", chipCode, int32Type);

  overridePlatformGlobal(module, "__oclc_ABI_version", abiVersion, int32Type);

  // Link oclc configurations as globals.
  auto *boolType = llvm::Type::getInt8Ty(module->getContext());
  static const std::vector<std::pair<std::string, bool>> rocdlGlobalParams(
      {{"__oclc_finite_only_opt", false},
       {"__oclc_daz_opt", false},
       {"__oclc_correctly_rounded_sqrt32", true},
       {"__oclc_unsafe_math_opt", false}});
  for (auto &globalParam : rocdlGlobalParams) {
    overridePlatformGlobal(module, globalParam.first, globalParam.second,
                           boolType);
  }
  overridePlatformGlobal(module, "__oclc_wavefrontsize64", isWave64, boolType);

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
