// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./ROCMTargetUtils.h"

#include "iree/compiler/Utils/ToolUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

//===========Link LLVM Module to ROCDL Start===================/
// Inspiration of code from this section comes from XLA Kernel Gen Project
// https://github.com/openxla/xla/blob/main/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc

bool couldNeedDeviceBitcode(const llvm::Module &module) {
  for (const llvm::Function &function : module.functions()) {
    // The list of prefixes should be in sync with library functions used in
    // target_util.cc.
    if (!function.isIntrinsic() && function.isDeclaration() &&
        (function.getName().startswith("__ocml_") ||
         function.getName().startswith("__ockl_"))) {
      return true;
    }
  }
  return false;
}

static void dieWithSMDiagnosticError(llvm::SMDiagnostic *diagnostic) {
  llvm::WithColor::error(llvm::errs())
      << diagnostic->getFilename().str() << ":" << diagnostic->getLineNo()
      << ":" << diagnostic->getColumnNo() << ": "
      << diagnostic->getMessage().str();
}

std::unique_ptr<llvm::Module> loadIRModule(const std::string &filename,
                                           llvm::LLVMContext *llvm_context) {
  llvm::SMDiagnostic diagnostic_err;
  std::unique_ptr<llvm::Module> module(
      llvm::parseIRFile(llvm::StringRef(filename.data(), filename.size()),
                        diagnostic_err, *llvm_context));

  if (module == nullptr) {
    dieWithSMDiagnosticError(&diagnostic_err);
  }

  return module;
}

LogicalResult
linkWithBitcodeVector(llvm::Module *module,
                      const std::vector<std::string> &bitcode_path_vector) {
  llvm::Linker linker(*module);

  for (auto &bitcode_path : bitcode_path_vector) {
    if (!(llvm::sys::fs::exists(bitcode_path))) {
      llvm::WithColor::error(llvm::errs())
          << "bitcode module is required by this HLO module but was "
             "not found at "
          << bitcode_path;
      return failure();
    }
    std::unique_ptr<llvm::Module> bitcode_module =
        loadIRModule(bitcode_path, &module->getContext());
    // Ignore the data layout of the module we're importing. This avoids a
    // warning from the linker.
    bitcode_module->setDataLayout(module->getDataLayout());
    if (linker.linkInModule(
            std::move(bitcode_module), llvm::Linker::Flags::LinkOnlyNeeded,
            [](llvm::Module &M, const llvm::StringSet<> &GVS) {
              llvm::internalizeModule(M, [&GVS](const llvm::GlobalValue &GV) {
                return !GV.hasName() || (GVS.count(GV.getName()) == 0);
              });
            })) {
      llvm::WithColor::error(llvm::errs()) << "Link Bitcode error.\n";
      return failure();
    }
  }
  return success();
}

static std::vector<std::string> getROCDLPaths(std::string targetChip,
                                              std::string bitCodeDir) {
  // AMDGPU bitcodes.
  static const std::vector<std::string> rocdlFilenames(
      {"opencl.bc", "ocml.bc", "ockl.bc"});

  // Construct full path to ROCDL bitcode libraries.
  std::vector<std::string> result;
  std::string app = "/";
  for (auto &filename : rocdlFilenames) {
    result.push_back(bitCodeDir + app + filename);
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

static LogicalResult linkModuleWithGlobal(llvm::Module *module,
                                          std::string &targetChip) {
  // Link target chip ISA version as global.
  const int kLenOfChipPrefix = 3;
  std::string chipId = targetChip.substr(kLenOfChipPrefix);
  // i.e gfx90a -> 9000 series.
  int chipArch = stoi(chipId.substr(0, chipId.length() - 1)) * 100;
  // Oldest GFX arch supported is gfx60x.
  if (chipArch < 6000)
    return failure();
  // Latest GFX arch supported is gfx115x.
  if (chipArch > 11000)
    return failure();
  // Get chip code from suffix. i.e gfx1103 -> `3`.
  // gfx90a -> `a` == `10`.
  // gfx90c -> `c` == `12`.
  std::string chipSuffix = chipId.substr(chipId.length() - 1);
  uint32_t chipCode;
  if (chipSuffix == "a") {
    chipCode = chipArch + 10;
  } else if (chipSuffix == "c") {
    chipCode = chipArch + 12;
  } else {
    if (!std::isdigit(chipSuffix[0]))
      return failure();
    chipCode = chipArch + stoi(chipSuffix);
  }
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

// Links ROCm-Device-Libs into the given module if the module needs it.
void linkROCDLIfNecessary(llvm::Module *module, std::string targetChip,
                          std::string bitCodeDir) {
  if (!couldNeedDeviceBitcode(*module)) {
    return;
  }
  if (!succeeded(HAL::linkWithBitcodeVector(
          module, getROCDLPaths(targetChip, bitCodeDir)))) {
    llvm::WithColor::error(llvm::errs()) << "Fail to Link ROCDL.\n";
  }
  if (!succeeded(HAL::linkModuleWithGlobal(module, targetChip))) {
    llvm::WithColor::error(llvm::errs()) << "Fail to Link with Globals.\n";
  };
}

//===========Link LLVM Module to ROCDL End===================/

//=====================Create HSACO Begin=============//
// Link object file using lld lnker to generate code object
// Inspiration from this section comes from LLVM-PROJECT-MLIR by
// ROCmSoftwarePlatform
// https://github.com/ROCmSoftwarePlatform/rocMLIR/blob/0ec7b2176308229ac05f1594f5b5019d58cd9e15/mlir/lib/ExecutionEngine/ROCm/BackendUtils.cpp
std::string createHsaco(Location loc, const std::string isa, StringRef name) {
  // Save the ISA binary to a temp file.
  int tempIsaBinaryFd = -1;
  SmallString<128> tempIsaBinaryFilename;
  std::error_code ec = llvm::sys::fs::createTemporaryFile(
      "kernel", "o", tempIsaBinaryFd, tempIsaBinaryFilename);
  if (ec) {
    mlir::emitError(loc) << "temporary file for ISA binary creation error";
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
    mlir::emitError(loc) << "temporary file for HSA code object creation error";
    return {};
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  // Invoke lld. Expect a true return value from lld.
  // Searching for LLD
  const SmallVector<std::string> &toolNames{"iree-lld", "lld"};
  std::string lldProgram = findTool(toolNames);
  if (lldProgram.empty()) {
    mlir::emitError(loc) << "unable to find iree-lld";
    return {};
  }
  std::vector<llvm::StringRef> lldArgs{
      lldProgram,
      llvm::StringRef("-flavor"),
      llvm::StringRef("gnu"),
      llvm::StringRef("-shared"),
      tempIsaBinaryFilename.str(),
      llvm::StringRef("-o"),
      tempHsacoFilename.str(),
  };

  // Executing LLD
  std::string errorMessage;
  int lldResult = llvm::sys::ExecuteAndWait(
      unescapeCommandLineComponent(lldProgram),
      llvm::ArrayRef<llvm::StringRef>(lldArgs),
      llvm::StringRef("LLD_VERSION=IREE"), {}, 0, 0, &errorMessage);
  if (lldResult) {
    mlir::emitError(loc) << "iree-lld execute fail:" << errorMessage
                         << "Error Code:" << lldResult;
    return {};
  }

  // Load the HSA code object.
  auto hsacoFile = mlir::openInputFile(tempHsacoFilename);
  if (!hsacoFile) {
    mlir::emitError(loc) << "read HSA code object from temp file error";
    return {};
  }
  std::string strHSACO(hsacoFile->getBuffer().begin(),
                       hsacoFile->getBuffer().end());
  return strHSACO;
}
//==============Create HSACO End=============//

} // namespace HAL
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
