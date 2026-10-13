// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/MetalSPIRV/MSLToMetalLib.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-msl-to-metal-lib"

namespace mlir::iree_compiler::IREE::HAL {

std::optional<std::string> findMetalToolchain() {
  auto xcrun = llvm::sys::findProgramByName("xcrun");
  if (!xcrun) {
    LLVM_DEBUG(llvm::dbgs() << "Metal tools: xcrun not found in PATH\n");
    return std::nullopt;
  }

  // Verify both metal and metallib are reachable via xcrun.
  for (const char *tool : {"metal", "metallib"}) {
    llvm::StringRef args[] = {"xcrun", "--find", tool};
    std::optional<llvm::StringRef> redirects[] = {
        llvm::StringRef(""), llvm::StringRef(""), llvm::StringRef("")};
    if (llvm::sys::ExecuteAndWait(*xcrun, args, /*Env=*/std::nullopt,
                                  redirects, /*SecondsToWait=*/10) != 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Metal tools: 'xcrun --find " << tool << "' failed\n");
      return std::nullopt;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Metal tools: found at " << *xcrun << "\n");
  return *xcrun;
}

/// Runs a command, capturing stderr on failure.
static LogicalResult runCommand(llvm::StringRef program,
                                llvm::ArrayRef<llvm::StringRef> args,
                                std::string &errOutput) {
  llvm::SmallString<128> stderrFile;
  if (llvm::sys::fs::createTemporaryFile("metal-err", "log", stderrFile)) {
    errOutput = "failed to create temporary file for stderr capture";
    return failure();
  }
  llvm::FileRemover stderrRemover(stderrFile);

  std::optional<llvm::StringRef> redirects[] = {
      /*stdin=*/std::nullopt,
      /*stdout=*/std::nullopt,
      /*stderr=*/llvm::StringRef(stderrFile)};

  std::string execErr;
  int rc = llvm::sys::ExecuteAndWait(program, args, /*Env=*/std::nullopt,
                                     redirects, /*SecondsToWait=*/0,
                                     /*MemoryLimit=*/0, &execErr);
  if (rc != 0) {
    if (auto buf = llvm::MemoryBuffer::getFile(stderrFile))
      errOutput = (*buf)->getBuffer().str();
    else if (!execErr.empty())
      errOutput = execErr;
    else
      errOutput = "exited with code " + std::to_string(rc);
    return failure();
  }
  return success();
}

/// Compiles MSL source to metallib via xcrun (MSL -> AIR -> metallib).
static LogicalResult compileMetalShader(MetalTargetPlatform platform,
                                        llvm::StringRef mslFile,
                                        llvm::StringRef libFile,
                                        llvm::StringRef xcrunPath,
                                        std::string &errMsg) {
  const char *sdk = platform == MetalTargetPlatform::macOS     ? "macosx"
                    : platform == MetalTargetPlatform::iOS     ? "iphoneos"
                                                               : "iphonesimulator";

  llvm::SmallString<256> airFile(mslFile);
  llvm::sys::path::replace_extension(airFile, "air");
  llvm::FileRemover airRemover(airFile);

  // Step 1: MSL -> AIR
  llvm::StringRef metalArgs[] = {"xcrun", "-sdk", sdk,     "metal", "-c",
                                 mslFile, "-o",   airFile};
  if (failed(runCommand(xcrunPath, metalArgs, errMsg)))
    return failure();

  // Step 2: AIR -> metallib
  llvm::StringRef metallibArgs[] = {"xcrun", "-sdk", sdk, "metallib",
                                    airFile, "-o",   libFile};
  if (failed(runCommand(xcrunPath, metallibArgs, errMsg)))
    return failure();

  return success();
}

std::unique_ptr<llvm::MemoryBuffer>
compileMSLToMetalLib(MetalTargetPlatform targetPlatform,
                     llvm::StringRef mslCode, llvm::StringRef entryPoint,
                     llvm::StringRef xcrunPath, std::string &errMsg) {
  llvm::SmallString<32> mslFile, libFile;
  int mslFd = 0;
  llvm::sys::fs::createTemporaryFile(entryPoint, "metal", mslFd, mslFile);
  llvm::sys::fs::createTemporaryFile(entryPoint, "metallib", libFile);
  llvm::FileRemover mslRemover(mslFile.c_str());
  llvm::FileRemover libRemover(libFile.c_str());

  { // Write input MSL code to the temporary file.
    llvm::raw_fd_ostream inputStream(mslFd, /*shouldClose=*/true);
    inputStream << mslCode << "\n";
  }

  if (failed(compileMetalShader(targetPlatform, mslFile, libFile, xcrunPath,
                                errMsg)))
    return nullptr;

  auto fileOrErr = llvm::MemoryBuffer::getFile(libFile, /*IsText=*/false);
  if (std::error_code error = fileOrErr.getError()) {
    errMsg = "failed to read generated metallib: " + error.message();
    return nullptr;
  }

  return std::move(*fileOrErr);
}

} // namespace mlir::iree_compiler::IREE::HAL
