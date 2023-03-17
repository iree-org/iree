// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/MetalSPIRV/MSLToMetalLib.h"

#include <stdlib.h>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "iree-msl-to-metal-lib"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

/// Returns the command to compile the given MSL source file into Metal library.
static std::string getMetalCompileCommand(StringRef mslFile,
                                          StringRef libFile) {
  return llvm::Twine("xcrun -sdk macosx metal -c ")
      .concat(mslFile)
      .concat(" -o - | xcrun -sdk macosx metallib - -o ")
      .concat(libFile)
      .str();
}

/// Returns the given command via system shell.
static LogicalResult runSystemCommand(StringRef command) {
  LLVM_DEBUG(llvm::dbgs() << "Running system command: '" << command << "'\n");
  int exitCode = system(command.data());
  if (exitCode == 0) return success();
  llvm::errs() << "Failed to run system command '" << command
               << "' with error code: " << exitCode << "\n";
  return failure();
}

std::unique_ptr<llvm::MemoryBuffer> compileMSLToMetalLib(StringRef mslCode,
                                                         StringRef entryPoint) {
  SmallString<32> mslFile, airFile, libFile;
  int mslFd = 0;
  llvm::sys::fs::createTemporaryFile(entryPoint, "metal", mslFd, mslFile);
  llvm::sys::fs::createTemporaryFile(entryPoint, "metallib", libFile);
  llvm::FileRemover mslRemover(mslFile.c_str());
  llvm::FileRemover libRemover(libFile.c_str());

  {  // Write input MSL code to the temporary file.
    llvm::raw_fd_ostream inputStream(mslFd, /*shouldClose=*/true);
    inputStream << mslCode << "\n";
  }

  std::string command = getMetalCompileCommand(mslFile, libFile);
  if (failed(runSystemCommand(command))) return nullptr;

  auto fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(libFile, /*isText=*/false);
  if (std::error_code error = fileOrErr.getError()) {
    llvm::errs() << "Failed to open generated metallib file '" << libFile
                 << "' with error: " << error.message();
    return nullptr;
  }

  return std::move(*fileOrErr);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
