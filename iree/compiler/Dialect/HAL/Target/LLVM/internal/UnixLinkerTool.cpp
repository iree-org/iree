// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVM/LinkerTool.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "llvmaot-linker"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Unix linker (ld-like); for ELF files.
class UnixLinkerTool : public LinkerTool {
 public:
  using LinkerTool::LinkerTool;

  std::string getToolPath() const override {
    // First check for setting the linker explicitly.
    auto toolPath = LinkerTool::getToolPath();
    if (!toolPath.empty()) return toolPath;

    // No explicit linker specified, search the environment for common tools.
    toolPath = findToolInEnvironment({"ld", "ld.gold", "ld.lld"});
    if (!toolPath.empty()) return toolPath;

    llvm::errs() << "No Unix linker tool specified or discovered\n";
    return "";
  }

  Optional<Artifacts> linkDynamicLibrary(
      StringRef libraryName, ArrayRef<Artifact> objectFiles) override {
    Artifacts artifacts;

    // Create the shared object name; if we only have a single input object we
    // can just reuse that.
    if (objectFiles.size() == 1) {
      artifacts.libraryFile =
          Artifact::createVariant(objectFiles.front().path, "so");
    } else {
      artifacts.libraryFile = Artifact::createTemporary(libraryName, "so");
    }
    artifacts.libraryFile.close();

    SmallVector<std::string, 8> flags = {
        getToolPath(),
        "-o " + artifacts.libraryFile.path,
    };

    if (targetTriple.isOSDarwin() || targetTriple.isiOS()) {
      // Statically link all dependencies so we don't have any runtime deps.
      // We cannot have any imports in the module we produce.
      flags.push_back("-static");

      // Produce a Mach-O dylib file.
      flags.push_back("-dylib");
      flags.push_back("-flat_namespace");
      flags.push_back(
          "-L /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib "
          "-lSystem");

      // HACK: we insert libm calls. This is *not good*.
      // Until the MLIR LLVM lowering paths no longer introduce these,
      // we are stuck with this.
      flags.push_back("-undefined suppress");
    } else {
      // Avoids including any libc/startup files that initialize the CRT as
      // we don't use any of that. Our shared libraries must be freestanding.
      flags.push_back("-nostdlib");  // -nodefaultlibs + -nostartfiles

      // Statically link all dependencies so we don't have any runtime deps.
      // We cannot have any imports in the module we produce.
      // flags.push_back("-static");

      // HACK: we insert mallocs and libm calls. This is *not good*.
      // We need hermetic binaries that pull in no imports; the MLIR LLVM
      // lowering paths introduce a bunch, though, so this is what we are
      // stuck with.
      flags.push_back("-shared");
      flags.push_back("-undefined suppress");
    }

    // Strip debug information (only, no relocations) when not requested.
    if (!targetOptions.debugSymbols) {
      flags.push_back("--strip-debug");
    }

    // Link all input objects. Note that we are not linking whole-archive as
    // we want to allow dropping of unused codegen outputs.
    for (auto &objectFile : objectFiles) {
      flags.push_back(objectFile.path);
    }

    auto commandLine = llvm::join(flags, " ");
    if (failed(runLinkCommand(commandLine))) return llvm::None;
    return artifacts;
  }
};

std::unique_ptr<LinkerTool> createUnixLinkerTool(
    llvm::Triple &targetTriple, LLVMTargetOptions &targetOptions) {
  return std::make_unique<UnixLinkerTool>(targetTriple, targetOptions);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
