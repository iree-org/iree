// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LinkerTool.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "llvm-linker"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Unix linker (ld-like); for ELF files.
class UnixLinkerTool : public LinkerTool {
 public:
  using LinkerTool::LinkerTool;

  std::string getSystemToolPath() const override {
    // First check for setting the linker explicitly.
    auto toolPath = LinkerTool::getSystemToolPath();
    if (!toolPath.empty()) return toolPath;

    // No explicit linker specified, search the environment for common tools.
    // We want LLD:
    // * On Apple, we want the system linker, which is named `ld`
    if (targetIsApple()) {
      // On macOS, the standard system linker is `ld`, and it's
      // unconditionally what we want to use.
      toolPath = findToolInEnvironment({"ld"});
    } else {
      // On Linux, the only linker basename that's standard is `ld` but it could
      // be any of ld.bfd, ld.gold, ld.lld, which are inequivalent in the way
      // explained in the comment below on the -shared flag. We specifically
      // want ld.lld here, however we still search for `ld` as a fallback name,
      // in case the linker would be ld.lld but would be installed only under
      // the name `ld`.
      //
      // Having `ld` as a fallback name also makes sense (at least
      // theoretically) on "generic Unix": `ld` is the standard name of the
      // system linker, and `-static -shared` should in theory be supported by
      // the system linker (as suggested by both the FreeBSD and GNU man pages
      // for ld).
      //
      // On the other hand, on Linux where the possible fallbacks are ld.bfd or
      // ld.gold, we are specifically not interested in falling back on any
      // of these, at least given current behavior.
      toolPath = findToolInEnvironment({"ld.lld", "ld"});
    }
    if (!toolPath.empty()) return toolPath;

    llvm::errs() << "No Unix linker tool found in environment.\n";
    return "";
  }

  std::optional<Artifacts> linkDynamicLibrary(
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
        getSystemToolPath(),
        "-o " + artifacts.libraryFile.path,
    };

    if (targetIsApple()) {
      // Statically link all dependencies so we don't have any runtime deps.
      // We cannot have any imports in the module we produce.
      flags.push_back("-static");

      // Produce a Mach-O dylib file.
      flags.push_back("-dylib");
      flags.push_back("-flat_namespace");
      flags.push_back(
          "-L /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib "
          "-lSystem");
    } else {
      // Avoids including any libc/startup files that initialize the CRT as
      // we don't use any of that. Our shared libraries must be freestanding.
      flags.push_back("-nostdlib");  // -nodefaultlibs + -nostartfiles

      // Statically link all dependencies so we don't have any runtime deps.
      // We cannot have any imports in the module we produce.
      flags.push_back("-static");

      // Generate a dynamic library (ELF type: ET_DYN), otherwise dlopen()
      // won't succeed on it. This is not incompatible with -static. The GNU
      // man page for ld, `man ld`, says the following:
      //
      //   -static
      //       Do not link against shared libraries. [...] This option can be
      //       used with -shared. Doing so means that a shared library is
      //       being created but that all of the library's external references
      //       must be resolved by pulling in entries from static libraries.
      //
      // While that much is said in the GNU ld man page, the reality is that
      // out of ld.bfd, ld.gold and ld.lld, only ld.lld actually implements
      // that. Meanwhile, ld.bfd interprets -static -shared as just -static,
      // and ld.gold rejects -static -shared outright as "incompatible".
      flags.push_back("-shared");
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
    if (failed(runLinkCommand(commandLine))) return std::nullopt;
    return artifacts;
  }

 private:
  bool targetIsApple() const {
    return targetTriple.isOSDarwin() || targetTriple.isiOS();
  }
};

std::unique_ptr<LinkerTool> createUnixLinkerTool(
    const llvm::Triple &targetTriple, LLVMTargetOptions &targetOptions) {
  return std::make_unique<UnixLinkerTool>(targetTriple, targetOptions);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
