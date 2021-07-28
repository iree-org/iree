// Copyright 2021 The IREE Authors
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

// Embedded ELF linker targeting IREE's ELF loader (or Android/Linux).
// This uses lld exclusively (though it can be overridden) as that lets us
// ensure we are consistently generating ELFs such that they can be used
// across our target platforms and with our loader.
//
// For consistency we follow the Linux ABI rules on all architectures and
// limit what we allow:
// - Triples of the form "{arch}-pc-linux-elf" only.
// - No builtin libraries are available.
// - Extra GNU-style symbol lookups are disabled (sysv only) to save binary
//   size. The loader does not use any hash tables but .hash is mandatory in
//   the spec and included for compatibility.
// - No lazy binding; all symbols must be resolved on load.
// - GNU_RELRO is optional but used here as we don't support lazy binding.
//
// We allow debug information to be included in the ELFs however we don't
// currently have a use for it at runtime. When unstripped we can possibly feed
// it to tools or use it ourselves to generate backtraces but since all release
// usage should be stripped nothing relies upon it.
class EmbeddedLinkerTool : public LinkerTool {
 public:
  using LinkerTool::LinkerTool;

  std::string getToolPath() const override {
    // Always try to use the tool specified for this exact configuration first.
    // Hopefully some day soon we'll be able to statically link LLD in and call
    // a C function to do the linking instead of needing a separate tool.
    if (!targetOptions.embeddedLinkerPath.empty()) {
      return targetOptions.embeddedLinkerPath;
    }

    // Fall back to check for setting the linker explicitly via environment
    // variables or flags. Users may do this to use their own lld with custom
    // architectures built in.
    auto toolPath = LinkerTool::getToolPath();
    if (!toolPath.empty()) return toolPath;

    // No explicit linker specified, search the environment for common tools.
    toolPath = findToolInEnvironment({"ld.lld"});
    if (!toolPath.empty()) return toolPath;

    llvm::errs() << "LLD (ld.lld) not found on path; specify with the "
                    "IREE_LLVMAOT_LINKER_PATH environment variable or "
                    "-iree-llvm-linker-path=\n";
    return "";
  }

  LogicalResult configureModule(
      llvm::Module *llvmModule,
      ArrayRef<llvm::Function *> exportedFuncs) override {
    for (auto &func : *llvmModule) {
      // -fno-plt - prevent PLT on calls to imports.
      func.addFnAttr("nonlazybind");
    }
    return success();
  }

  Optional<Artifacts> linkDynamicLibrary(
      StringRef libraryName, ArrayRef<Artifact> objectFiles) override {
    Artifacts artifacts;

    // Create the shared object name; if we only have a single input object we
    // can just reuse that.
    if (!objectFiles.empty()) {
      artifacts.libraryFile =
          Artifact::createVariant(objectFiles.front().path, "so");
    } else {
      artifacts.libraryFile = Artifact::createTemporary(libraryName, "so");
    }
    artifacts.libraryFile.close();

    SmallVector<std::string, 8> flags = {
        getToolPath(),

        // Forces LLD to act like gnu ld and produce ELF files.
        // If not specified then lld tries to figure out what it is by progname
        // (ld, ld64, link, etc).
        // NOTE: must be first because lld sniffs argv[1]/argv[2].
        "-flavor gnu",

        "-o " + artifacts.libraryFile.path,
    };

    // Avoids including any libc/startup files that initialize the CRT as
    // we don't use any of that. Our shared libraries must be freestanding.
    flags.push_back("-nostdlib");  // -nodefaultlibs + -nostartfiles

    // Statically link all dependencies so we don't have any runtime deps.
    // We cannot have any imports in the module we produce.
    flags.push_back("-static");

    // Creating a shared library.
    flags.push_back("-shared");

    // Drop unused sections.
    flags.push_back("--gc-sections");

    // Hardening (that also makes runtime linking easier):
    // - bind all import symbols during load
    // - make all relocations readonly.
    // See: https://blog.quarkslab.com/clang-hardening-cheat-sheet.html
    flags.push_back("-z now");
    flags.push_back("-z relro");

    // Strip local symbols; we only care about the global ones for lookup.
    // This shrinks the .symtab to a single entry.
    flags.push_back("--discard-all");

    // Use sysv .hash lookup table only; we have literally a single symbol and
    // the .gnu.hash overhead is not worth it (either in the ELF or in the
    // runtime loader).
    flags.push_back("--hash-style=sysv");

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
    if (failed(runLinkCommand(commandLine))) {
      // Ensure we save inputs if we fail so that the user can replicate the
      // command themselves.
      if (targetOptions.keepLinkerArtifacts) {
        for (auto &objectFile : objectFiles) {
          llvm::errs() << "linker input preserved: "
                       << objectFile.outputFile->getFilename();
          objectFile.outputFile->keep();
        }
      }
      return llvm::None;
    }
    return artifacts;
  }
};

std::unique_ptr<LinkerTool> createEmbeddedLinkerTool(
    llvm::Triple &targetTriple, LLVMTargetOptions &targetOptions) {
  return std::make_unique<EmbeddedLinkerTool>(targetTriple, targetOptions);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
