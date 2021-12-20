// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVM/LinkerTool.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Host.h"

#define DEBUG_TYPE "llvmaot-linker"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

using llvm::Triple;

// Returns the canonical host name for the Android NDK prebuilt versions:
//   https://developer.android.com/ndk/guides/other_build_systems
//
// If we want to support self-built variants we'll need an env var (or just make
// the user set IREE_LLVMAOT_SYSTEM_LINKER_PATH).
static const char *getNDKHostPlatform() {
  auto hostTriple = Triple(llvm::sys::getProcessTriple());
  if (hostTriple.isOSLinux() && hostTriple.getArch() == Triple::x86_64) {
    return "linux-x86_64";
  } else if (hostTriple.isMacOSX() && hostTriple.getArch() == Triple::x86_64) {
    return "darwin-x86_64";
  } else if (hostTriple.isOSWindows() &&
             hostTriple.getArch() == Triple::x86_64) {
    return "windows-x86_64";
  } else if (hostTriple.isOSWindows() && hostTriple.getArch() == Triple::x86) {
    return "windows";
  } else {
    llvm::errs()
        << "No (known) Android NDK prebuilt name for this host platform ('"
        << hostTriple.str() << "')";
    return "";
  }
}

// Returns the canonical target name for the Android NDK prebuilt versions.
static const char *getNDKTargetPlatform(const Triple &targetTriple) {
  switch (targetTriple.getArch()) {
    case Triple::arm:
      return "armv7a";
    case Triple::aarch64:
      return "aarch64";
    case Triple::x86:
      return "i686";
    case Triple::x86_64:
      return "x86_64";
    default:
      llvm::errs()
          << "No (known) Android NDK prebuilt name for this target platform ('"
          << targetTriple.str() << "')";
      return "";
  }
}

// Android linker using the Android NDK toolchain.
class AndroidLinkerTool : public LinkerTool {
 public:
  using LinkerTool::LinkerTool;

  std::string getSystemToolPath() const override {
    auto toolPath = LinkerTool::getSystemToolPath();
    if (!toolPath.empty()) return toolPath;

    // ANDROID_NDK must be set for us to infer the tool path.
    char *androidNDKPath = std::getenv("ANDROID_NDK");
    if (!androidNDKPath) {
      llvm::errs() << "ANDROID_NDK environment variable must be set\n";
      return "";
    }

    // Extract the Android version from the `android30` like triple piece.
    llvm::VersionTuple androidEnv = targetTriple.getEnvironmentVersion();
    unsigned androidVersion = androidEnv.getMajor();  // like '30'

    // Select prebuilt toolchain based on both host and target
    // architecture/platform:
    //   https://developer.android.com/ndk/guides/other_build_systems
    return llvm::Twine(androidNDKPath)
        .concat("/toolchains/llvm/prebuilt/")
        .concat(getNDKHostPlatform())
        .concat("/bin/")
        .concat(getNDKTargetPlatform(targetTriple))
        .concat("-linux-android")
        .concat(std::to_string(androidVersion))
        .concat("-clang")
        .str();
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
        getSystemToolPath(),

        // Avoids including any libc/startup files that initialize the CRT as
        // we don't use any of that. Our shared libraries must be freestanding.
        "-nostdlib",  // -nodefaultlibs + -nostartfiles

        // Statically link all dependencies so we don't have any runtime deps.
        // We cannot have any imports in the module we produce.
        "-static",

        "-o " + artifacts.libraryFile.path,
    };

    // Strip debug information (only, no relocations) when not requested.
    if (!targetOptions.debugSymbols) {
      flags.push_back("-Wl,--strip-debug");
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

std::unique_ptr<LinkerTool> createAndroidLinkerTool(
    Triple &targetTriple, LLVMTargetOptions &targetOptions) {
  assert(targetTriple.isAndroid() &&
         "only use the AndroidLinkerTool for Android targets");
  return std::make_unique<AndroidLinkerTool>(targetTriple, targetOptions);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
