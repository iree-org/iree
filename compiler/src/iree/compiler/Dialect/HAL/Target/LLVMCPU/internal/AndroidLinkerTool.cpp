// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LinkerTool.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#define DEBUG_TYPE "llvm-linker"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

using llvm::Triple;

// Returns the canonical host name for the Android NDK prebuilt versions:
//   https://developer.android.com/ndk/guides/other_build_systems
//
// If we want to support self-built variants we'll need an env var (or just make
// the user set IREE_LLVM_SYSTEM_LINKER_PATH).
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
//
// Specifically, using the clang drivers specific to a target architecture and
// API version, e.g. aarch64-linux-android29-clang.
//
// Do we really need to bother using the NDK and using that specific clang
// driver? What if we just used a standard ld driver --- what if we just used
// UnixLinkerTool for Android too? At least when the host is Linux?
// At first glance, that seems to just work, but the following suggests to
// still bother with Android NDK clang drivers:
//
// While such drivers end up exec'ing just one standard clang driver, which in
// turn ends up exec'ing just one standard ld driver, at each stage some
// significant flags are passed, so it seems wise to rely on the NDK to set
// all these flags for us.
//
// Example: with strace, we find that as of NDK r23, the driver
//
//   android-ndk-r23/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang
//
// execs
//
//   android-ndk-r23/toolchains/llvm/prebuilt/linux-x86_64/bin/clang
//     --target=aarch64-linux-android29
//
// which in turn execs
//
//   android-ndk-r23/toolchains/llvm/prebuilt/linux-x86_64/bin/ld
//     -pie -z noexecstack -EL --fix-cortex-a53-843419
//     --warn-shared-textrel -z now -z relro -z max-page-size=4096
//     --hash-style=gnu --enable-new-dtags --eh-frame-hdr
//     -m aarch64linux -dynamic-linker /system/bin/linker64
//
// And that ld driver really is LLD:
//
//   $ android-ndk-r23/toolchains/llvm/prebuilt/linux-x86_64/bin/ld --version
//   LLD 12.0.5 (/buildbot/src/android/llvm-toolchain/out/llvm-project/lld ...
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

    SmallVector<std::string> flags = {
        getSystemToolPath(),

        // Avoids including any libc/startup files that initialize the CRT as
        // we don't use any of that. Our shared libraries must be freestanding.
        //
        // It matters that this flag isn't prefixed with --for-linker=. Doing so
        // results in a dlopen error: 'cannot locate symbol "main" referenced by
        // "iree_dylib_foo.so"'
        "-nostdlib",  // -nodefaultlibs + -nostartfiles

        "-o " + artifacts.libraryFile.path,
    };

    // Link all input objects. Note that we are not linking whole-archive as
    // we want to allow dropping of unused codegen outputs.
    for (auto &objectFile : objectFiles) {
      flags.push_back(objectFile.path);
    }

    // Since we are using a clang driver, we need to prefix the flags that are
    // meant to be only interpreted by the linker.
    SmallVector<std::string> flagsToPrefixForLinker = {
        // Statically link all dependencies so we don't have any runtime deps.
        // We cannot have any imports in the module we produce.
        "-static",

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
        //
        // So here we are effectively relying on the linker being ld.lld, which
        // is the case because we are using Android NDK clang, which execs
        // Android NDK ld, which is ld.lld, see strace results mentioned in
        // AndroidLinkerTool class comment.
        "-shared",

        // As seen in the strace results mentioned in the AndroidLinkerTool
        // class comment, when Android NDK clang execs ld, it passes -pie to it.
        // ld considers that to be incompatible with -shared (maybe simply
        // because -pie stands for position independent EXECUTABLE and -shared
        // means generate an ET_DYN, not an ET_EXEC?), so we have to pass
        // this flag now to undo -pie.
        "-no-pie",
    };

    // Strip debug information (only, no relocations) when not requested.
    if (!targetOptions.debugSymbols) {
      flagsToPrefixForLinker.push_back("--strip-debug");
    }

    // Prefix and fold flagsToPrefixForLinker into flags.
    for (const auto &f : flagsToPrefixForLinker) {
      flags.push_back("--for-linker=" + f);
    }
    flagsToPrefixForLinker.clear();

    auto commandLine = llvm::join(flags, " ");
    if (failed(runLinkCommand(commandLine))) return std::nullopt;
    return artifacts;
  }
};

std::unique_ptr<LinkerTool> createAndroidLinkerTool(
    const llvm::Triple &targetTriple, LLVMTargetOptions &targetOptions) {
  assert(targetTriple.isAndroid() &&
         "only use the AndroidLinkerTool for Android targets");
  return std::make_unique<AndroidLinkerTool>(targetTriple, targetOptions);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
