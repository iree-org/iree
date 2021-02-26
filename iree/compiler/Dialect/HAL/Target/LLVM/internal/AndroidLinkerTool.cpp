// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
// the user set IREE_LLVMAOT_LINKER_PATH).
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

  std::string getToolPath() const override {
    auto toolPath = LinkerTool::getToolPath();
    if (!toolPath.empty()) return toolPath;

    // ANDROID_NDK must be set for us to infer the tool path.
    char *androidNDKPath = std::getenv("ANDROID_NDK");
    if (!androidNDKPath) return toolPath;

    // Extract the Android version from the `android30` like triple piece.
    unsigned androidEnv[3];
    targetTriple.getEnvironmentVersion(androidEnv[0], androidEnv[1],
                                       androidEnv[2]);
    unsigned androidVersion = androidEnv[0];  // like '30'

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

  LogicalResult configureModule(llvm::Module *llvmModule,
                                ArrayRef<StringRef> entryPointNames) override {
    for (auto &func : *llvmModule) {
      // Enable frame pointers to ensure that stack unwinding works, e.g. in
      // Tracy. In principle this could also be achieved by enabling unwind
      // tables, but we tried that and that didn't work in Tracy (which uses
      // libbacktrace), while enabling frame pointers worked.
      // https://github.com/google/iree/issues/3957
      func.addFnAttr("frame-pointer", "all");

      // -ffreestanding-like behavior.
      func.addFnAttr("no-builtins");
    }

    // TODO(benvanik): switch to executable libraries w/ internal functions.
    for (auto entryPointName : entryPointNames) {
      auto *entryPointFn = llvmModule->getFunction(entryPointName);
      entryPointFn->setLinkage(
          llvm::GlobalValue::LinkageTypes::ExternalLinkage);
      entryPointFn->setVisibility(
          llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
    }

    return success();
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

        // Avoids including any libc/startup files that initialize the CRT as
        // we don't use any of that. Our shared libraries must be freestanding.
        "-nostdlib",  // -nodefaultlibs + -nostartfiles

        // Statically link all dependencies so we don't have any runtime deps.
        // We cannot have any imports in the module we produce.
        // "-static",

        // HACK: we insert mallocs and junk. This is *not good*.
        // We should be statically linking and not require anything from libc.
        "-shared",
        "-lc",

        // Currently we are emitting calls to libm (expf, cosf, etc). We ideally
        // should not be doing this - those functions are all generally terrible
        // and indicate some extremely non-optimal code paths.
        "-lm",

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
