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

#include "iree/base/target_platform.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LinkerTool.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
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
    auto toolPath = LinkerTool::getToolPath();
    if (!toolPath.empty()) return toolPath;

    if (targetTriple.isAndroid()) {
      char *androidNDKPath = std::getenv("ANDROID_NDK");
      if (!androidNDKPath) return toolPath;

      // Select prebuilt toolchain based on host architecture/platform:
      // https://developer.android.com/ndk/guides/other_build_systems
      std::string toolchains_binary_path;
#if defined(IREE_PLATFORM_LINUX) && defined(IREE_ARCH_X86_64)
      toolchains_binary_path = "/toolchains/llvm/prebuilt/linux-x86_64/bin/";
#elif defined(IREE_PLATFORM_APPLE) && defined(IREE_ARCH_X86_64)
      toolchains_binary_path = "/toolchains/llvm/prebuilt/darwin-x86_64/bin/";
#elif defined(IREE_PLATFORM_WINDOWS) && defined(IREE_ARCH_X86_32)
      toolchains_binary_path = "/toolchains/llvm/prebuilt/windows/bin/";
#elif defined(IREE_PLATFORM_WINDOWS) && defined(IREE_ARCH_X86_64)
      toolchains_binary_path = "/toolchains/llvm/prebuilt/windows-x86_64/bin/";
#else
      llvm::errs() << "Unknown architecture/platform combination"
                   << "\n";
      return "";
#endif  // IREE_PLATFORM_* && IREE_ARCH_*

      // TODO(ataei): Set target architecture and ABI from targetTriple.
      return llvm::Twine(androidNDKPath)
          .concat(toolchains_binary_path)
          .concat("aarch64-linux-android30-clang++")
          .str();
    }

// TODO(ataei, benvanik): Windows cross-linking discovery support.
#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_MACOS)
#define UNIX_SYS_LINKER_PATH_LENGTH 255
    auto sysLinkers = {"ld", "ld.gold", "lld.ld"};
    for (auto syslinker : sysLinkers) {
      FILE *pipe =
          popen(llvm::Twine("which ").concat(syslinker).str().c_str(), "r");
      char linkerPath[UNIX_SYS_LINKER_PATH_LENGTH];
      if (fgets(linkerPath, sizeof(linkerPath), pipe) != NULL) {
        return strtok(linkerPath, "\n");
      }
    }
    return toolPath;
#undef UNIX_SYS_LINKER_PATH_LENGTH
#else
    return toolPath;
#endif  // IREE_PLATFORM_LINUX || IREE_PLATFORM_MACOS
  }

  LogicalResult configureModule(llvm::Module *llvmModule,
                                ArrayRef<StringRef> entryPointNames) override {
    // Enable frame pointers to ensure that stack unwinding works, e.g. in
    // Tracy. In principle this could also be achieved by enabling unwind
    // tables, but we tried that and that didn't work in Tracy (which uses
    // libbacktrace), while enabling frame pointers worked.
    // https://github.com/google/iree/issues/3957
    for (auto &func : *llvmModule) {
      auto attrs = func.getAttributes();
      attrs = attrs.addAttribute(llvmModule->getContext(),
                                 llvm::AttributeList::FunctionIndex,
                                 "frame-pointer", "all");
      func.setAttributes(attrs);
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
#if defined(IREE_PLATFORM_MACOS)
      "-dylib",
      "-undefined suppress",
      "-flat_namespace",
#else
      "-shared",
#endif
      "-o " + artifacts.libraryFile.path,
    };

    if (targetTriple.isAndroid()) {
      flags.push_back("-static-libstdc++");
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
