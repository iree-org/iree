// Copyright 2021 Google LLC
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

// RISCV linker (ld-like); for ELF files.
class RiscvLinkerTool : public LinkerTool {
 public:
  using LinkerTool::LinkerTool;

  std::string getToolPath() const override {
    // First check for setting the linker explicitly.
    auto toolPath = LinkerTool::getToolPath();
    if (!toolPath.empty()) return toolPath;

    char *riscvToolchainRoot = std::getenv("RISCV_TOOLCHAIN_ROOT");
    if (!riscvToolchainRoot) {
      llvm::errs() << "RISCV_TOOLCHAIN_ROOT environment variable must be set\n";
      return "";
    }

    // Select LLVM linker location based on riscv-gnu-toolchain install code
    // structure:
    // https://github.com/riscv/riscv-gnu-toolchain/blob/master/README.md
    return llvm::Twine(riscvToolchainRoot).concat("/bin/clang").str();
  }

  LogicalResult configureModule(
      llvm::Module *llvmModule,
      ArrayRef<llvm::Function *> exportedFuncs) override {
    for (auto &func : *llvmModule) {
      // Enable frame pointers to ensure that stack unwinding works.
      func.addFnAttr("frame-pointer", "all");

      // -ffreestanding-like behavior.
      func.addFnAttr("no-builtins");
    }
    return success();
  }

  Optional<Artifacts> linkDynamicLibrary(
      StringRef libraryName, ArrayRef<Artifact> objectFiles) override {
    Artifacts artifacts;

    // Currently the function mostly follows the typical Unix linker setting.
    // It should be possible to support other LLVM target triples with RISC-V
    // arch.

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

std::unique_ptr<LinkerTool> createRiscvLinkerTool(
    llvm::Triple &targetTriple, LLVMTargetOptions &targetOptions) {
  return std::make_unique<RiscvLinkerTool>(targetTriple, targetOptions);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
