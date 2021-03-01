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
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "llvmaot-linker"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Wasm linker using wasm-ld for producing WebAssembly binaries.
// wasm-ld behaves like traditional ELF linkers and uses similar flags.
//
// For details on the linking process and file formats, see:
// * https://lld.llvm.org/WebAssembly.html
// * https://github.com/WebAssembly/tool-conventions/blob/master/Linking.md
//
// For more background on WebAssembly, see:
// * https://webassembly.org/
// * https://developer.mozilla.org/en-US/docs/WebAssembly
//
// When working with WebAssembly files, these projects are useful:
// * https://github.com/WebAssembly/wabt
// * https://github.com/bytecodealliance/wasmtime
//
// Use -iree-llvm-target-triple=wasm32-unknown-unknown.
class WasmLinkerTool : public LinkerTool {
 public:
  using LinkerTool::LinkerTool;

  std::string getToolPath() const override {
    // First check for setting the linker explicitly.
    auto toolPath = LinkerTool::getToolPath();
    if (!toolPath.empty()) return toolPath;

    // No explicit linker specified, search the environment for common tools.
    toolPath = findToolInEnvironment({"wasm-ld"});
    if (!toolPath.empty()) return toolPath;

    llvm::errs() << "No Wasm linker tool specified or discovered\n";
    return "";
  }

  LogicalResult configureModule(llvm::Module *llvmModule,
                                ArrayRef<StringRef> entryPointNames) override {
    for (auto &func : *llvmModule) {
      // Enable frame pointers to ensure that stack unwinding works.
      func.addFnAttr("frame-pointer", "all");

      // -ffreestanding-like behavior.
      func.addFnAttr("no-builtins");
    }

    // https://lld.llvm.org/WebAssembly.html#exports
    for (auto entryPointName : entryPointNames) {
      auto *entryPointFn = llvmModule->getFunction(entryPointName);
      entryPointFn->addFnAttr("wasm-export-name", entryPointName);
    }

    return success();
  }

  Optional<Artifacts> linkDynamicLibrary(
      StringRef libraryName, ArrayRef<Artifact> objectFiles) override {
    Artifacts artifacts;

    // Create the wasm binary file name; if we only have a single input object
    // we can just reuse that.
    if (objectFiles.size() == 1) {
      artifacts.libraryFile =
          Artifact::createVariant(objectFiles.front().path, "wasm");
    } else {
      artifacts.libraryFile = Artifact::createTemporary(libraryName, "wasm");
    }
    artifacts.libraryFile.close();

    SmallVector<std::string, 8> flags = {
        getToolPath(),

        // entry symbol not defined (pass --no-entry to suppress): _start
        "--no-entry",

        // Allow undefined symbols, provided by the runtime environment (?)
        // TODO(scotttodd): figure out how to avoid this.
        "--allow-undefined",

        // Treat warnings as errors.
        "--fatal-warnings",

        "-o " + artifacts.libraryFile.path,
    };

    // Strip debug information when not requested.
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

std::unique_ptr<LinkerTool> createWasmLinkerTool(
    llvm::Triple &targetTriple, LLVMTargetOptions &targetOptions) {
  return std::make_unique<WasmLinkerTool>(targetTriple, targetOptions);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
