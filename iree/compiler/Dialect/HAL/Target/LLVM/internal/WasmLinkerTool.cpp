// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
// Use with `-iree-llvm-target-triple=wasm32-unknown-unknown` (or equivalent).
// For SIMD support, also set `-iree-llvm-target-cpu-features=+simd128`.
class WasmLinkerTool : public LinkerTool {
 public:
  using LinkerTool::LinkerTool;

  std::string getSystemToolPath() const override {
    // First check for setting the linker explicitly.
    auto toolPath = LinkerTool::getSystemToolPath();
    if (!toolPath.empty()) return toolPath;

    // No explicit linker specified, search the environment for common tools.
    toolPath = findToolInEnvironment({"wasm-ld"});
    if (!toolPath.empty()) return toolPath;

    llvm::errs() << "No Wasm linker tool specified or discovered\n";
    return "";
  }

  LogicalResult configureModule(
      llvm::Module *llvmModule,
      ArrayRef<llvm::Function *> exportedFuncs) override {
    // https://lld.llvm.org/WebAssembly.html#exports
    // Note: once we can set --shared this shouldn't be needed, since we set
    // default visibility on exported functions.
    for (auto func : exportedFuncs) {
      func->addFnAttr("wasm-export-name", func->getName());
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
        getSystemToolPath(),

        // entry symbol not defined (pass --no-entry to suppress): _start
        "--no-entry",

        // Treat warnings as errors.
        "--fatal-warnings",

        // Generated a shared object, not an executable.
        // Note: disabled since creating shared libraries is not yet supported.
        // "--shared",

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
