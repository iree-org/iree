// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVM/LinkerTool.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "llvm-linker"

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
// Use with `-iree-llvmcpu-target-triple=wasm32-unknown-unknown` (or
// equivalent). For SIMD support, also set
// `-iree-llvmcpu-target-cpu-features=+simd128`.
class WasmLinkerTool : public LinkerTool {
 public:
  using LinkerTool::LinkerTool;

  std::string getWasmToolPath() const {
    // Always use the --iree-llvmcpu-wasm-linker-path flag when specified as
    // it's explicitly telling us what to use.
    if (!targetOptions.wasmLinkerPath.empty()) {
      return targetOptions.wasmLinkerPath;
    }

    // Allow overriding the automatic search with an environment variable.
    char *linkerPath = std::getenv("IREE_LLVM_WASM_LINKER_PATH");
    if (linkerPath) {
      return std::string(linkerPath);
    }

    // No explicit linker specified, search the environment (i.e. our own build
    // or install directories) for common tools.
    std::string toolPath = findToolFromExecutableDir(
        {"wasm-ld", "iree-lld", "lld", "ld.lld", "lld-link"});
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
        getWasmToolPath(),

        // Forces LLD to act like wasm ld and produce WebAssembly files.
        // If not specified then lld tries to figure out what it is by progname
        // (ld, ld64, link, etc).
        // NOTE: must be first because lld sniffs argv[1]/argv[2].
        "-flavor wasm",

        // entry symbol not defined (pass --no-entry to suppress): _start
        "--no-entry",

        // Treat warnings as errors.
        "--fatal-warnings",

        // Generated a shared object containing position-independent-code.
        "--experimental-pic",
        "--shared",

        // Import [shared] memory from the environment.
        // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WebAssembly/Memory#creating_a_shared_memory
        // TODO(scotttodd): Add a flag controlling these - some combination is
        //   required when using multithreading + SharedArrayBuffer, but they
        //   must be left off when running single threaded.
        // "--import-memory",
        // "--shared-memory",

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
    if (failed(runLinkCommand(commandLine))) return std::nullopt;
    return artifacts;
  }
};

std::unique_ptr<LinkerTool> createWasmLinkerTool(
    const llvm::Triple &targetTriple, LLVMTargetOptions &targetOptions) {
  return std::make_unique<WasmLinkerTool>(targetTriple, targetOptions);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
