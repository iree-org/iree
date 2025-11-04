// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Tools/init_dialects.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "iree/compiler/tool_entry_points_api.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace llvm;
using namespace mlir::iree_compiler;

#if defined(_MSC_VER)
#define fileno _fileno
#endif // _MSC_VER

// Parses and verifies the input MLIR file. Returns null on error.
static OwningOpRef<ModuleOp> loadModule(MLIRContext &context,
                                        StringRef inputFilename) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  ParserConfig config(&context);
  sourceMgr->AddNewSourceBuffer(std::move(file), SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, config);
}

// Writes a module to file with format detection based on flag and extension.
// |forceOutputAsBytecode| can be used to override detection.
static LogicalResult writeModuleToFile(ModuleOp module,
                                       StringRef outputFilename,
                                       bool forceOutputAsBytecode) {
  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Determine output format: explicit flag overrides auto-detection.
  bool outputAsBytecode =
      forceOutputAsBytecode || outputFilename.ends_with_insensitive(".mlirbc");
  if (outputAsBytecode) {
    BytecodeWriterConfig config;
    if (failed(writeBytecodeToFile(module, output->os(), config))) {
      llvm::errs() << "Failed to write bytecode output\n";
      return failure();
    }
  } else {
    module.print(output->os());
  }

  output->keep();
  return success();
}

// TODO(benvanik): split to a full-featured llvm-ar equivalent.
// In the future if this becomes a useful tool we can split out an iree-ar tool
// and make this just about linking (to match llvm-ar/llvm-link). It's not
// currently worth having another release binary that may be statically linked
// (and thus 100-200MB depending on platform) given that I don't know how much
// use it will see.
static LogicalResult performBundleCreation(MLIRContext &context,
                                           ArrayRef<std::string> linkModules,
                                           StringRef outputFilename,
                                           bool outputAsBytecode) {
  // Create anonymous outer module to hold all bundled modules.
  // If no modules are provided this creates an empty bundle.
  OwningOpRef<ModuleOp> bundleModuleOp =
      ModuleOp::create(UnknownLoc::get(&context));

  // Load and add each source module to the bundle.
  // Bundles just concatenate modules without resolving dependencies.
  OpBuilder builder(&context);
  for (const auto &modulePath : linkModules) {
    OwningOpRef<ModuleOp> sourceModuleOp = loadModule(context, modulePath);
    if (!sourceModuleOp) {
      llvm::errs() << "Failed to load module: " << modulePath << "\n";
      return failure();
    }
    if (auto nameAttr = sourceModuleOp->getSymNameAttr()) {
      // Named module - insert directly into the bundle.
      builder.setInsertionPointToEnd(bundleModuleOp->getBody());
      builder.insert(sourceModuleOp.release());
    } else {
      // Anonymous module - merge contents into bundle (handles symbol
      // conflicts). This is a slower path and large bundles should prefer to
      // name themselves and avoid potential conflicts.
      builder.setInsertionPointToEnd(bundleModuleOp->getBody());
      if (failed(mergeModuleInto(*sourceModuleOp, *bundleModuleOp, builder))) {
        return failure();
      }
    }
  }

  return writeModuleToFile(*bundleModuleOp, outputFilename, outputAsBytecode);
}

static LogicalResult performLinking(MLIRContext &context,
                                    StringRef inputFilename,
                                    ArrayRef<std::string> linkModules,
                                    ArrayRef<std::string> libraryPaths,
                                    StringRef outputFilename,
                                    bool outputAsBytecode, bool listSymbols) {
  // Load the main module.
  OwningOpRef<ModuleOp> moduleOp = loadModule(context, inputFilename);
  if (!moduleOp) {
    return failure();
  }

  // --list-symbols mode: just print symbols and exit.
  if (listSymbols) {
    llvm::outs() << "Public symbols in " << inputFilename << ":\n";
    for (auto funcOp : moduleOp->getOps<FunctionOpInterface>()) {
      if (!funcOp.isExternal() && funcOp.isPublic()) {
        llvm::outs() << "  @" << funcOp.getName() << "\n";
      }
    }
    return success();
  }

  // Configure pass options.
  IREE::Util::LinkModulesPassOptions options;
  for (const auto &module : linkModules) {
    options.linkModules.push_back(module);
  }
  for (const auto &path : libraryPaths) {
    options.libraryPaths.push_back(path);
  }

  // Create and run the linking pass.
  PassManager passManager(&context);
  passManager.addPass(IREE::Util::createLinkModulesPass(options));
  if (failed(passManager.run(*moduleOp))) {
    // Note that we should have printed a good error during the pass, this is
    // just in case and to ensure the last thing we print is a clear message (in
    // case a bunch of IR spew has scrolled the original error off screen).
    llvm::errs() << "Failed to link modules (likely an implementation bug)\n";
    return failure();
  }

  return writeModuleToFile(*moduleOp, outputFilename, outputAsBytecode);
}

extern "C" IREE_EMBED_EXPORTED int ireeLinkRunMain(int argc, char **argv) {
  cl::OptionCategory linkCategory("iree-link options");

  cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"),
                                     cl::value_desc("filename"), cl::init("-"),
                                     cl::cat(linkCategory));

  cl::opt<bool> createBundle("bundle",
                             cl::desc("Create a bundle from input modules"),
                             cl::init(false), cl::cat(linkCategory));

  cl::opt<bool> listSymbols(
      "list-symbols",
      cl::desc("List public symbols in the module and exit (no linking)."),
      cl::init(false), cl::cat(linkCategory));

  cl::list<std::string> linkModules(
      "link-module", cl::desc("Source module to link from (can repeat)."),
      cl::value_desc("filename"), cl::cat(linkCategory));

  cl::list<std::string> libraryPaths(
      "library-path",
      cl::desc("Directory to search for modules during automatic discovery "
               "(can repeat)."),
      cl::value_desc("path"), cl::cat(linkCategory));

  cl::opt<std::string> outputFilename(
      "o", cl::desc("Output filename for the linked module."),
      cl::value_desc("filename"), cl::init("-"), cl::cat(linkCategory));
  cl::opt<bool> outputAsBytecode(
      "output-bytecode",
      cl::desc("Force bytecode output (overrides auto-detection from .mlirbc "
               "extension)."),
      cl::init(false), cl::cat(linkCategory));

  cl::HideUnrelatedOptions(linkCategory);

  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "IREE module linker\n");

  // Initialize MLIR context and register all dialects.
  DialectRegistry registry;
  mlir::iree_compiler::registerAllDialects(registry);
  MLIRContext context(registry);

  // Eagerly load all registered dialects to avoid lazy loading during
  // multi-threaded pass execution. The linker may parse source modules
  // during pass execution which would otherwise trigger lazy dialect loading.
  context.loadAllAvailableDialects();

  // --bundle mode.
  if (createBundle) {
    if (failed(performBundleCreation(context, llvm::to_vector(linkModules),
                                     outputFilename, outputAsBytecode))) {
      return 1;
    }
    return 0;
  }

  // When reading from stdin and the input is a tty, it is often a user mistake
  // and the process "appears to be stuck". Print a message to let the user
  // know about it!
  if (inputFilename == "-" &&
      llvm::sys::Process::FileDescriptorIsDisplayed(fileno(stdin))) {
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";
  }

  if (failed(performLinking(context, inputFilename,
                            llvm::to_vector(linkModules),
                            llvm::to_vector(libraryPaths), outputFilename,
                            outputAsBytecode, listSymbols))) {
    return 1;
  }

  return 0;
}
