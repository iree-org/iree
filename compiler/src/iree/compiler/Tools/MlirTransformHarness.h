// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Harness for one-shot MLIR transformations via iree-bazel-try.
// Uses standard MLIR tool patterns for I/O, flag parsing, and pass pipelines.
//
// Basic transform (stdin -> transform -> stdout):
//   echo 'func.func @test() { return }' | iree-bazel-try -e '
//   #include "iree/compiler/Tools/MlirTransformHarness.h"
//   void myTransform(ModuleOp module) {
//     module.walk([](mlir::Operation *op) {
//       llvm::outs() << "Found: " << op->getName() << "\n";
//     });
//   }
//   MLIR_TRANSFORM_MAIN(myTransform)
//   '
//
// File input and output:
//   iree-bazel-try -e '...' -- input.mlir -o output.mlir
//
// Bytecode format support (auto-detected on input, explicit on output):
//   iree-bazel-try -e '...' -- input.mlirbc -o output.mlirbc --emit-bytecode
//
// Pass pipeline preprocessing:
//   echo 'func.func @f(%a: i32) { ... }' | iree-bazel-try -e '
//   #include "iree/compiler/Tools/MlirTransformHarness.h"
//   void analyze(ModuleOp m) {
//     // Analyze IR after canonicalization.
//     int ops = 0;
//     m.walk([&](Operation *op) { ops++; });
//     llvm::outs() << "Operations: " << ops << "\n";
//   }
//   MLIR_TRANSFORM_MAIN_NO_PRINT(analyze)
//   ' -- --pass-pipeline='builtin.module(func.func(canonicalize,cse))'
//
// Pattern application:
//   echo 'module { ... }' | iree-bazel-try -e '
//   #include "iree/compiler/Tools/MlirTransformHarness.h"
//   struct MyPattern : public mlir::OpRewritePattern<SomeOp> {
//     using OpRewritePattern::OpRewritePattern;
//     LogicalResult matchAndRewrite(SomeOp op, PatternRewriter &rw) const
//     override {
//       // ... pattern implementation
//       return success();
//     }
//   };
//   MLIR_PATTERN_MAIN(MyPattern)
//   '
//
// The harness handles:
// - Standard MLIR file I/O (supports stdin, .mlir, .mlirbc formats)
// - Flag parsing (--pass-pipeline, --emit-bytecode, -o, etc.)
// - Full IREE compiler dialect/pass registration (same as iree-opt)
// - Pass pipeline execution before user transform
// - Calling your transform function or applying patterns
// - Printing the result to stdout (unless NO_PRINT variant used)

#ifndef IREE_COMPILER_TOOLS_MLIRTRANSFORMHARNESS_H_
#define IREE_COMPILER_TOOLS_MLIRTRANSFORMHARNESS_H_

#include <functional>
#include <memory>
#include <string>

#include "iree/compiler/Tools/init_dialects.h"
#include "iree/compiler/Tools/init_passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

// Options for controlling transform harness behavior.
struct TransformHarnessOptions {
  // Print module to output after transform completes.
  bool printOutput = true;

  // Emit bytecode format instead of textual MLIR.
  bool emitBytecode = false;

  // Verify IR after passes (when pass pipeline is used).
  bool verifyEachPass = true;

  // Allow unregistered dialects in parsed IR.
  bool allowUnregisteredDialects = true;
};

// Main entry point for transform harness.
// Handles argument parsing, file I/O, pass pipeline execution, and user
// transform.
//
// Args:
//   argc, argv: Command line arguments (uses llvm::cl for parsing)
//   transformFn: User's transform function to apply to the module
//   options: Harness configuration options
//
// Returns: 0 on success, non-zero on failure
//
// Supported flags:
//   <input-file>                  Input MLIR file (- for stdin, auto-detects
//                                 .mlir/.mlirbc)
//   -o <file>                     Output file (- for stdout)
//   --pass-pipeline-before=<str>  Pass pipeline to run BEFORE transform
//   --pass-pipeline-after=<str>   Pass pipeline to run AFTER transform
//   --emit-bytecode               Emit bytecode instead of text
//   --verify-each                 Verify IR after each pass (default: true)
//   --allow-unregistered-dialect  Allow unregistered dialects (default: true)
inline int
runMlirTransformHarness(int argc, char **argv,
                        std::function<void(mlir::ModuleOp)> transformFn,
                        TransformHarnessOptions options = {}) {
  // Initialize LLVM (handles signal handlers, etc.).
  llvm::InitLLVM y(argc, argv);

  // Register command-line options (static ensures single initialization).
  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file or '-' for stdin>"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> passPipelineBefore(
      "pass-pipeline-before",
      llvm::cl::desc("Pass pipeline to run before transform"),
      llvm::cl::value_desc("pipeline"), llvm::cl::init(""));

  static llvm::cl::opt<std::string> passPipelineAfter(
      "pass-pipeline-after",
      llvm::cl::desc("Pass pipeline to run after transform"),
      llvm::cl::value_desc("pipeline"), llvm::cl::init(""));

  static llvm::cl::opt<bool> emitBytecode(
      "emit-bytecode", llvm::cl::desc("Emit bytecode instead of textual IR"),
      llvm::cl::init(false));

  static llvm::cl::opt<bool> verifyEach(
      "verify-each", llvm::cl::desc("Verify IR after each pass"),
      llvm::cl::init(true));

  static llvm::cl::opt<bool> allowUnregisteredDialect(
      "allow-unregistered-dialect",
      llvm::cl::desc("Allow unregistered dialects"), llvm::cl::init(true));

  // Parse command line options.
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "IREE Transform Harness - MLIR transformation experimentation tool\n");

  // Register all IREE dialects and passes (same as iree-opt).
  mlir::DialectRegistry registry;
  mlir::iree_compiler::registerAllDialects(registry);
  mlir::iree_compiler::registerAllPasses();

  // Create context with all registered dialects.
  mlir::MLIRContext context(registry);
  if (allowUnregisteredDialect || options.allowUnregisteredDialects) {
    context.allowUnregisteredDialects();
  }
  context.loadAllAvailableDialects();

  // Open input file (handles stdin via "-", auto-detects .mlir vs .mlirbc).
  std::string errorMessage;
  auto inputFile = mlir::openInputFile(inputFilename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << "Error opening input file: " << errorMessage << "\n";
    return 1;
  }

  // Create source manager with shared ownership (allows zero-copy buffer
  // references).
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(inputFile), llvm::SMLoc());

  // Parse input (auto-detects text vs bytecode format).
  mlir::ParserConfig parserConfig(&context);
  mlir::OwningOpRef<mlir::ModuleOp> moduleRef =
      mlir::parseSourceFile<mlir::ModuleOp>(*sourceMgr, parserConfig);

  if (!moduleRef) {
    llvm::errs() << "Failed to parse input file\n";
    return 1;
  }

  // Run pass pipeline before user transform if specified.
  if (!passPipelineBefore.getValue().empty()) {
    mlir::PassManager pm(&context);

    // Enable verification if requested.
    if (verifyEach || options.verifyEachPass) {
      pm.enableVerifier();
    }

    // Parse pipeline string (e.g., "builtin.module(canonicalize,cse)").
    if (failed(mlir::parsePassPipeline(passPipelineBefore, pm))) {
      llvm::errs() << "Failed to parse pass pipeline: " << passPipelineBefore
                   << "\n";
      return 1;
    }

    // Run the pipeline.
    if (failed(pm.run(*moduleRef))) {
      llvm::errs() << "Pass pipeline execution failed\n";
      return 1;
    }
  }

  // Run the user's transform function.
  try {
    transformFn(*moduleRef);
  } catch (const std::exception &e) {
    llvm::errs() << "Transform function threw exception: " << e.what() << "\n";
    return 1;
  }

  // Run pass pipeline after user transform if specified.
  if (!passPipelineAfter.getValue().empty()) {
    mlir::PassManager pm(&context);

    // Enable verification if requested.
    if (verifyEach || options.verifyEachPass) {
      pm.enableVerifier();
    }

    // Parse pipeline string (e.g., "builtin.module(canonicalize,cse)").
    if (failed(mlir::parsePassPipeline(passPipelineAfter, pm))) {
      llvm::errs() << "Failed to parse pass pipeline: " << passPipelineAfter
                   << "\n";
      return 1;
    }

    // Run the pipeline.
    if (failed(pm.run(*moduleRef))) {
      llvm::errs() << "Pass pipeline execution failed\n";
      return 1;
    }
  }

  // Write output if requested.
  if (options.printOutput) {
    auto outputFile = mlir::openOutputFile(outputFilename, &errorMessage);
    if (!outputFile) {
      llvm::errs() << "Error opening output file: " << errorMessage << "\n";
      return 1;
    }

    // Choose output format (bytecode or text).
    if (emitBytecode || options.emitBytecode) {
      mlir::BytecodeWriterConfig bytecodeConfig;
      if (failed(mlir::writeBytecodeToFile(*moduleRef, outputFile->os(),
                                           bytecodeConfig))) {
        llvm::errs() << "Failed to write bytecode output\n";
        return 1;
      }
    } else {
      // Text output with default formatting.
      moduleRef->print(outputFile->os());
    }

    // Keep output file only if write succeeded.
    outputFile->keep();
  }

  return 0;
}

// Pattern-based transform helper.
// Applies patterns using greedy rewrite driver.
//
// Args:
//   argc, argv: Command line arguments (uses llvm::cl for parsing)
//   populatePatterns: Function that adds patterns to the set
//   options: Harness configuration options
//
// Returns: 0 on success, non-zero on failure
inline int runMlirPatternHarness(
    int argc, char **argv,
    std::function<void(mlir::RewritePatternSet &)> populatePatterns,
    TransformHarnessOptions options = {}) {
  // Wrap pattern application in a transform function.
  return runMlirTransformHarness(
      argc, argv,
      [&](mlir::ModuleOp module) {
        mlir::RewritePatternSet patterns(module.getContext());
        populatePatterns(patterns);

        // Apply patterns greedily until fixpoint.
        (void)mlir::applyPatternsGreedily(module, std::move(patterns));
      },
      options);
}

} // namespace mlir::iree_compiler

// Convenience macro to define main() with output printing.
#define MLIR_TRANSFORM_MAIN(transformFn)                                       \
  int main(int argc, char **argv) {                                            \
    return mlir::iree_compiler::runMlirTransformHarness(argc, argv,            \
                                                        transformFn, {});      \
  }

// Convenience macro to define main() without output printing.
#define MLIR_TRANSFORM_MAIN_NO_PRINT(transformFn)                              \
  int main(int argc, char **argv) {                                            \
    return mlir::iree_compiler::runMlirTransformHarness(                       \
        argc, argv, transformFn, {.printOutput = false});                      \
  }

// Convenience macro for pattern-based transforms.
// Usage: MLIR_PATTERN_MAIN(MyPattern) or MLIR_PATTERN_MAIN(Pat1, Pat2, Pat3)
#define MLIR_PATTERN_MAIN(...)                                                 \
  int main(int argc, char **argv) {                                            \
    return mlir::iree_compiler::runMlirPatternHarness(                         \
        argc, argv,                                                            \
        [](mlir::RewritePatternSet &patterns) {                                \
          patterns.add<__VA_ARGS__>(patterns.getContext());                    \
        },                                                                     \
        {});                                                                   \
  }

// Convenience macro for pattern-based transforms without output printing.
#define MLIR_PATTERN_MAIN_NO_PRINT(...)                                        \
  int main(int argc, char **argv) {                                            \
    return mlir::iree_compiler::runMlirPatternHarness(                         \
        argc, argv,                                                            \
        [](mlir::RewritePatternSet &patterns) {                                \
          patterns.add<__VA_ARGS__>(patterns.getContext());                    \
        },                                                                     \
        {.printOutput = false});                                               \
  }

// Bring common namespaces into scope so user code looks like normal IREE code.
using namespace mlir;
using namespace mlir::iree_compiler;

#endif // IREE_COMPILER_TOOLS_MLIRTRANSFORMHARNESS_H_
