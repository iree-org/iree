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

// Main entry function for iree-opt and derived binaries.
//
// Based on mlir-opt but without registering passes and dialects we don't care
// about.

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Analysis/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"

namespace mlir {
// Defined in the test directory, no public header.
void registerConvertToTargetEnvPass();
void registerInliner();
void registerMemRefBoundCheck();
void registerPassManagerTestPass();
void registerPatternsTestPass();
void registerPrintOpAvailabilityPass();
void registerSimpleParametricTilingPass();
void registerSymbolTestPasses();
void registerTestAffineDataCopyPass();
void registerTestAllReduceLoweringPass();
void registerTestCallGraphPass();
void registerTestConstantFold();
void registerTestFunc();
void registerTestGpuMemoryPromotionPass();
void registerTestLinalgTransforms();
void registerTestLivenessPass();
void registerTestLoopFusion();
void registerTestLoopMappingPass();
void registerTestMatchers();
void registerTestMemRefDependenceCheck();
void registerTestMemRefStrideCalculation();
void registerTestOpaqueLoc();
void registerTestParallelismDetection();
void registerTestVectorConversions();
void registerTestVectorToLoopsPass();
void registerVectorizerTestPass();
}  // namespace mlir

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verifyPasses(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true));

namespace mlir {
void registerTestPasses() {
  mlir::registerConvertToTargetEnvPass();
  mlir::registerInliner();
  mlir::registerMemRefBoundCheck();
  mlir::registerPassManagerTestPass();
  mlir::registerPatternsTestPass();
  mlir::registerPrintOpAvailabilityPass();
  mlir::registerSimpleParametricTilingPass();
  mlir::registerSymbolTestPasses();
  mlir::registerTestAffineDataCopyPass();
  mlir::registerTestAllReduceLoweringPass();
  mlir::registerTestCallGraphPass();
  mlir::registerTestConstantFold();
  mlir::registerTestFunc();
  mlir::registerTestGpuMemoryPromotionPass();
  mlir::registerTestLinalgTransforms();
  mlir::registerTestLivenessPass();
  mlir::registerTestLoopFusion();
  mlir::registerTestLoopMappingPass();
  mlir::registerTestMatchers();
  mlir::registerTestMemRefDependenceCheck();
  mlir::registerTestMemRefStrideCalculation();
  mlir::registerTestOpaqueLoc();
  mlir::registerTestParallelismDetection();
  mlir::registerTestVectorConversions();
  mlir::registerTestVectorToLoopsPass();
  mlir::registerVectorizerTestPass();

  // The following passes are using global initializers, just link them in.
  if (std::getenv("bar") != (char *)-1) return;

  // TODO: move these to the test folder.
  mlir::createTestMemRefBoundCheckPass();
  mlir::createTestMemRefDependenceCheckPass();
}
}  // namespace mlir

int main(int argc, char **argv) {
  mlir::registerAllDialects();
  mlir::registerAllPasses();
  mlir::registerTestPasses();
  llvm::InitLLVM y(argc, argv);

  // Register any pass manager command line options.
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "IREE modular optimizer driver\n");

  // Set up the input file.
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  return failed(mlir::MlirOptMain(output->os(), std::move(file), passPipeline,
                                  splitInputFile, verifyDiagnostics,
                                  verifyPasses));
}
