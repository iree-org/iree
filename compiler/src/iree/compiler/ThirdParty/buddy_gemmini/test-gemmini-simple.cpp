//===- test-gemmini-simple.cpp - Simple test for Gemmini dialect ---------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "Gemmini/GemminiDialect.h"

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  
  // Parse command line arguments
  llvm::cl::ParseCommandLineOptions(argc, argv, 
      "Gemmini dialect test - verifies dialect loads correctly\n");

  mlir::MLIRContext context;
  
  // Register the Gemmini dialect
  context.getOrLoadDialect<buddy::gemmini::GemminiDialect>();
  
  // Also register commonly needed dialects
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  
  // Parse input file
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error parsing input file\n";
    return 1;
  }

  // Verify the module
  if (failed(mlir::verify(*module))) {
    llvm::errs() << "Module verification failed\n";
    return 1;
  }

  // Output the module
  std::string errorMessage;
  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  module->print(output->os());
  output->keep();

  llvm::outs() << "Successfully loaded and processed module with Gemmini dialect!\n";
  return 0;
}
