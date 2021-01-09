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

// Main entry-point for the XLA proto importer frontend.
// This will read from XLA proto files and produce MLIR MHLO assembly.

#include <fstream>
#include <iostream>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/FileUtilities.h"
#include "tensorflow/compiler/mlir/xla/hlo_to_mlir_hlo.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/core/platform/protobuf.h"

using namespace llvm;
using namespace mlir;

namespace {

enum XlaFormat {
  binary_proto,
  text_proto,
};

// Error collector that prints errors.
class PrintErrorCollector : public tensorflow::protobuf::io::ErrorCollector {
 public:
  PrintErrorCollector(std::string filePath) : filePath(std::move(filePath)) {}
  void AddError(int line, int column, const std::string &message) override {
    llvm::errs() << "Text protobuf parse error(" << filePath << ":" << line
                 << ":" << column << "): " << message << "\n";
    hadError = true;
  }

  std::string filePath;
  bool hadError = false;
};

class IStreamCopyingInputStream
    : public tensorflow::protobuf::io::CopyingInputStream {
 public:
  IStreamCopyingInputStream(std::istream *input) : input(input) {}
  int Read(void *buffer, int size) override {
    input->read(static_cast<char *>(buffer), size);
    if (input->fail()) return -1;
    return input->gcount();
  }

 private:
  std::istream *input;
};

}  // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  static cl::opt<std::string> inputPath(
      cl::Positional, cl::desc("<XLA Protocol Buffer Path>"), cl::Required);
  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));
  static llvm::cl::opt<std::string> saveTempIreeImport(
      "save-temp-iree-input",
      llvm::cl::desc("Save the resultant IR to this file (useful for saving an "
                     "intermediate in a pipeline)"),
      llvm::cl::init(""));
  static llvm::cl::opt<XlaFormat> inputFormat(
      "xla-format", cl::desc("XLA Format"),
      cl::values(clEnumVal(binary_proto, "Parse a binary protocol buffer"),
                 clEnumVal(text_proto, "Parse a text protocol buffer")));

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv);

  DialectRegistry registry;

  // Read the protocol buffer.
  std::ifstream fileInputStream;
  std::istream *inputStream;
  if (inputPath == "-") {
    inputStream = &std::cin;
  } else {
    fileInputStream.open(inputPath, std::ios::in | std::ios::binary);
    if (!fileInputStream.is_open()) {
      llvm::errs() << "Unable to open input file " << inputPath << "\n";
      return 1;
    }
    inputStream = &fileInputStream;
  }

  xla::HloProto hloProto;
  switch (inputFormat) {
    case binary_proto: {
      if (!hloProto.mutable_hlo_module()->ParseFromIstream(inputStream)) {
        llvm::errs() << "Could not parse binary protocol buffer from "
                     << inputPath << "\n";
        return 1;
      }
      break;
    }
    case text_proto: {
      tensorflow::protobuf::TextFormat::Parser parser;
      PrintErrorCollector collector(inputPath);
      IStreamCopyingInputStream copyingStream(inputStream);
      tensorflow::protobuf::io::CopyingInputStreamAdaptor streamAdaptor(
          &copyingStream);
      parser.RecordErrorsTo(&collector);
      parser.Parse(&streamAdaptor, hloProto.mutable_hlo_module());
      if (collector.hadError) {
        llvm::errs() << "Unable to parse text format protocol buffer\n";
        return 1;
      }
      break;
    }
    default:
      llvm_unreachable("illegal XlaFormat");
  }

  // Convert the Module proto into MLIR.
  MLIRContext context;
  OwningModuleRef module = ModuleOp::create(mlir::UnknownLoc::get(&context));
  registry.loadAll(&context);

  auto status =
      ConvertHloToMlirHlo(module.get(), hloProto.mutable_hlo_module());
  if (!status.ok()) {
    llvm::errs() << "Error converting HLO Module Proto to MLIR: "
                 << status.ToString() << "\n";
    return 2;
  }

  // Find the entry function an annotate it as exported.
  // Note that the XLA importer always produced an MLIR module with a @main
  // function.
  std::string entryName = "main";
  SymbolTable symbolTable(module.get());
  Operation *mainFunc = symbolTable.lookup(entryName);
  if (!mainFunc) {
    llvm::errs() << "Unable to find main function '" << entryName
                 << "' in converted module.\n";
    return 3;
  }
  mainFunc->setAttr("iree.module.export", UnitAttr::get(&context));

  // Save.
  auto saveToFile = [&](llvm::StringRef savePath) -> LogicalResult {
    auto outputFile = openOutputFile(savePath);
    if (!outputFile) {
      llvm::errs() << "Could not open output file: " << savePath << "\n";
      return failure();
    }
    OpPrintingFlags printFlags;
    printFlags.enableDebugInfo();
    printFlags.printGenericOpForm();
    module->print(outputFile->os(), printFlags);
    outputFile->os() << "\n";
    outputFile->keep();
    return success();
  };

  // Save temp output.
  if (!saveTempIreeImport.empty()) {
    if (failed(saveToFile(saveTempIreeImport))) return 10;
  }

  // Save output.
  if (failed(saveToFile(outputFilename))) return 3;
  return 0;
}
