// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Main entry-point for the XLA proto importer frontend.
// This will read from XLA proto files and produce MLIR MHLO assembly.

#include <fstream>
#include <iostream>

#include "iree_tf_compiler/MHLO/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mhlo/IR/register.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "tensorflow/core/platform/protobuf.h"

using namespace llvm;
using namespace mlir;

namespace {

enum XlaFormat {
  binary_proto,
  text_proto,
  hlo_text,
  mlir_text,
};

enum class OutputFormat {
  none,
  mlir_ir,
  mlir_bytecode,
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

LogicalResult ReadHloTextFormatFromStream(std::istream *in,
                                          xla::HloModuleProto *moduleProto) {
  std::string contents(std::istreambuf_iterator<char>(*in), {});
  if (in->fail()) {
    llvm::errs() << "Error reading input stream\n";
    return failure();
  }
  auto moduleOr = xla::ParseAndReturnUnverifiedModule(contents);
  if (!moduleOr.ok()) {
    llvm::errs() << "XLA failed to parse a text format HloModule:\n"
                 << moduleOr.status().ToString() << "\n";
    return failure();
  }

  auto module = std::move(*moduleOr);
  *moduleProto = module->ToProto();
  return success();
}

}  // namespace

int main(int argc, char **argv) {
  llvm::setBugReportMsg(
      "Please report issues to https://github.com/openxla/iree/issues and "
      "include the crash backtrace.\n");
  llvm::InitLLVM y(argc, argv);

  static cl::opt<std::string> inputPath(
      cl::Positional, cl::desc("<XLA Protocol Buffer Path>"), cl::Required);
  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));
  static llvm::cl::opt<std::string> saveTempMhloInput(
      "save-temp-mhlo-input",
      llvm::cl::desc("Save the MHLO pipeline input IR to this file"),
      llvm::cl::init(""));
  static llvm::cl::opt<std::string> saveTempIreeImport(
      "save-temp-iree-input",
      llvm::cl::desc("Save the resultant IR to this file (useful for saving an "
                     "intermediate in a pipeline)"),
      llvm::cl::init(""));
  static llvm::cl::opt<XlaFormat> inputFormat(
      "xla-format", cl::desc("XLA Format"),
      cl::values(
          clEnumVal(binary_proto, "Parse a binary protocol buffer"),
          clEnumVal(text_proto, "Parse a text protocol buffer"),
          clEnumVal(hlo_text, "Parse an HLO module in its native text format"),
          clEnumVal(mlir_text, "Parse MLIR text containing MHLO ops")));

  // The output format flag is the master control for what we do with the
  // in-memory compiled form.
  llvm::cl::opt<OutputFormat> outputFormat(
      "output-format", llvm::cl::desc("Format of imported output"),
      llvm::cl::values(clEnumValN(OutputFormat::mlir_bytecode, "mlir-bytecode",
                                  "MLIR Bytecode (default)"),
                       clEnumValN(OutputFormat::mlir_ir, "mlir-ir", "MLIR IR")),
      llvm::cl::init(OutputFormat::mlir_bytecode));

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv);

  auto openInputStream =
      [&]() -> std::optional<
                std::pair<std::istream *, std::unique_ptr<std::ifstream>>> {
    auto fileInputStream = std::make_unique<std::ifstream>();
    std::istream *inputStream;
    if (inputPath == "-") {
      inputStream = &std::cin;
    } else {
      fileInputStream->open(inputPath, std::ios::in | std::ios::binary);
      if (!fileInputStream->is_open()) {
        llvm::errs() << "Unable to open input file " << inputPath << "\n";
        return std::nullopt;
      }
      inputStream = fileInputStream.get();
    }
    return std::make_pair(inputStream, std::move(fileInputStream));
  };

  DialectRegistry registry;
  mlir::mhlo::registerAllMhloDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::math::MathDialect>();
  MLIRContext context;
  OwningOpRef<mlir::ModuleOp> module =
      ModuleOp::create(mlir::UnknownLoc::get(&context));
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  auto loadHloProtoIntoModule = [&](xla::HloProto &hloProto) -> LogicalResult {
    auto status =
        ConvertHloToMlirHlo(module.get(), hloProto.mutable_hlo_module());
    if (!status.ok()) {
      llvm::errs() << "Error converting HLO Module Proto to MLIR: "
                   << status.ToString() << "\n";
      return failure();
    }
    return success();
  };

  switch (inputFormat) {
    case binary_proto: {
      xla::HloProto hloProto;
      auto input = openInputStream();
      if (!input) {
        return 1;
      }
      if (!hloProto.mutable_hlo_module()->ParseFromIstream(input->first)) {
        llvm::errs() << "Could not parse binary protocol buffer from "
                     << inputPath << "\n";
        return 1;
      }
      if (failed(loadHloProtoIntoModule(hloProto))) return 2;
      break;
    }
    case text_proto: {
      xla::HloProto hloProto;
      auto input = openInputStream();
      if (!input) {
        return 1;
      }
      tensorflow::protobuf::TextFormat::Parser parser;
      PrintErrorCollector collector(inputPath);
      IStreamCopyingInputStream copyingStream(input->first);
      tensorflow::protobuf::io::CopyingInputStreamAdaptor streamAdaptor(
          &copyingStream);
      parser.RecordErrorsTo(&collector);
      parser.Parse(&streamAdaptor, hloProto.mutable_hlo_module());
      if (collector.hadError) {
        llvm::errs() << "Unable to parse text format protocol buffer\n";
        return 1;
      }
      if (failed(loadHloProtoIntoModule(hloProto))) return 2;
      break;
    }
    case hlo_text: {
      xla::HloProto hloProto;
      auto input = openInputStream();
      if (!input) {
        return 1;
      }
      if (failed(ReadHloTextFormatFromStream(input->first,
                                             hloProto.mutable_hlo_module()))) {
        return 1;
      }
      if (failed(loadHloProtoIntoModule(hloProto))) return 2;
      break;
    }
    case mlir_text: {
      std::string errorMessage;
      auto file = openInputFile(inputPath, &errorMessage);
      if (!file) {
        llvm::errs() << errorMessage << "\n";
        return 1;
      }
      sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
      module = parseSourceFile<ModuleOp>(sourceMgr, &context);
      if (!module) return 2;
      break;
    }
    default:
      assert(false && "illegal XlaFormat");
  }

  // Find the entry function and annotate it as exported.
  // Note that the XLA importer always produced an MLIR module with a @main
  // function.
  std::string entryName = "main";
  SymbolTable symbolTable(module.get());
  auto mainFunc = symbolTable.lookup<func::FuncOp>(entryName);
  if (!mainFunc) {
    llvm::errs() << "Unable to find main function '" << entryName
                 << "' in converted module.\n";
    return 3;
  }
  mainFunc.setPublic();

  // Save.
  auto saveToFile = [&](llvm::StringRef savePath) -> LogicalResult {
    auto outputFile = openOutputFile(savePath);
    if (!outputFile) {
      llvm::errs() << "Could not open output file: " << savePath << "\n";
      return failure();
    }

    if (outputFormat == OutputFormat::mlir_ir) {
      OpPrintingFlags printFlags;
      module->print(outputFile->os(), printFlags);
      outputFile->os() << "\n";
      outputFile->keep();
      return success();
    }

    if (outputFormat == OutputFormat::mlir_bytecode) {
      mlir::writeBytecodeToFile(*module, outputFile->os());
      outputFile->keep();
      return success();
    }
    llvm::errs() << "Unknown output format\n";
    return failure();
  };

  // Save temp output.
  if (!saveTempMhloInput.empty()) {
    if (failed(saveToFile(saveTempMhloInput))) return 10;
  }

  // Run passes.
  PassManager pm(&context, module.get()->getName().getStringRef(),
                 PassManager::Nesting::Implicit);
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Failed to apply pass manager CL options\n";
    return 1;
  }
  applyDefaultTimingPassManagerCLOptions(pm);

  iree_integrations::MHLO::buildMHLOImportPassPipeline(pm);

  // Note that we emit the ABI last since any needed function-level
  // transformations (i.e. de-tupling, etc) should have been done.
  pm.addNestedPass<func::FuncOp>(
      iree_integrations::MHLO::createEmitDefaultIREEABIPass());

  if (failed(pm.run(*module))) {
    llvm::errs() << "Running iree-xla-import MHLO import pass pipeline failed "
                    "(see diagnostics)\n";
    return 2;
  }

  // Save temp output.
  if (!saveTempIreeImport.empty()) {
    if (failed(saveToFile(saveTempIreeImport))) return 10;
  }

  // Save output.
  if (failed(saveToFile(outputFilename))) return 3;
  return 0;
}
