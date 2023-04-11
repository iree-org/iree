// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Main entry function for the iree-import-tf tool (and derived binaries).
// Note that this is not an e2e tool: it is purely the first stage of the
// pipeline intended to lower TensorFlow GraphDefs and SavedModels to a form
// suitable for input to the IREE compiler.
//
// Since none of the TensorFlow imports come from an MLIR text form, it is a bit
// of an odd fit for a *-translate style tool, which is why this diverges.

#include "iree_tf_compiler/MHLO/Passes.h"
#include "iree_tf_compiler/TF/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/core/platform/errors.h"

using namespace llvm;
using namespace mlir;

namespace {

enum ImportType {
  savedmodel_v2,
  savedmodel_v1,
};

enum class OutputFormat {
  none,
  mlir_ir,
  mlir_bytecode,
};

}  // namespace

static OwningOpRef<mlir::ModuleOp> importSavedModelV2(
    MLIRContext &context, const std::string &inputPath,
    const std::string &savedModelExportedNames) {
  tensorflow::SavedModelV2Bundle bundle;
  auto loadStatus = tensorflow::SavedModelV2Bundle::Load(inputPath, &bundle);
  if (!loadStatus.ok()) {
    llvm::errs() << "TensorFlow reported error loading saved model:\n  "
                 << loadStatus.ToString() << "\n\n";
    if (!tensorflow::errors::IsNotFound(loadStatus)) {
      llvm::errs()
          << "Note: Attempted to load V2 SavedModel. Double check that "
             "this is correct "
          << "and adjust via the flag "
             "--tf-import-type=savedmodel_v1|savedmodel_v2\n";
    }
    return nullptr;
  }

  std::vector<std::string> exportedNamesVector =
      absl::StrSplit(savedModelExportedNames, ',', absl::SkipEmpty());
  auto loadedModule = tensorflow::ConvertSavedModelToMlir(
      &bundle, &context, absl::MakeSpan(exportedNamesVector));
  if (!loadedModule.ok()) {
    llvm::errs() << "Error performing initial import from SavedModel to MLIR. "
                 << "Reported error below (and see diagnostics):\n"
                 << "  " << loadedModule.status().ToString() << "\n";
    return nullptr;
  }

  return std::move(loadedModule).value();
}

static OwningOpRef<mlir::ModuleOp> importSavedModelV1(
    MLIRContext &context, const std::string &inputPath,
    const std::string &savedModelExportedNames,
    const std::string &savedModelTags) {
  tensorflow::SavedModelBundle bundle;
  tensorflow::SessionOptions session_options;
  // Force saved model states to be restored to CPU.
  (*session_options.config.mutable_device_count())["GPU"] = 0;

  std::unordered_set<std::string> tags =
      absl::StrSplit(savedModelTags, ',', absl::SkipEmpty());
  auto loadStatus =
      tensorflow::LoadSavedModel(session_options,
                                 /*run_options=*/{}, inputPath, tags, &bundle);
  if (!loadStatus.ok()) {
    llvm::errs() << "TensorFlow reported error loading saved model:\n  "
                 << loadStatus.ToString() << "\n\n";
    if (!tensorflow::errors::IsNotFound(loadStatus)) {
      llvm::errs()
          << "Note: Attempted to load V1 SavedModel. Double check that "
             "this is correct "
          << "and adjust via the flag "
             "--tf-import-type=savedmodel_v1|savedmodel_v2\n";
    }
    return nullptr;
  }

  std::vector<std::string> exportedNamesVector =
      absl::StrSplit(savedModelExportedNames, ',', absl::SkipEmpty());

  tensorflow::MLIRImportOptions import_options;
  auto loadedModule = ConvertSavedModelV1ToMlir(
      bundle, absl::MakeSpan(exportedNamesVector), &context, import_options);

  if (!loadedModule.ok()) {
    llvm::errs() << "Error performing initial import from SavedModel to MLIR. "
                 << "Reported error below (and see diagnostics):\n"
                 << "  " << loadedModule.status().ToString() << "\n";
    return nullptr;
  }

  return std::move(loadedModule).value();
}

int main(int argc, char **argv) {
  tensorflow::InitMlir y(&argc, &argv);
  llvm::setBugReportMsg(
      "Please report issues to https://github.com/openxla/iree/issues and "
      "include the crash backtrace.\n");

  static cl::opt<std::string> inputPath(
      cl::Positional, cl::desc("<saved model directory>"), cl::Required);
  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));
  static cl::opt<ImportType> importType(
      "tf-import-type", cl::desc("The type of TensorFlow model to import"),
      cl::values(clEnumVal(savedmodel_v2,
                           "Import a TensorFlow SavedModel V2 (directory)"),
                 clEnumVal(savedmodel_v1,
                           "Import a TensorFlow SavedModel V1 (directory)")));

  // The output format flag is the master control for what we do with the
  // in-memory compiled form.
  llvm::cl::opt<OutputFormat> outputFormat(
      "output-format", llvm::cl::desc("Format of imported output"),
      llvm::cl::values(clEnumValN(OutputFormat::mlir_bytecode, "mlir-bytecode",
                                  "MLIR Bytecode (default)"),
                       clEnumValN(OutputFormat::mlir_ir, "mlir-ir", "MLIR IR")),
      llvm::cl::init(OutputFormat::mlir_bytecode));

  static llvm::cl::opt<std::string> savedModelExportedNames(
      "tf-savedmodel-exported-names",
      llvm::cl::desc("Names to export from SavedModel, separated by ','. Empty "
                     "(the default) means export all."),
      llvm::cl::init(""));

  static llvm::cl::opt<std::string> savedModelTags(
      "tf-savedmodel-tags",
      llvm::cl::desc("Tags used to indicate which MetaGraphDef to import, "
                     "separated by ','"),
      llvm::cl::init("serve"));
  static llvm::cl::opt<std::string> saveTempTfInput(
      "save-temp-tf-input",
      llvm::cl::desc("Save the TF pipeline input to this file"),
      llvm::cl::init(""));
  static llvm::cl::opt<std::string> saveTempMidLevelImport(
      "save-temp-mid-level-input",
      llvm::cl::desc("Save the mid level IR to this file"), llvm::cl::init(""));
  static llvm::cl::opt<std::string> saveTempIreeImport(
      "save-temp-iree-input",
      llvm::cl::desc("Save the resultant IR to this file (useful for saving an "
                     "intermediate in a pipeline)"),
      llvm::cl::init(""));
  static llvm::cl::opt<bool> prettifyTfDebugInfo(
      "prettify-tf-debug-info",
      llvm::cl::desc("Prettifies TF debug information to make it easier "
                     "to look at"),
      llvm::cl::init(true));
  static llvm::cl::opt<bool> useTosa(
      "use-tosa", llvm::cl::desc("Use tosa as the intermediate IR"),
      llvm::cl::init(false));

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv);

  DialectRegistry registry;
  RegisterAllTensorFlowDialects(registry);

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  OwningOpRef<mlir::ModuleOp> module;

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

  // First stage import.
  switch (importType) {
    case savedmodel_v2:
      module = importSavedModelV2(context, inputPath, savedModelExportedNames);
      break;
    case savedmodel_v1:
      module = importSavedModelV1(context, inputPath, savedModelExportedNames,
                                  savedModelTags);
      break;
    default:
      assert(false && "unsupported import type enum");
  }
  if (!module) return 1;

  // Save temp output.
  if (!saveTempTfInput.empty()) {
    if (failed(saveToFile(saveTempTfInput))) return 10;
  }

  // Run passes.
  {
    PassManager pm(&context, module.get()->getName().getStringRef(),
                   PassManager::Nesting::Implicit);
    if (failed(applyPassManagerCLOptions(pm))) {
      llvm::errs() << "Failed to apply pass manager CL options\n";
      return 1;
    }

    if (prettifyTfDebugInfo) {
      pm.addPass(iree_integrations::TF::createPrettifyDebugInfoPass());
    }

    iree_integrations::TF::buildTFImportPassPipeline(pm, useTosa);
    if (failed(pm.run(*module))) {
      llvm::errs() << "Running iree-import-tf TF import pass pipeline failed "
                      "(see diagnostics)\n";
      return 2;
    }
    if (!saveTempMidLevelImport.empty()) {
      if (failed(saveToFile(saveTempMidLevelImport))) return 10;
    }
  }

  // Save temp output.
  if (!saveTempIreeImport.empty()) {
    if (failed(saveToFile(saveTempIreeImport))) return 10;
  }

  // Save output.
  if (failed(saveToFile(outputFilename))) return 3;
  return 0;
}
