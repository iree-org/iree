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

// Main entry function for the iree-tf-import tool (and derived binaries).
// Note that this is not an e2e tool: it is purely the first stage of the
// pipeline intended to lower TensorFlow GraphDefs and SavedModels to a form
// suitable for input to the IREE compiler.
//
// Since none of the TensorFlow imports come from an MLIR text form, it is a bit
// of an odd fit for a *-translate style tool, which is why this diverges.

#include "iree_tf_compiler/TF/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
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

}  // namespace

static OwningModuleRef importSavedModelV2(
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

  return loadedModule.ConsumeValueOrDie();
}

static OwningModuleRef importSavedModelV1(
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

  return loadedModule.ConsumeValueOrDie();
}

int main(int argc, char **argv) {
  tensorflow::InitMlir y(&argc, &argv);

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
  static llvm::cl::opt<std::string> saveTempIreeImport(
      "save-temp-iree-input",
      llvm::cl::desc("Save the resultant IR to this file (useful for saving an "
                     "intermediate in a pipeline)"),
      llvm::cl::init(""));

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv);

  DialectRegistry registry;
  RegisterAllTensorFlowDialects(registry);

  MLIRContext context;
  OwningModuleRef module;
  registry.loadAll(&context);

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
      llvm_unreachable("unsupported import type enum");
  }
  if (!module) return 1;

  // Save temp output.
  if (!saveTempTfInput.empty()) {
    if (failed(saveToFile(saveTempTfInput))) return 10;
  }

  // Run passes.
  PassManager pm(&context, PassManager::Nesting::Implicit);
  applyPassManagerCLOptions(pm);

  iree_integrations::TF::buildTFImportPassPipeline(pm);
  if (failed(pm.run(*module))) {
    llvm::errs()
        << "Running iree-tf-import pass pipeline failed (see diagnostics)\n";
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
