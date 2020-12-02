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

#include "integrations/tensorflow/compiler/Passes.h"
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

static llvm::cl::opt<std::string> savedModelTags(
    "tf-savedmodel-tags",
    llvm::cl::desc("Tags used to indicate which MetaGraphDef to import, "
                   "separated by ','"),
    llvm::cl::init("serve"));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> savedModelExportedNames(
    "tf-savedmodel-exported-names",
    llvm::cl::desc("Names to export from SavedModel, separated by ','. Empty "
                   "(the default) means export all."),
    llvm::cl::init(""));

OwningModuleRef importSavedModelV2(MLIRContext &context,
                                   const std::string &inputPath) {
  tensorflow::SavedModelV2Bundle bundle;
  auto loadStatus = tensorflow::SavedModelV2Bundle::Load(inputPath, &bundle);
  if (!loadStatus.ok()) {
    std::cerr << "TensorFlow reported error loading saved model:\n  "
              << loadStatus << "\n\n";
    if (!tensorflow::errors::IsNotFound(loadStatus)) {
      std::cerr << "Note: Attempted to load V2 SavedModel. Double check that "
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
    std::cerr << "Error performing initial import from SavedModel to MLIR. "
              << "Reported error below (and see diagnostics):\n"
              << "  " << loadedModule.status() << "\n";
    return nullptr;
  }

  return loadedModule.ConsumeValueOrDie();
}

OwningModuleRef importSavedModelV1(MLIRContext &context,
                                   const std::string &inputPath) {
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
    std::cerr << "TensorFlow reported error loading saved model:\n  "
              << loadStatus << "\n\n";
    if (!tensorflow::errors::IsNotFound(loadStatus)) {
      std::cerr << "Note: Attempted to load V1 SavedModel. Double check that "
                   "this is correct "
                << "and adjust via the flag "
                   "--tf-import-type=savedmodel_v1|savedmodel_v2\n";
    }
    return nullptr;
  }

  std::vector<std::string> exportedNamesVector =
      absl::StrSplit(savedModelExportedNames, ',', absl::SkipEmpty());

  auto loadedModule = ConvertSavedModelV1ToMlir(
      bundle, absl::MakeSpan(exportedNamesVector), &context,
      /*upgrade_legacy=*/false);

  if (!loadedModule.ok()) {
    std::cerr << "Error performing initial import from SavedModel to MLIR. "
              << "Reported error below (and see diagnostics):\n"
              << "  " << loadedModule.status() << "\n";
    return nullptr;
  }

  return loadedModule.ConsumeValueOrDie();
}

}  // namespace

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

  // First stage import.
  switch (importType) {
    case savedmodel_v2:
      module = importSavedModelV2(context, inputPath);
      break;
    case savedmodel_v1:
      module = importSavedModelV1(context, inputPath);
      break;
    default:
      llvm_unreachable("unsupported import type enum");
  }
  if (!module) return 1;

  // Run passes.
  PassManager pm(&context, PassManager::Nesting::Implicit);
  iree_compiler::createIreeTfImportPipeline(pm);
  if (failed(pm.run(*module))) {
    std::cerr
        << "Running iree-tf-import pass pipeline failed (see diagnostics)\n";
    return 2;
  }

  // Output.
  auto outputFile = openOutputFile(outputFilename);
  if (!outputFile) {
    return 3;
  }
  OpPrintingFlags printFlags;
  printFlags.enableDebugInfo();
  module->print(outputFile->os(), printFlags);
  (outputFile->os()) << "\n";
  outputFile->keep();

  return 0;
}
