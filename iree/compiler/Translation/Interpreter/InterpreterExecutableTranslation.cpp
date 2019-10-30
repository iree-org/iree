// Copyright 2019 Google LLC
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

#include "iree/compiler/Translation/Interpreter/InterpreterExecutableTranslation.h"

#include <cstdint>
#include <iostream>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/minireflect.h"
#include "iree/compiler/IR/ConfigOps.h"
#include "iree/compiler/IR/Interpreter/OpWriters.h"
#include "iree/compiler/IR/Types.h"
#include "iree/compiler/Serialization/VMFunctionBuilder.h"
#include "iree/compiler/Serialization/VMFunctionTableBuilder.h"
#include "iree/compiler/Serialization/VMModuleBuilder.h"
#include "iree/compiler/Transforms/Interpreter/Passes.h"
#include "iree/compiler/Transforms/Passes.h"
#include "iree/compiler/Utils/Macros.h"
#include "iree/compiler/Utils/OpUtils.h"
#include "iree/compiler/Utils/TranslationUtils.h"
#include "iree/schemas/executable_def_generated.h"
#include "iree/schemas/module_def_generated.h"
#include "third_party/llvm/llvm/include/llvm/ADT/STLExtras.h"
#include "third_party/llvm/llvm/include/llvm/ADT/StringRef.h"
#include "third_party/llvm/llvm/include/llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Translation.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Builds a pass pipeline that optimizes and legalizes the module to the form
// expected by translation.
void buildLegalizeInputPassPipeline(PassManager *passManager) {
  OpPassManager &optPM = passManager->nest<FuncOp>();

  // Standard passes that shake out a lot of garbage.
  // Some may have been run prior to translation but this ensures we are always
  // in a known state.
  optPM.addPass(createCanonicalizerPass());
  optPM.addPass(createLoopFusionPass());
  optPM.addPass(createLoopInvariantCodeMotionPass());
  optPM.addPass(createMemRefDataFlowOptPass());
  optPM.addPass(createCanonicalizerPass());
  optPM.addPass(createSimplifyAffineStructuresPass());
  optPM.addPass(createCSEPass());
  optPM.addPass(createCanonicalizerPass());

  // Eliminate ops we don't care about based on a lack of side-effects.
  // IREE does not guarantee exception/error behavior of dead ops.
  optPM.addPass(createAggressiveOpEliminationPass());

  // Expand uses of tuples into independent args/results.
  passManager->addPass(createConvertFromTupleCallingConventionPass());
  passManager->addNestedPass<FuncOp>(createCanonicalizerPass());
}

// Builds a pass pipeline that converts functions to the iree_hl_interp dialect.
void buildInterpreterConversionPassPipeline(PassManager *passManager) {
  // We don't need the IREE binding ops anymore, as we match the calling
  // convention exactly (we're the same VM).
  passManager->addPass(createMakeExecutableABIPass());

  // Convert to the memref calling convention and optimize away as many
  // loads and stores as we can prior to progressing.
  passManager->addPass(createConvertToMemRefCallingConventionPass());
  passManager->addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager->addPass(createMemRefDataFlowOptPass());

  // Convert various dialects to IREE opcodes and cleanup leftover conversions.
  passManager->addPass(createLowerToInterpreterDialectPass());
  passManager->addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager->addPass(createAggressiveOpEliminationPass());

  // Widen reduction functions (that have iree.executable.reduction attrs) to
  // use their primitive IREE ops.
  passManager->addPass(createExpandReductionsToOpsPass());

  // Convert any uses of index to int32_t (as we explicitly don't want to
  // support dynamic index width).
  // This also looks for other weird types (i1, etc).
  passManager->addPass(createLegalizeTypeStoragePass());

  // Perform any last-minute optimizations to trim down the IR.
  passManager->addPass(createAggressiveOpEliminationPass());
  passManager->addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager->addPass(createLoopFusionPass());
  passManager->addPass(createLoopInvariantCodeMotionPass());
  passManager->addPass(createMemRefDataFlowOptPass());
  passManager->addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager->addNestedPass<FuncOp>(createCSEPass());
  passManager->addNestedPass<FuncOp>(createCanonicalizerPass());

  // Drop all functions that are not reachable.
  passManager->addPass(createDropUnreachableExecutableFunctionsPass());
}

// Builds a pass pipeline that lowers the iree_hl_interp dialect to the
// iree_ll_interp dialect and prepares for serialization.
void buildInterpreterLoweringPassPipeline(PassManager *passManager) {
  // Lower iree_hl_interp -> iree_ll_interp.
  passManager->addPass(createLowerInterpreterDialectPass());

  // Assign ordinals used by the bytecode to reference executables and
  // functions.
  passManager->addPass(createAssignFunctionOrdinalsPass());
}

class InterpreterTranslator {
 public:
  explicit InterpreterTranslator(ExecutableTranslationOptions options)
      : options_(options) {}

  const ExecutableTranslationOptions &options() const { return options_; }

  std::unique_ptr<iree::ExecutableDefT> translateExecutable(
      IREE::ExecutableOp executableOp);

 private:
  LogicalResult translateExecutableModule(IREE::ExecutableOp executableOp,
                                          ModuleOp moduleOp,
                                          VMModuleBuilder *moduleBuilder);
  LogicalResult declareFunction(FuncOp function,
                                VMModuleBuilder *moduleBuilder);
  LogicalResult defineFunction(FuncOp function, VMModuleBuilder *moduleBuilder);

  ExecutableTranslationOptions options_;
};

std::unique_ptr<iree::ExecutableDefT>
InterpreterTranslator::translateExecutable(IREE::ExecutableOp executableOp) {
  auto moduleOp = executableOp.getInnerModule();

  // Run all passes to go from input to the iree_ll_interp dialect.
  auto executableConversionPasses =
      createPassManager(moduleOp.getContext(), options());
  buildLegalizeInputPassPipeline(executableConversionPasses.get());
  buildInterpreterConversionPassPipeline(executableConversionPasses.get());
  buildInterpreterLoweringPassPipeline(executableConversionPasses.get());
  if (failed(runPassPipeline(options(), executableConversionPasses.get(),
                             moduleOp))) {
    executableOp.emitError() << "Failed to run conversion passes";
    return {};
  }

  // Build the module bytecode.
  ::flatbuffers::FlatBufferBuilder fbb;
  VMModuleBuilder moduleBuilder(&fbb);
  if (failed(
          translateExecutableModule(executableOp, moduleOp, &moduleBuilder))) {
    executableOp.emitError() << "Failed to translate executable module";
    return {};
  }
  auto moduleDef = moduleBuilder.Finish();
  if (moduleDef.IsNull()) {
    moduleOp.emitError() << "Failed to verify completed module def";
    return {};
  }
  auto bytes = moduleBuilder.Serialize(moduleDef);
  if (bytes.empty()) {
    moduleOp.emitError() << "Failed to serialize final module def";
    return {};
  }

  OpBuilder builder(executableOp);
  executableOp.setAttr("format", builder.getI32IntegerAttr(static_cast<int32_t>(
                                     IREE::ExecutableFormat::IreeBytecode)));

  auto executableDef = std::make_unique<iree::ExecutableDefT>();
  executableDef->format =
      static_cast<uint32_t>(IREE::ExecutableFormat::IreeBytecode);
  executableDef->supported_features = iree::ExecutableFeature::kDebugging;
  executableDef->contents = std::move(bytes);
  return executableDef;
}

LogicalResult InterpreterTranslator::translateExecutableModule(
    IREE::ExecutableOp executableOp, ModuleOp moduleOp,
    VMModuleBuilder *moduleBuilder) {
  // Declare functions first so that we get stable indices during declaration
  // (as call ops need to use the function table).
  for (auto function : moduleOp.getOps<FuncOp>()) {
    RETURN_IF_FAILURE(declareFunction(function, moduleBuilder));
  }

  // Define functions now that all functions have been declared.
  for (auto function : moduleOp.getOps<FuncOp>()) {
    RETURN_IF_FAILURE(defineFunction(function, moduleBuilder));
  }

  return success();
}

LogicalResult InterpreterTranslator::declareFunction(
    FuncOp function, VMModuleBuilder *moduleBuilder) {
  auto *functionTable = moduleBuilder->function_table();
  if (functionTable->IsFunctionDeclared(function)) {
    // Already declared.
    return success();
  }

  LinkageType linkageType;
  if (function.isExternal()) {
    linkageType = LinkageType::kImport;
  } else if (function.getAttr("iree.executable.export")) {
    linkageType = LinkageType::kExport;
  } else {
    linkageType = LinkageType::kInternal;
  }
  if (failed(functionTable->DeclareFunction(function, linkageType))) {
    return function.emitError() << "Unable to declare function";
  }

  // Import functions must have their definition defined here so we get their
  // type. Internal and export functions will be defined during conversion.
  if (linkageType == LinkageType::kImport) {
    VMFunctionBuilder functionBuilder(function, moduleBuilder->function_table(),
                                      moduleBuilder->fbb());
    auto functionOffset = functionBuilder.Finish();
    if (functionOffset.IsNull()) {
      return function.emitError()
             << "Failed to create import function bytecode";
    }
    RETURN_IF_FAILURE(
        functionTable->DefineFunction(function, functionOffset, {}));
  }

  return success();
}

LogicalResult InterpreterTranslator::defineFunction(
    FuncOp function, VMModuleBuilder *moduleBuilder) {
  VMFunctionBuilder functionBuilder(function, moduleBuilder->function_table(),
                                    moduleBuilder->fbb());
  registerInterpreterCustomWriters(&functionBuilder);
  RETURN_IF_FAILURE(functionBuilder.ConvertBytecode());
  auto functionOffset = functionBuilder.Finish();
  if (functionOffset.IsNull()) {
    return function.emitError() << "Failed to serialize function";
  }
  RETURN_IF_FAILURE(moduleBuilder->function_table()->DefineFunction(
      function, functionOffset, functionBuilder.source_map()));
  return success();
}

}  // namespace

llvm::Optional<ExecutableTranslationResult>
translateExecutableToInterpreterExecutable(
    ArrayRef<IREE::ExecutableOp> executableOps,
    ExecutableTranslationOptions options) {
  InterpreterTranslator translator(options);
  ExecutableTranslationResult translationResult;
  for (auto executableOp : llvm::make_early_inc_range(executableOps)) {
    auto executableDef = translator.translateExecutable(executableOp);
    if (!executableDef) {
      executableOp.emitError() << "Failed to translate one or more executables";
      return llvm::None;
    }
    translationResult.executable_defs.push_back(std::move(executableDef));
  }
  return translationResult;
}

static ExecutableTranslationRegistration
    InterpreterExecutableTranslationRegistration(
        "interpreter-bytecode", translateExecutableToInterpreterExecutable);

}  // namespace iree_compiler
}  // namespace mlir
