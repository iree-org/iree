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

#include "iree/compiler/Translation/Sequencer/SequencerModuleTranslation.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/minireflect.h"
#include "iree/base/status.h"
#include "iree/compiler/IR/ConfigOps.h"
#include "iree/compiler/IR/Sequencer/OpWriters.h"
#include "iree/compiler/IR/StructureOps.h"
#include "iree/compiler/IR/Types.h"
#include "iree/compiler/Serialization/VMFunctionBuilder.h"
#include "iree/compiler/Serialization/VMFunctionTableBuilder.h"
#include "iree/compiler/Serialization/VMModuleBuilder.h"
#include "iree/compiler/Transforms/Passes.h"
#include "iree/compiler/Transforms/Sequencer/Passes.h"
#include "iree/compiler/Utils/Macros.h"
#include "iree/compiler/Utils/OpUtils.h"
#include "iree/compiler/Utils/TranslationUtils.h"
#include "iree/hal/executable_format.h"
#include "iree/schemas/executable_def_generated.h"
#include "iree/schemas/executable_table_def_generated.h"
#include "iree/schemas/module_def_generated.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Translation.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Builds a pass pipeline that optimizes and legalizes the module to the form
// expected by partitioning.
void buildLegalizeInputPassPipeline(PassManager *passManager) {
  // Convert to the subset of XLA HLO and Standard dialects supported as IREE
  // input. In particular, move from XLA HLO to standard control flow.
  passManager->addPass(xla_hlo::createLegalizeControlFlowPass());
  passManager->addPass(createLegalizeInputOpsPass());

  // Standard passes that shake out a lot of garbage.
  // Some may have been run prior to translation but this ensures we are always
  // in a known state.
  passManager->addPass(createCanonicalizerPass());
  passManager->addPass(createLoopFusionPass());
  passManager->addPass(createLoopInvariantCodeMotionPass());
  passManager->addPass(createMemRefDataFlowOptPass());
  passManager->addPass(createCanonicalizerPass());
  passManager->addPass(createSimplifyAffineStructuresPass());
  passManager->addPass(createCSEPass());
  passManager->addPass(createCanonicalizerPass());

  // Expand uses of tuples into independent args/results.
  passManager->addPass(createConvertFromTupleCallingConventionPass());
  passManager->addPass(createCanonicalizerPass());
}

// Builds a pass pipeline that partitions the module into sequencer functions
// and executables ready to be translated.
void buildPartitioningPassPipeline(PassManager *passManager) {
  // Find reduction ops and create iree.reduction_regions. We do this prior to
  // performing dispatch region identification so that we can build as big of
  // fused reduction regions as possible. The remaining ops will be put into
  // dispatch regions.
  passManager->addPass(createIdentifyReductionRegionsPass());
  passManager->addPass(createCSEPass());

  // Create all of the dispatch regions, CSE their workloads, and fold.
  passManager->addPass(createIdentifyDispatchRegionsPass());
  passManager->addPass(createCSEPass());
  passManager->addPass(createFoldCompatibleDispatchRegionsPass());

  // Note that as we are rematerializing things here it's critical we do not run
  // the canonicalizer/CSE between now and when we outline - otherwise it'll
  // undo all of our work!
  passManager->addPass(createRematerializeDispatchConstantsPass());

  // Outline the dispatch regions into their own functions. This separates the
  // sequencer functions performing dispatches from the dispatchees.
  passManager->addPass(createOutlineDispatchRegionsPass());
  passManager->addPass(createOutlineReductionRegionsPass());

  // Cleanup identity sequencer tensor-to-memref ops that clutter up the IR.
  passManager->addPass(createCanonicalizerPass());

  // Drop all functions that are no longer reachable.
  // This is important as many of the functions remaining are probably
  // dispatchable and unused now that we've outlined them executables.
  passManager->addPass(createDropUnreachableModuleFunctionsPass());

  // Drop all unused executables.
  // Note that we need to have dropped unreachable functions first otherwise
  // references could keep executables that are unreachable from exported
  // functions alive.
  passManager->addPass(createDropUnusedExecutablesPass());
}

// Builds a pass pipeline that converts sequencer functions to the iree_seq.hl
// dialect.
void buildSequencerConversionPassPipeline(PassManager *passManager) {
  passManager->addPass(createConvertToMemRefCallingConventionPass());

  // Convert ops that are supported by the sequencer directly to the sequencer
  // dialect. The ops that remain should be only those that can be moved into
  // dispatch regions.
  passManager->addPass(createLowerToSequencerDialectPass());

  // Cleanup identity sequencer tensor-to-memref ops and other memory accesses
  // that clutter up the IR.
  passManager->addPass(createCanonicalizerPass());
  passManager->addPass(createMemRefDataFlowOptPass());

  // Convert unsupported types (e.g. index -> i32, i1 -> i8).
  passManager->addPass(createLegalizeTypeStoragePass());

  // Eliminate ops we don't care about based on a lack of side-effects.
  // IREE does not guarantee exception/error behavior of dead ops.
  passManager->addPass(createAggressiveOpEliminationPass());

  // Perform any last-minute optimizations to trim down the IR.
  passManager->addPass(createCanonicalizerPass());
  passManager->addPass(createMemRefDataFlowOptPass());
  passManager->addPass(createCSEPass());
}

// Builds a pass pipeline that lowers the iree_seq.hl dialect to the iree_seq.ll
// dialect and prepares for serialization.
void buildSequencerLoweringPassPipeline(PassManager *passManager) {
  // Lower iree_hl_seq -> iree_ll_seq.
  passManager->addPass(createLowerSequencerDialectPass());
  passManager->addPass(createCanonicalizerPass());
  passManager->addPass(createMemRefDataFlowOptPass());
  passManager->addPass(createAggressiveOpEliminationPass());

  // Assign ordinals used by the bytecode to reference executables and
  // functions.
  passManager->addPass(createAssignFunctionOrdinalsPass());
  passManager->addPass(createAssignExecutableOrdinalsPass());

  // Plumb workload information down into executable entry points. This allows
  // the backends to calculate their workgroup sizes, indexing, etc.
  passManager->addPass(createAssignExecutableWorkloadAttrsPass());
}

// Inserts one or more iree.executable_target_config ops based on the
// translation options.
void insertTargetConfigOps(const ModuleTranslationOptions &options,
                           OpBuilder &builder) {
  llvm::StringSet<> targetBackends;
  if (options.target_backends.empty()) {
    // Add all backends when none are explicitly provided.
    targetBackends.insert(getExecutableTranslationRegistry().keys().begin(),
                          getExecutableTranslationRegistry().keys().end());
  } else {
    for (auto &targetBackend : options.target_backends) {
      for (auto &matchedBackend :
           matchExecutableTranslationBackendNames(targetBackend)) {
        targetBackends.insert(matchedBackend);
      }
    }
  }
  for (auto &targetBackend : targetBackends) {
    builder.create<IREE::ExecutableTargetConfigOp>(builder.getUnknownLoc(),
                                                   targetBackend.getKey());
  }
}

class SequencerTranslator {
 public:
  explicit SequencerTranslator(ModuleTranslationOptions options)
      : options_(options) {}

  const ModuleTranslationOptions &options() const { return options_; }

  std::vector<uint8_t> translateModule(ModuleOp module);

 private:
  LogicalResult translateMultiArchExecutable(
      IREE::MultiArchExecutableOp executableOp, VMModuleBuilder *moduleBuilder);

  LogicalResult translateSequencerModule(ModuleOp module,
                                         VMModuleBuilder *moduleBuilder);
  LogicalResult declareFunction(FuncOp function,
                                VMModuleBuilder *moduleBuilder);
  LogicalResult defineFunction(FuncOp function, VMModuleBuilder *moduleBuilder);

  ModuleTranslationOptions options_;
};

std::vector<uint8_t> SequencerTranslator::translateModule(ModuleOp module) {
  // Run one large set of passes to get to a partitioned module.
  auto partitioningPasses = createPassManager(module.getContext(), options());
  buildLegalizeInputPassPipeline(partitioningPasses.get());
  buildPartitioningPassPipeline(partitioningPasses.get());
  if (failed(runPassPipeline(options(), partitioningPasses.get(), module))) {
    module.emitError() << "Failed to run partitioning passes";
    return {};
  }

  // Run the sequencer-specific conversion passes on the module.
  auto sequencerConversionPasses =
      createPassManager(module.getContext(), options());
  buildSequencerConversionPassPipeline(sequencerConversionPasses.get());
  if (failed(runPassPipeline(options(), sequencerConversionPasses.get(),
                             module))) {
    module.emitError() << "Failed to run sequencer conversion passes";
    return {};
  }

  // Lower sequencer functions to their final form.
  auto sequencerLoweringPasses =
      createPassManager(module.getContext(), options());
  buildSequencerLoweringPassPipeline(sequencerLoweringPasses.get());
  if (failed(
          runPassPipeline(options(), sequencerLoweringPasses.get(), module))) {
    module.emitError() << "Failed to run sequencer lowering passes";
    return {};
  }

  // Perform translation on all executables.
  // We then know exactly what executable formats we have and can query them to
  // see if we need to do any additional processing (such as to support better
  // types/etc).
  ::flatbuffers::FlatBufferBuilder fbb;
  VMModuleBuilder moduleBuilder(&fbb);
  for (auto multiArchExecutableOp :
       module.getOps<IREE::MultiArchExecutableOp>()) {
    if (failed(translateMultiArchExecutable(multiArchExecutableOp,
                                            &moduleBuilder))) {
      module.emitError() << "Failed to translate multi-arch-executable";
      return {};
    }
  }

  // Build the module bytecode.
  if (failed(translateSequencerModule(module, &moduleBuilder))) {
    module.emitError() << "Unable to translate sequencer module";
    return {};
  }
  auto moduleDef = moduleBuilder.Finish();
  if (moduleDef.IsNull()) {
    module.emitError() << "Failed to verify completed module def";
    return {};
  }
  auto bytes = moduleBuilder.Serialize(moduleDef);
  if (bytes.empty()) {
    module.emitError() << "Failed to serialize final module def";
    return {};
  }
  return bytes;
}

LogicalResult SequencerTranslator::translateMultiArchExecutable(
    IREE::MultiArchExecutableOp multiArchExecutableOp,
    VMModuleBuilder *moduleBuilder) {
  auto &fbb = *moduleBuilder->fbb();

  // Find the unspecified executable. This is the template from which we will
  // translate to other targets.
  IREE::ExecutableOp templateExecutableOp;
  for (auto executableOp :
       multiArchExecutableOp.getBlock().getOps<IREE::ExecutableOp>()) {
    if (executableOp.format() ==
        static_cast<uint32_t>(IREE::ExecutableFormat::Unspecified)) {
      templateExecutableOp = executableOp;
      break;
    }
  }
  if (!templateExecutableOp) {
    // Fine for there to be no unspecified executable - just ignore.
    return success();
  }
  int entryPointCount = 0;
  for (auto func : templateExecutableOp.getInnerModule().getOps<FuncOp>()) {
    if (func.getAttr("iree.executable.export")) {
      ++entryPointCount;
    }
  }

  // For now we just add target config ops based on options. In the future we
  // could do this earlier via an analysis pass determining which targets should
  // be used for each executable.
  OpBuilder configBuilder(templateExecutableOp);
  configBuilder.setInsertionPointToStart(&templateExecutableOp.getBlock());
  insertTargetConfigOps(options(), configBuilder);

  // Find all target configs and bucket them into the backends that will
  // translate them. This way we can batch the translations and possibly enable
  // backends to dedupe some things.
  DenseMap<StringRef, std::vector<IREE::ExecutableTargetConfigOp>>
      backendTargetConfigOps;
  for (auto targetConfigOp : templateExecutableOp.getBlock()
                                 .getOps<IREE::ExecutableTargetConfigOp>()) {
    auto &targetConfigOps = backendTargetConfigOps[targetConfigOp.backend()];
    targetConfigOps.push_back(targetConfigOp);
  }
  if (backendTargetConfigOps.empty()) {
    // There are no target configs - which likely means we've already translated
    // this in a previous pass.
    return success();
  }

  ExecutableTranslationOptions translationOptions;
  translationOptions.CopyFrom(options());

  // Invoke each backend translator on the template executables to produce new
  // executables. The backends may produce any number of executables that we
  // then merge back in to the iree.multi_arch_executable and the module
  // flatbuffer.
  std::vector<std::unique_ptr<iree::ExecutableDefT>> translatedExecutableDefs;
  for (auto it : backendTargetConfigOps) {
    const auto &backendKey = it.first;
    const auto &targetConfigOps = it.second;

    // Find the translator to use in the registry. It must have been linked in
    // and the name must match what is used in the registration macro.
    auto translateExecutableFn =
        getExecutableTranslationRegistry().lookup(backendKey);
    if (!translateExecutableFn) {
      return multiArchExecutableOp.emitError()
             << "No registered backend found for target '" << backendKey.str()
             << "'; ensure it is linked in to your binary (have: "
             << llvm::join(getExecutableTranslationRegistry().keys(), ", ")
             << ")";
    }

    // Clone the executable for each config so that the translator is allowed to
    // modify it in-place.
    // We also need to strip all of the other configs so that the translator
    // backend only sees the one for each of its configs.
    OpBuilder builder(&multiArchExecutableOp.getBlock());
    builder.setInsertionPoint(multiArchExecutableOp.getBlock().getTerminator());
    SmallVector<IREE::ExecutableOp, 4> clonedExecutableOps;
    for (auto targetConfigOp : targetConfigOps) {
      auto executableCloneOp = cast<IREE::ExecutableOp>(
          builder.clone(*templateExecutableOp.getOperation()));
      for (auto existingTargetConfigOp : llvm::make_early_inc_range(
               executableCloneOp.getBlock()
                   .getOps<IREE::ExecutableTargetConfigOp>())) {
        existingTargetConfigOp.erase();
      }
      OpBuilder configBuilder(executableCloneOp);
      configBuilder.setInsertionPointToStart(&executableCloneOp.getBlock());
      configBuilder.clone(*targetConfigOp.getOperation());
      clonedExecutableOps.push_back(executableCloneOp);
    }

    // Perform translation on all of the backend-specific targets.
    // Note that the results here may not have the same number of executables we
    // started with if the backend either couldn't satisfy some of the requests
    // or decided to dedupe or expand certain ones.
    auto translationResults =
        translateExecutableFn(clonedExecutableOps, translationOptions);
    if (!translationResults.hasValue()) {
      return multiArchExecutableOp.emitError()
             << "Failed to translate executable with backend " << backendKey;
    }
    for (auto &executableDef : translationResults.getValue().executable_defs) {
      translatedExecutableDefs.push_back(std::move(executableDef));
    }
  }

  // Remove configs from the template executable so that if we are called again
  // we don't re-translate.
  for (auto targetConfigOp : llvm::make_early_inc_range(
           templateExecutableOp.getBlock()
               .getOps<IREE::ExecutableTargetConfigOp>())) {
    targetConfigOp.erase();
  }

  // Create multi-arch executable with all of the target-specific executables.
  iree::MultiArchExecutableDefT maedf;
  maedf.name = multiArchExecutableOp.getName();
  maedf.entry_point_count = entryPointCount;
  maedf.executables = std::move(translatedExecutableDefs);
  auto maedfOffset = iree::MultiArchExecutableDef::Pack(fbb, &maedf);
  RETURN_IF_FAILURE(
      moduleBuilder->executable_table()->AddMultiArchExecutable(maedfOffset));

  return success();
}

LogicalResult SequencerTranslator::translateSequencerModule(
    ModuleOp module, VMModuleBuilder *moduleBuilder) {
  // Declare functions. This must happen first so that we get stable indices
  // during declaration (as call ops need to use the function table).
  for (auto function : module.getOps<FuncOp>()) {
    RETURN_IF_FAILURE(declareFunction(function, moduleBuilder));
  }

  // Define functions and convert their bodies to bytecode.
  for (auto function : module.getOps<FuncOp>()) {
    RETURN_IF_FAILURE(defineFunction(function, moduleBuilder));
  }

  return success();
}

LogicalResult SequencerTranslator::declareFunction(
    FuncOp function, VMModuleBuilder *moduleBuilder) {
  auto *functionTable = moduleBuilder->function_table();
  if (functionTable->IsFunctionDeclared(function)) {
    // Already declared.
    return success();
  }

  LinkageType linkageType;
  if (function.isExternal()) {
    linkageType = LinkageType::kImport;
  } else if (function.getAttr("iree.module.export")) {
    linkageType = LinkageType::kExport;
  } else {
    linkageType = LinkageType::kInternal;
  }
  if (failed(functionTable->DeclareFunction(function, linkageType))) {
    return function.emitError()
           << "Unable to declare function " << function.getName();
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

LogicalResult SequencerTranslator::defineFunction(
    FuncOp function, VMModuleBuilder *moduleBuilder) {
  VMFunctionBuilder functionBuilder(function, moduleBuilder->function_table(),
                                    moduleBuilder->fbb());
  registerSequencerCustomWriters(&functionBuilder);
  RETURN_IF_FAILURE(functionBuilder.ConvertBytecode());
  auto functionOffset = functionBuilder.Finish();
  if (functionOffset.IsNull()) {
    return function.emitError() << "Failed to convert function to bytecode";
  }
  RETURN_IF_FAILURE(moduleBuilder->function_table()->DefineFunction(
      function, functionOffset, functionBuilder.source_map()));
  return success();
}

}  // namespace

std::vector<uint8_t> translateMlirToIreeSequencerModule(
    ModuleOp module, ModuleTranslationOptions options) {
  SequencerTranslator translator(options);
  return translator.translateModule(module);
}

LogicalResult translateMlirToIreeSequencerModuleFile(
    ModuleOp module, llvm::raw_ostream &output) {
  ModuleTranslationOptions options;
  SequencerTranslator translator(options);
  auto bytecodeModule = translator.translateModule(module);
  if (bytecodeModule.empty()) {
    return emitError(UnknownLoc::get(module.getContext()),
                     "failed to translate module");
  }

  output.write(reinterpret_cast<const char *>(bytecodeModule.data()),
               bytecodeModule.size());
  return success();
}

static TranslateFromMLIRRegistration MlirToIreeSequencerModuleTranslate(
    "mlir-to-iree-module", translateMlirToIreeSequencerModuleFile);

}  // namespace iree_compiler
}  // namespace mlir
