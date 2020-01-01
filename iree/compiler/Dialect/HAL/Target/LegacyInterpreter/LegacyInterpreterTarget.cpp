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

#include "iree/compiler/Dialect/HAL/Target/LegacyInterpreter/LegacyInterpreterTarget.h"

#include <utility>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/minireflect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/LegacyUtil.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Translation/Interpreter/IR/OpWriters.h"
#include "iree/compiler/Translation/Interpreter/Serialization/VMFunctionBuilder.h"
#include "iree/compiler/Translation/Interpreter/Serialization/VMModuleBuilder.h"
#include "iree/compiler/Translation/Interpreter/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// TODO(benvanik): add flags.
// static llvm::cl::OptionCategory halLegacyInterpreterOptionsCategory(
//     "IREE legacy interpreter backend options");

LegacyInterpreterTargetOptions getLegacyInterpreterTargetOptionsFromFlags() {
  LegacyInterpreterTargetOptions targetOptions;
  // TODO(benvanik): flags.
  return targetOptions;
}

// Builds a pass pipeline that optimizes and legalizes the module to the form
// expected by translation.
static void buildLegalizeInputPassPipeline(OpPassManager *passManager) {
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
static void buildInterpreterConversionPassPipeline(OpPassManager *passManager) {
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
static void buildInterpreterLoweringPassPipeline(OpPassManager *passManager) {
  // Lower iree_hl_interp -> iree_ll_interp.
  passManager->addPass(createLowerInterpreterDialectPass());

  // Assign ordinals used by the bytecode to reference executables and
  // functions.
  passManager->addPass(createAssignFunctionOrdinalsPass());
}

static LogicalResult declareInterpreterFunction(
    FuncOp funcOp, LegacyInterpreterTargetOptions targetOptions,
    VMModuleBuilder *moduleBuilder) {
  auto *functionTable = moduleBuilder->function_table();
  if (functionTable->IsFunctionDeclared(funcOp)) {
    // Already declared.
    return success();
  }

  LinkageType linkageType;
  if (funcOp.isExternal()) {
    linkageType = LinkageType::kImport;
  } else if (funcOp.getAttr("iree.executable.export")) {
    linkageType = LinkageType::kExport;
  } else {
    linkageType = LinkageType::kInternal;
  }
  if (failed(functionTable->DeclareFunction(funcOp, linkageType))) {
    return funcOp.emitError() << "unable to declare function";
  }

  // Import functions must have their definition defined here so we get their
  // type. Internal and export functions will be defined during conversion.
  if (linkageType == LinkageType::kImport) {
    VMFunctionBuilder functionBuilder(funcOp, moduleBuilder->function_table(),
                                      moduleBuilder->fbb());
    auto functionOffset = functionBuilder.Finish();
    if (functionOffset.IsNull()) {
      return funcOp.emitError() << "failed to create import function bytecode";
    }
    if (failed(functionTable->DefineFunction(funcOp, functionOffset))) {
      return failure();
    }
  }

  return success();
}

static LogicalResult defineInterpreterFunction(
    FuncOp funcOp, LegacyInterpreterTargetOptions targetOptions,
    VMModuleBuilder *moduleBuilder) {
  VMFunctionBuilder functionBuilder(funcOp, moduleBuilder->function_table(),
                                    moduleBuilder->fbb());
  registerInterpreterCustomWriters(&functionBuilder);
  if (failed(functionBuilder.ConvertBytecode())) {
    return failure();
  }
  auto functionOffset = functionBuilder.Finish();
  if (functionOffset.IsNull()) {
    return funcOp.emitError() << "failed to serialize function";
  }
  return moduleBuilder->function_table()->DefineFunction(funcOp,
                                                         functionOffset);
}

LogicalResult translateToLegacyInterpreterExecutable(
    IREE::Flow::ExecutableOp sourceOp, IREE::HAL::ExecutableOp targetOp,
    ExecutableTargetOptions executableOptions,
    LegacyInterpreterTargetOptions targetOptions) {
  // Clone the module containing the things we want to translate. We do this so
  // that multiple targets can pull from the same source without conflicting.
  auto moduleOp = sourceOp.getInnerModule().clone();
  makeLegacyExecutableABI(sourceOp, moduleOp, targetOp);

  // Run all passes to go from input to the iree_ll_interp dialect.
  PassManager conversionPassManager(moduleOp.getContext());
  buildLegalizeInputPassPipeline(&conversionPassManager);
  buildInterpreterConversionPassPipeline(&conversionPassManager);
  buildInterpreterLoweringPassPipeline(&conversionPassManager);
  if (failed(conversionPassManager.run(moduleOp))) {
    return moduleOp.emitError() << "failed to run conversion passes";
  }

  ::flatbuffers::FlatBufferBuilder fbb;
  VMModuleBuilder moduleBuilder(&fbb);

  // Declare functions first so that we get stable indices during declaration
  // (as call ops need to use the function table).
  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    if (failed(declareInterpreterFunction(funcOp, targetOptions,
                                          &moduleBuilder))) {
      return failure();
    }
  }

  // Define functions now that all functions have been declared.
  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    if (failed(
            defineInterpreterFunction(funcOp, targetOptions, &moduleBuilder))) {
      return failure();
    }
  }

  // Serialize the module to bytecode.
  auto moduleDef = moduleBuilder.Finish();
  if (moduleDef.IsNull()) {
    return moduleOp.emitError() << "failed to verify completed module def";
  }
  auto bytes = moduleBuilder.Serialize(moduleDef);
  if (bytes.empty()) {
    return moduleOp.emitError() << "failed to serialize final module def";
  }

  // Add the binary data to the target executable.
  OpBuilder targetBuilder(&targetOp.getBlock());
  targetBuilder.setInsertionPoint(&targetOp.getBlock().back());
  auto binaryOp = targetBuilder.create<IREE::HAL::ExecutableBinaryOp>(
      targetOp.getLoc(),
      static_cast<uint32_t>(IREE::HAL::ExecutableFormat::IreeBytecode),
      std::move(bytes));
  binaryOp.getBlock().getOperations().insert(
      Block::iterator(binaryOp.getBlock().back()), moduleOp);
  return success();
}

static ExecutableTargetRegistration targetRegistration(
    "interpreter-bytecode",
    +[](IREE::Flow::ExecutableOp sourceOp, IREE::HAL::ExecutableOp targetOp,
        ExecutableTargetOptions executableOptions) {
      return translateToLegacyInterpreterExecutable(
          sourceOp, targetOp, std::move(executableOptions),
          getLegacyInterpreterTargetOptionsFromFlags());
    });

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
