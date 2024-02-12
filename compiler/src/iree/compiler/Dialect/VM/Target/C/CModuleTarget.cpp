// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Target/C/CModuleTarget.h"

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/ConvertVMToEmitC.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/DropExcludedExports.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::VM {

/// Adapted from BytecodeModuleTarget and extended by C specific passes
static LogicalResult
canonicalizeModule(IREE::VM::ModuleOp moduleOp,
                   IREE::VM::CTargetOptions targetOptions) {
  RewritePatternSet patterns(moduleOp.getContext());
  ConversionTarget target(*moduleOp.getContext());
  target.addLegalDialect<IREE::VM::VMDialect>();
  target.addLegalOp<IREE::Util::OptimizationBarrierOp>();

  // Add all VM canonicalization patterns and mark pseudo-ops illegal.
  auto *context = moduleOp.getContext();
  for (auto op : context->getRegisteredOperations()) {
    // Non-serializable ops must be removed prior to serialization.
    if (op.hasTrait<OpTrait::IREE::VM::PseudoOp>()) {
      op.getCanonicalizationPatterns(patterns, context);
      target.setOpAction(op, ConversionTarget::LegalizationAction::Illegal);
    }

    // Debug ops must not be present when stripping.
    // TODO(benvanik): add RemoveDisabledDebugOp pattern.
    if (op.hasTrait<OpTrait::IREE::VM::DebugOnly>() &&
        targetOptions.stripDebugOps) {
      target.setOpAction(op, ConversionTarget::LegalizationAction::Illegal);
    }
  }

  if (failed(applyFullConversion(moduleOp, target, std::move(patterns)))) {
    return moduleOp.emitError() << "unable to fully apply conversion to module";
  }

  PassManager passManager(context);
  if (failed(mlir::applyPassManagerCLOptions(passManager))) {
    return moduleOp.emitError() << "Failed to apply pass manager CL options";
  }
  mlir::applyDefaultTimingPassManagerCLOptions(passManager);
  auto &modulePasses = passManager.nest<IREE::VM::ModuleOp>();

  // TODO(benvanik): these ideally happen beforehand but when performing
  // serialization the input IR often has some of these low-level VM ops. In
  // real workflows these have already run earlier and are no-ops.
  modulePasses.addPass(IREE::VM::createGlobalInitializationPass());
  modulePasses.addPass(IREE::VM::createDropEmptyModuleInitializersPass());

  if (targetOptions.optimize) {
    // TODO(benvanik): run this as part of a fixed-point iteration.
    modulePasses.addPass(mlir::createInlinerPass());
    modulePasses.addPass(mlir::createCSEPass());
    modulePasses.addPass(mlir::createCanonicalizerPass());
  }

  // C target specific pass

  // Erase exports annotated with 'emitc.exclude'. This makes testing
  // of partially supported ops easier. For the DCE pass to remove the
  // referenced function it must be unused and marked private.
  modulePasses.addPass(createDropExcludedExportsPass());
  modulePasses.addPass(mlir::createSymbolDCEPass());

  // In the the Bytecode module the order is:
  // * `createDropCompilerHintsPass()`
  // * `IREE::VM::createOrdinalAllocationPass()`
  // Here, we have to reverse the order and run
  // `createConvertVMToEmitCPass()` inbetween to test the EmitC pass.
  // Otherwise, the constants get folded by the canonicalizer.

  // Mark up the module with ordinals for each top-level op (func, etc).
  // This will make it easier to correlate the MLIR textual output to the
  // binary output.
  // We don't want any more modifications after this point as they could
  // invalidate the ordinals.
  modulePasses.addPass(IREE::VM::createOrdinalAllocationPass());

  // C target specific pass
  modulePasses.addPass(createConvertVMToEmitCPass());

  modulePasses.addPass(IREE::Util::createDropCompilerHintsPass());
  modulePasses.addPass(mlir::createCanonicalizerPass());

  if (failed(passManager.run(moduleOp->getParentOfType<mlir::ModuleOp>()))) {
    return moduleOp.emitError() << "failed during transform passes";
  }

  return success();
}

LogicalResult translateModuleToC(IREE::VM::ModuleOp moduleOp,
                                 CTargetOptions targetOptions,
                                 llvm::raw_ostream &output) {
  moduleOp.getContext()->getOrLoadDialect<IREE::Util::UtilDialect>();

  if (failed(canonicalizeModule(moduleOp, targetOptions))) {
    return moduleOp.emitError()
           << "failed to canonicalize vm.module to a serializable form";
  }
  auto innerModules = moduleOp.getOps<mlir::ModuleOp>();
  if (innerModules.empty()) {
    return moduleOp.emitError()
           << "vm module does not contain an inner builtin.module op";
  }
  mlir::ModuleOp mlirModule = *innerModules.begin();

  if (targetOptions.outputFormat == COutputFormat::kMlirText) {
    // Use the standard MLIR text printer.
    mlirModule.getOperation()->print(output);
    output << "\n";
    return success();
  }

  return mlir::emitc::translateToCpp(mlirModule.getOperation(), output, true);
}

LogicalResult translateModuleToC(mlir::ModuleOp outerModuleOp,
                                 CTargetOptions targetOptions,
                                 llvm::raw_ostream &output) {
  auto moduleOps = outerModuleOp.getOps<IREE::VM::ModuleOp>();
  if (moduleOps.empty()) {
    return outerModuleOp.emitError()
           << "outer module does not contain a vm.module op";
  }
  return translateModuleToC(*moduleOps.begin(), targetOptions, output);
}

} // namespace mlir::iree_compiler::IREE::VM
