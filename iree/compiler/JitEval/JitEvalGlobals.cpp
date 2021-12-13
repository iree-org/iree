// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/TranslationFlags.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/JitEval/PassDetail.h"
#include "iree/compiler/JitEval/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_compiler {
namespace JitEval {

namespace {

struct ProgramExtractor {
 public:
  ProgramExtractor(Operation *sourceModuleOp, Operation *targetModuleOp)
      : sourceSymbolTable(sourceModuleOp),
        targetSymbolTable(targetModuleOp),
        builder(OpBuilder::atBlockEnd(&targetModuleOp->getRegion(0).front())) {}

  // Creates an accessor function to load the given global value.
  // Returns the created symbol name.
  StringAttr createAccessor(IREE::Util::GlobalOp globalOp) {
    Location loc = globalOp.getLoc();
    std::string name = (llvm::Twine("get$") + globalOp.getSymbolName()).str();
    Type globalType = globalOp.type();
    auto funcType =
        builder.getType<FunctionType>(TypeRange{}, TypeRange{globalType});
    auto funcOp = FuncOp::create(loc, name, funcType);
    StringAttr funcSymbolName = targetSymbolTable.insert(funcOp);
    Block *entryBlock = funcOp.addEntryBlock();

    OpBuilder funcBuilder = OpBuilder::atBlockEnd(entryBlock);
    Value loadValue =
        funcBuilder.create<IREE::Util::GlobalLoadOp>(loc, globalOp);
    funcBuilder.create<ReturnOp>(loc, ValueRange{loadValue});
    return funcSymbolName;
  }

  // Imports an op from the source module into the target. Cannot be used to
  // import symbols.
  void importOperation(Operation *sourceOp) {
    Operation *targetOp = sourceOp->clone();
    builder.insert(targetOp);
    scanDependentSymbols(targetOp);
  }

  // Imports any dependencies. Should be called after all user-required imports
  // are completed.
  LogicalResult importDependencies() {
    SmallVector<StringAttr> iterWorklist;

    while (!symbolImportWorklist.empty()) {
      iterWorklist.clear();
      iterWorklist.swap(symbolImportWorklist);

      for (StringAttr symbolRef : iterWorklist) {
        if (targetSymbolTable.lookup(symbolRef)) continue;

        Operation *sourceOp = sourceSymbolTable.lookup(symbolRef);
        if (!sourceOp) {
          return mlir::emitError(targetSymbolTable.getOp()->getLoc())
                 << "symbol not found while building jit-eval module: "
                 << symbolRef;
        }

        // TODO: Don't force insert to top. Globals out of order causes
        // segfault: https://github.com/google/iree/issues/7859
        auto ip = targetSymbolTable.getOp()->getRegion(0).front().begin();
        Operation *targetOp = sourceOp->clone();
        targetSymbolTable.insert(targetOp, ip);
        scanDependentSymbols(targetOp);
      }
    }

    return success();
  }

  void scanDependentSymbols(Operation *parentOp) {
    // Find any global accessors and note their dependent symbols.
    parentOp->walk([&](IREE::Util::GlobalAccessorOpInterface accessorOp) {
      FlatSymbolRefAttr refAttr = accessorOp.getGlobalRefAttr();
      symbolImportWorklist.push_back(refAttr.getAttr());
    });

    // TODO: Scan for functions, etc.
  }

 private:
  SymbolTable sourceSymbolTable;
  SymbolTable targetSymbolTable;
  OpBuilder builder;
  SmallVector<StringAttr> symbolImportWorklist;
};

// These options structs are not copy-constructable so we have to allocate them
// shared.
// TODO: See if we can make them copyable?
struct CompileOptions {
  IREE::Flow::TransformOptions flowOptions;
  IREE::HAL::TargetOptions executableOptions;
  IREE::Stream::TransformOptions streamOptions;
  IREE::VM::TargetOptions targetOptions;
};

struct JitEvalGlobalsPass : public JitEvalGlobalsBase<JitEvalGlobalsPass> {
  JitEvalGlobalsPass()
      : options(std::make_shared<CompileOptions>()),
        compilePipeline("builtin.module") {
    // Invoke IREE compilation flow.
    // TODO: Find a better place for this canonical list of passes?
    // TODO: Something better?
    options->executableOptions.targets.push_back("vmvx");
    options->targetOptions.i64Extension = true;
    options->targetOptions.f32Extension = true;
    options->targetOptions.f64Extension = true;

    IREE::Flow::buildFlowTransformPassPipeline(compilePipeline,
                                               options->flowOptions);
    IREE::Stream::buildStreamTransformPassPipeline(compilePipeline,
                                                   options->streamOptions);
    IREE::HAL::buildHALTransformPassPipeline(compilePipeline,
                                             options->executableOptions);
    IREE::VM::buildVMTransformPassPipeline(compilePipeline,
                                           options->targetOptions);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    compilePipeline.getDependentDialects(registry);
  }

  void runOnOperation() override {
    auto outerModule = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(outerModule.getBody());
    auto innerModule = builder.create<ModuleOp>(outerModule.getLoc());
    ProgramExtractor extractor(outerModule, innerModule);

    // Import initializers.
    for (Operation &childOp : *outerModule.getBody()) {
      if (llvm::isa<IREE::Util::InitializerOp>(childOp)) {
        extractor.importOperation(&childOp);
      }
    }

    // Transitively import any dependencies.
    if (failed(extractor.importDependencies())) {
      signalPassFailure();
    }

    // Find any globals that we pulled in which lack an initializer. These
    // are the ones we will try to eval. Stash {func_symbol, global_symbol}
    // pairs for later.
    SmallVector<std::pair<StringAttr, StringAttr>> uninitializedGlobals;
    for (Operation &childOp : *innerModule.getBody()) {
      auto globalOp = llvm::dyn_cast<IREE::Util::GlobalOp>(childOp);
      if (!globalOp) continue;
      if (!globalOp.getInitialValueAttr()) {
        StringAttr funcSymbol = extractor.createAccessor(globalOp);
        uninitializedGlobals.emplace_back(funcSymbol, globalOp.sym_nameAttr());
      }
    }

    if (failed(runPipeline(compilePipeline, innerModule))) {
      return signalPassFailure();
    }

    // Make sure to kill the temporary program we constructed.
    // innerModule.erase();
  }

  std::shared_ptr<CompileOptions> options;
  OpPassManager compilePipeline;
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createJitEvalGlobalsPass() {
  return std::make_unique<JitEvalGlobalsPass>();
}

}  // namespace JitEval
}  // namespace iree_compiler
}  // namespace mlir
