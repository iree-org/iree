// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Modules/HAL/Loader/IR/HALLoaderDialect.h"
#include "iree/compiler/Modules/HAL/Loader/Transforms/PassDetail.h"
#include "iree/compiler/Modules/HAL/Loader/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace Loader {

static void replaceExecutableWithGlobal(IREE::HAL::ExecutableOp executableOp) {
  OpBuilder moduleBuilder(executableOp);

  auto loc = executableOp.getLoc();

  // Create global representing the loaded executable.
  // This matches the executable name and is used to directly access the
  // executable reference during dispatches.
  auto executableType = moduleBuilder.getType<IREE::HAL::ExecutableType>();
  auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
      loc, executableOp.getName(), /*isMutable=*/false, executableType);
  globalOp.setPrivate();

  // Create initializer that selects the right binary and loads it.
  auto initializerOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);
  auto entryBuilder = OpBuilder::atBlockBegin(initializerOp.addEntryBlock());

  // Reserve one block per attempt to load a binary.
  auto binaryOps =
      llvm::to_vector(executableOp.getOps<IREE::HAL::ExecutableBinaryOp>());
  SmallVector<Block *> queryBlocks;
  SmallVector<Block *> loadBlocks;
  for (size_t i = 0; i < binaryOps.size(); ++i) {
    queryBlocks.push_back(initializerOp.addBlock());
    loadBlocks.push_back(initializerOp.addBlock());
  }

  // Failure block when no binary is supported.
  auto *failBlock = initializerOp.addBlock();
  {
    auto failBuilder = OpBuilder::atBlockBegin(failBlock);
    Value status = failBuilder.create<arith::ConstantIntOp>(
        loc, static_cast<int>(IREE::Util::StatusCode::Unavailable), 32);
    failBuilder.create<IREE::Util::StatusCheckOkOp>(
        loc, status,
        "none of the executable binaries in the module are supported by the "
        "runtime");
    failBuilder.create<IREE::Util::ReturnOp>(loc);
  }

  // Exit block takes the loaded executable and stores it.
  auto *exitBlock = initializerOp.addBlock();
  {
    auto exitBuilder = OpBuilder::atBlockBegin(exitBlock);
    auto executableArg = exitBlock->addArgument(executableType, loc);
    exitBuilder.create<IREE::Util::GlobalStoreOp>(loc, executableArg,
                                                  globalOp.getName());
    exitBuilder.create<IREE::Util::ReturnOp>(loc);
  }

  // Start with the first try.
  if (!queryBlocks.empty()) {
    entryBuilder.create<cf::BranchOp>(loc, queryBlocks[0]);
  } else {
    entryBuilder.create<cf::BranchOp>(loc, failBlock);
  }

  // Build the full chain of try ops. An scf.switch would be nice...
  // We could also avoid this by having an op that given a list of formats
  // selected the ones that were supported - that'd result in smaller binary
  // sizes but not allow for customization of selection logic. Today this
  // looks bad because our selection logic is dumb :)
  //
  // ^queryBlock:
  //   %supported = executable.query_support "format" : i1
  //   cond_br %supported, ^loadBlock, ^nextBlock
  // ^loadBlock:
  //   %exe = executable.load : !hal.executable
  //   br ^exit(%exe)
  // ^nextBlock: ...
  for (unsigned i = 0; i < binaryOps.size(); ++i) {
    auto binaryOp = binaryOps[i];
    auto binaryLoc = binaryOp.getLoc();

    // Query whether the format is supported and branch to the load block if
    // it is. Otherwise we go to the next query block or fail if at the end.
    auto queryBuilder = OpBuilder::atBlockBegin(queryBlocks[i]);
    auto *nextBlock = i + 1 < binaryOps.size() ? queryBlocks[i + 1] : failBlock;
    Value isSupported =
        queryBuilder.create<IREE::HAL::Loader::ExecutableQuerySupportOp>(
            binaryLoc, queryBuilder.getI1Type(), binaryOp.getFormatAttr());
    queryBuilder.create<cf::CondBranchOp>(binaryLoc, isSupported, loadBlocks[i],
                                          ValueRange{}, nextBlock,
                                          ValueRange{});

    // Load the executable. This may still fail but it'll propagate the error
    // up to the user with the full status message instead of continuing
    // execution.
    auto loadBuilder = OpBuilder::atBlockBegin(loadBlocks[i]);
    auto alignmentAttr = loadBuilder.getIndexAttr(64);
    Value binaryData = loadBuilder.create<IREE::Util::BufferConstantOp>(
        binaryLoc, binaryOp.getNameAttr(), binaryOp.getData(), alignmentAttr,
        binaryOp.getMimeTypeAttr());
    SmallVector<Value> constants; // TBD
    Value executable = loadBuilder.create<IREE::HAL::Loader::ExecutableLoadOp>(
        binaryLoc, executableType, binaryOp.getFormatAttr(), binaryData,
        constants);
    loadBuilder.create<cf::BranchOp>(binaryLoc, exitBlock,
                                     ValueRange{executable});
  }

  // Op goes away to get replaced with a global.
  executableOp.erase();
}

// Runs conversion with registered input dialects.
class MaterializeExecutablesPass
    : public MaterializeExecutablesBase<MaterializeExecutablesPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect, IREE::HAL::HALDialect,
                    IREE::HAL::Loader::HALLoaderDialect, arith::ArithDialect,
                    cf::ControlFlowDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    // Walk executables and convert each one to a global.
    for (auto executableOp : llvm::make_early_inc_range(
             moduleOp.getOps<IREE::HAL::ExecutableOp>())) {
      replaceExecutableWithGlobal(executableOp);
    }

    // Find lookup ops referencing an executable and swap it to a global load.
    for (auto funcOp : llvm::make_early_inc_range(
             moduleOp.getOps<mlir::FunctionOpInterface>())) {
      funcOp.walk([&](IREE::HAL::Loader::ExecutableLookupOp lookupOp) {
        Value executable = OpBuilder(lookupOp).create<IREE::Util::GlobalLoadOp>(
            lookupOp.getLoc(), lookupOp.getResult().getType(),
            lookupOp.getExecutableAttr());
        lookupOp.replaceAllUsesWith(executable);
        lookupOp.erase();
      });
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createMaterializeExecutablesPass() {
  return std::make_unique<MaterializeExecutablesPass>();
}

} // namespace Loader
} // namespace mlir::iree_compiler::IREE::HAL
