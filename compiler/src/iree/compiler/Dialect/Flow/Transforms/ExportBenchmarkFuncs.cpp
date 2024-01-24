// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Flow {

// Creates a util.global with a primitive value of |type| initialized to zeros.
// Supports: ints, floats, vectors, and tensors.
//
// Example:
//  util.global @some_fn_arg0 = 4 : i32
//  util.global @some_fn_arg0 = dense<4> : tensor<4xi32>
static IREE::Util::GlobalOp
createPrimitiveDefaultGlobalOp(std::string name, Location loc, Type type,
                               SymbolTable &symbolTable,
                               OpBuilder &moduleBuilder) {
  // Get a zero-initialized constant attribute for the type, if supported.
  auto initialValue = moduleBuilder.getZeroAttr(type);
  if (!initialValue) {
    mlir::emitError(loc) << "unsupported function argument type: " << type;
    return {};
  }

  // Global with initializer; tensors will get turned into buffers eventually.
  auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
      loc, name,
      /*isMutable=*/false, type, initialValue);
  globalOp.setPrivate();
  globalOp->setAttr("noinline", moduleBuilder.getUnitAttr());
  symbolTable.insert(globalOp);
  return globalOp;
}

// Creates a util.global of the given |globalType| and initializes a buffer or
// buffer view as a zeroed |tensorType|.
static IREE::Util::GlobalOp
createBufferLikeGlobalOp(std::string name, Location loc, Type globalType,
                         TensorType tensorType, SymbolTable &symbolTable,
                         OpBuilder &moduleBuilder) {
  // Create !hal.buffer global for the storage buffer or buffer view.
  auto globalOp = moduleBuilder.create<IREE::Util::GlobalOp>(
      loc, name,
      /*isMutable=*/false, globalType);
  globalOp.setPrivate();
  globalOp->setAttr("noinline", moduleBuilder.getUnitAttr());
  symbolTable.insert(globalOp);

  // Create an initializer that allocates the buffer storage.
  // We do this by splatting and exporting to a buffer so that it looks like it
  // was created by the user.
  auto initializerOp = moduleBuilder.create<IREE::Util::InitializerOp>(loc);
  auto initializerBuilder =
      OpBuilder::atBlockBegin(initializerOp.addEntryBlock());
  auto zeroAttr = moduleBuilder.getZeroAttr(tensorType.getElementType());
  auto zeroOp = initializerBuilder.create<arith::ConstantOp>(loc, zeroAttr);
  // flow.tensor.splat 0
  auto splatOp = initializerBuilder.create<IREE::Flow::TensorSplatOp>(
      loc, tensorType, zeroOp, /*result_dims=*/ValueRange{});
  // hal.tensor.export
  auto bufferExportOp = initializerBuilder.create<IREE::HAL::TensorExportOp>(
      loc, globalOp.getType(), splatOp.getResult(),
      TypeAttr::get(splatOp.getType()), /*name=*/nullptr);
  // util.optimization_barrier (try to prevent optimizations across the export)
  auto barrierOp = initializerBuilder.create<IREE::Util::OptimizationBarrierOp>(
      loc, bufferExportOp.getTarget());
  // util.global.store
  initializerBuilder.create<IREE::Util::GlobalStoreOp>(
      loc, barrierOp.getResult(0), globalOp.getName());
  initializerBuilder.create<IREE::Util::ReturnOp>(loc);

  return globalOp;
}

// Creates a util.global with a buffer view initialized to the same parameters
// as |arg| is used by. Fails if analysis cannot find a hal.tensor.import op.
//
// Example:
//  %0 = hal.tensor.import %arg : !hal.buffer_view -> tensor<4xi32>
// ->
//  util.global @some_fn_arg0 : !hal.buffer_view
//  util.initializer { ... }
static IREE::Util::GlobalOp
createImportBufferViewGlobalOp(std::string name, BlockArgument arg,
                               SymbolTable &symbolTable,
                               OpBuilder &moduleBuilder, Explorer &explorer) {
  auto loc = arg.getLoc();

  // Find a hal.tensor.import user.
  IREE::HAL::TensorImportOp importOp;
  if (explorer.walkTransitiveUsers(arg, [&](Operation *op) -> WalkResult {
        importOp = dyn_cast<IREE::HAL::TensorImportOp>(op);
        return importOp ? WalkResult::interrupt() : WalkResult::advance();
      }) == TraversalResult::INCOMPLETE) {
    // Analysis failed to find an import op. User needs to rework their program.
    mlir::emitError(loc) << "unsupported dynamic buffer view import on " << arg;
    return {};
  }

  // Extract the type, which must be a static tensor.
  auto targetType = importOp.getTarget().getType();
  auto tensorType = llvm::dyn_cast<RankedTensorType>(targetType);
  if (!tensorType || !tensorType.hasStaticShape()) {
    mlir::emitError(loc) << "unsupported buffer view import tensor type on "
                         << arg << " used as " << targetType;
    return {};
  }

  // Create the global and initialize it by allocating a zeroed buffer.
  return createBufferLikeGlobalOp(name, loc, arg.getType(), tensorType,
                                  symbolTable, moduleBuilder);
}

// Creates a util.global with a buffer initialized to the required storage
// capacity |arg| is used by. Fails if analysis cannot find a hal.tensor.export
// op.
//
// Example:
//  %1 = hal.tensor.export %0 into(%storage) : tensor<4xi32> -> !hal.buffer_view
// ->
//  util.global @some_fn_arg0 : !hal.buffer
//  util.initializer { ... }
static IREE::Util::GlobalOp createExportBufferGlobalOp(std::string name,
                                                       BlockArgument arg,
                                                       SymbolTable &symbolTable,
                                                       OpBuilder &moduleBuilder,
                                                       Explorer &explorer) {
  auto loc = arg.getLoc();

  // Find a hal.tensor.export user.
  IREE::HAL::TensorExportOp exportOp;
  if (explorer.walkTransitiveUsers(arg, [&](Operation *op) -> WalkResult {
        exportOp = dyn_cast<IREE::HAL::TensorExportOp>(op);
        return exportOp ? WalkResult::interrupt() : WalkResult::advance();
      }) == TraversalResult::INCOMPLETE) {
    // Analysis failed to find an export op. User needs to rework their program.
    mlir::emitError(loc) << "unsupported dynamic buffer view export on " << arg;
    return {};
  }

  // Extract the type, which must be a static tensor.
  auto sourceType = exportOp.getSourceEncoding();
  auto tensorType = llvm::dyn_cast<RankedTensorType>(sourceType);
  if (!tensorType || !tensorType.hasStaticShape()) {
    mlir::emitError(loc) << "unsupported buffer view export tensor type on "
                         << arg << " used as " << sourceType;
    return {};
  }

  // Create the global and initialize it by allocating a zeroed buffer.
  return createBufferLikeGlobalOp(name, loc, arg.getType(), tensorType,
                                  symbolTable, moduleBuilder);
}

static IREE::Util::GlobalOp createDummyInput(const std::string &namePrefix,
                                             BlockArgument arg,
                                             SymbolTable &symbolTable,
                                             OpBuilder &moduleBuilder,
                                             Explorer &explorer) {
  std::string name = namePrefix + "_arg" + std::to_string(arg.getArgNumber());
  return TypeSwitch<Type, IREE::Util::GlobalOp>(arg.getType())
      .Case([&](IREE::HAL::BufferViewType type) {
        return createImportBufferViewGlobalOp(name, arg, symbolTable,
                                              moduleBuilder, explorer);
      })
      .Case([&](IREE::HAL::BufferType type) {
        return createExportBufferGlobalOp(name, arg, symbolTable, moduleBuilder,
                                          explorer);
      })
      .Default([&](Type type) {
        return createPrimitiveDefaultGlobalOp(name, arg.getLoc(), type,
                                              symbolTable, moduleBuilder);
      });
}

static LogicalResult
createEntryPointBenchmarkFunc(mlir::ModuleOp moduleOp,
                              mlir::func::FuncOp entryFuncOp,
                              Explorer &explorer) {
  auto symbolTable = explorer.getSymbolTables().getSymbolTable(moduleOp);
  OpBuilder moduleBuilder(moduleOp.getContext());
  moduleBuilder.setInsertionPointAfter(entryFuncOp);

  // We'll create all symbols based on this name prefix.
  auto funcName = std::string(entryFuncOp.getName()) + "_benchmark";

  // Create one dummy input variable per input. We may need to do some
  // analysis to find the actual type and initial value.
  SmallVector<IREE::Util::GlobalOp> dummyInputVariableOps;
  for (auto arg : entryFuncOp.getArguments()) {
    auto dummyVar =
        createDummyInput(funcName, arg, symbolTable, moduleBuilder, explorer);
    if (!dummyVar)
      return failure();
    dummyInputVariableOps.push_back(dummyVar);
  }

  // Create a `() -> ()` entry point op the benchmark tool can run.
  Location loc = entryFuncOp.getLoc();
  auto funcOp = moduleBuilder.create<mlir::func::FuncOp>(
      loc, funcName, moduleBuilder.getFunctionType({}, {}));
  funcOp.setPublic();
  funcOp->setAttr("iree.abi.stub", moduleBuilder.getUnitAttr());
  SmallVector<NamedAttribute> reflectionAttrs = {
      moduleBuilder.getNamedAttr("iree.benchmark",
                                 moduleBuilder.getStringAttr("entry")),
  };
  funcOp->setAttr("iree.reflection",
                  moduleBuilder.getDictionaryAttr(reflectionAttrs));
  Block *block = funcOp.addEntryBlock();

  // Call the existing function with dummy arguments.
  auto blockBuilder = OpBuilder::atBlockBegin(block);
  SmallVector<Value> args;
  for (int i = 0, e = entryFuncOp.getNumArguments(); i < e; ++i) {
    args.push_back(blockBuilder.createOrFold<IREE::Util::GlobalLoadOp>(
        loc, dummyInputVariableOps[i]));
  }
  auto callOp = blockBuilder.create<mlir::func::CallOp>(loc, entryFuncOp, args);

  // Sink all results with a barrier to ensure that DCE does not remove the
  // call.
  for (auto result : callOp.getResults()) {
    blockBuilder.create<IREE::Util::OptimizationBarrierOp>(loc, result);
  }
  blockBuilder.create<mlir::func::ReturnOp>(loc);

  // Ensure the original function is not exported and not inlined.
  entryFuncOp->setAttr("noinline", moduleBuilder.getUnitAttr());
  entryFuncOp->removeAttr("iree.reflection");
  entryFuncOp.setPrivate();

  return success();
}

// Clones each exported functions (including those just created) with
// placeholder constant inputs instead of arguments and removes the exported
// attribute from the old functions.
// The input are provided using util.globals.
class ExportBenchmarkFuncsPass
    : public ExportBenchmarkFuncsBase<ExportBenchmarkFuncsPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, IREE::Flow::FlowDialect,
                    IREE::HAL::HALDialect, IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // For analysis required on arguments and results.
    Explorer explorer(moduleOp, TraversalAction::SHALLOW);

    // Gather the functions we want to wrap for benchmarking and wrap them.
    // Since we are inserting new functions as part of this pass we must perform
    // the wrapping for only the inputs.
    SmallVector<mlir::func::FuncOp> entryFuncOps;
    for (auto entryFuncOp : moduleOp.getOps<mlir::func::FuncOp>()) {
      if (entryFuncOp.isPublic()) {
        entryFuncOps.push_back(entryFuncOp);
      }
    }
    for (auto entryFuncOp : entryFuncOps) {
      if (failed(
              createEntryPointBenchmarkFunc(moduleOp, entryFuncOp, explorer))) {
        signalPassFailure();
        return;
      }
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createExportBenchmarkFuncsPass() {
  return std::make_unique<ExportBenchmarkFuncsPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow
