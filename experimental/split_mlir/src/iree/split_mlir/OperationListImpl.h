// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "iree/compiler/Tools/init_dialects.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree {
namespace split_mlir {

using OpId = std::string;
using OpIndex = size_t;
using ResultIndex = size_t;
using ResultId = std::tuple<OpIndex, ResultIndex>;
using Arguments = std::vector<ResultId>;
using OperationList = std::vector<std::tuple<OpId, Arguments>>;

ResultIndex getResultIndex(OpOperand& operand) {
  OpResult opResult = operand.get().dyn_cast<OpResult>();
  if (opResult) {
    return opResult.getResultNumber();
  }

  BlockArgument blockArgument = operand.get().dyn_cast<BlockArgument>();
  assert(blockArgument);
  return blockArgument.getArgNumber();
}

FailureOr<OpIndex> getDefiningOpIndex(
    OpOperand& operand, Block& block,
    const std::unordered_map<Operation*, size_t>& operationInBlockIndexMap) {
  Value value = operand.get();
  if (value.isa<BlockArgument>()) {
    return 0;
  }

  OpResult opResult = value.dyn_cast<OpResult>();
  if (!opResult) {
    operand.getOwner()->emitError(
        Twine("Operand ") + std::to_string(operand.getOperandNumber()) +
        "is  neigher a block argument or a result of an operation");
    return failure();
  }
  if (value.getDefiningOp()->getBlock() != &block) {
    operand.getOwner()->emitError(
        "Can't extract call graph for block that is not isolated from above.");
    return failure();
  }

  auto it = operationInBlockIndexMap.find(value.getDefiningOp());
  assert(it != operationInBlockIndexMap.end());
  return it->second;
}

std::string getOpId(Operation& op) {
  func::CallOp callOp = dyn_cast<func::CallOp>(op);
  if (callOp) {
    return (Twine("call ") + callOp.getCallee()).str();
  }

  if (isa<func::ReturnOp>(op)) {
    return "return";
  }

  return op.getName().getStringRef().str();
}

FailureOr<OperationList> extractOperationList(Block& block) {
  OperationList res;
  // Block arguments don't depend on anything.
  res.emplace_back();
  // Index inside the block.
  std::unordered_map<Operation*, size_t> operationInBlockIndexMap;

  for (auto opIt : llvm::enumerate(block)) {
    operationInBlockIndexMap.insert({&opIt.value(), opIt.index() + 1});
    OpId id = getOpId(opIt.value());
    Arguments arguments;
    for (OpOperand& operand : opIt.value().getOpOperands()) {
      FailureOr<OpIndex> opIndex =
          getDefiningOpIndex(operand, block, operationInBlockIndexMap);
      FailureOr<ResultIndex> resultIndex = getResultIndex(operand);
      if (failed(opIndex) || failed(resultIndex)) {
        return failure();
      }
      arguments.emplace_back(opIndex.value(), resultIndex.value());
    }
    res.emplace_back(id, arguments);
  }

  return res;
}

FailureOr<OwningOpRef<ModuleOp>> loadMlir(const char* mlirFilePath,
                                          MLIRContext& context) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(mlirFilePath);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return failure();
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, &context);
}

func::FuncOp findFunction(Operation* root, StringRef name) {
  func::FuncOp res;
  root->walk([&](func::FuncOp op) {
    if (op.getSymName() == name) {
      res = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return res;
}

FailureOr<OperationList> extractOperationList(ModuleOp moduleOp,
                                              StringRef functionName) {
  func::FuncOp funcOp = findFunction(moduleOp.getOperation(), functionName);
  Region* region = funcOp.getCallableRegion();
  if (!region) {
    funcOp.emitError("No callable region found.");
    return failure();
  }
  if (region->getBlocks().size() != 1) {
    funcOp.emitError("Blocks count must be exactly 1.");
    return failure();
  }
  return extractOperationList(region->front());
}

FailureOr<OperationList> extractOperationList(const char* mlirFilePath,
                                              StringRef functionName,
                                              MLIRContext& context) {
  auto moduleOp = loadMlir(mlirFilePath, context);
  if (failed(moduleOp)) {
    return failure();
  }

  return extractOperationList(moduleOp->get(), functionName);
}

std::unique_ptr<mlir::MLIRContext> makeMlirContext() {
  mlir::DialectRegistry registry;
  mlir::iree_compiler::registerAllDialects(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  return context;
}

FailureOr<OperationList> extractOperationList(const char* mlirFilePath,
                                              StringRef functionName) {
  auto context = makeMlirContext();
  return extractOperationList(mlirFilePath, functionName, *context);
}

}  // namespace split_mlir
}  // namespace iree
}  // namespace mlir
