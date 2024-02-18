// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Utils/IndexSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::Flow {

namespace {

// Appends a function calling the given |exportOp| to the module.
static void buildDispatchExportFunction(IREE::Flow::ExecutableOp executableOp,
                                        IREE::Flow::ExecutableExportOp exportOp,
                                        OpBuilder &moduleBuilder) {
  auto loc = executableOp.getLoc();

  std::string baseName =
      (executableOp.getName() + "_" + exportOp.getName()).str();

  auto exportedFunc = llvm::dyn_cast_if_present<FunctionOpInterface>(
      SymbolTable::lookupSymbolIn(executableOp.getInnerModule(),
                                  exportOp.getName()));
  if (!exportedFunc || !exportedFunc.getResultTypes().empty()) {
    return;
  }

  SmallVector<Type> argumentTypes;
  for (auto workloadType : exportOp.getWorkgroupCount().getArgumentTypes()) {
    if (!workloadType.isIndex()) {
      // Assume 64 bit index to be safe.
      argumentTypes.push_back(moduleBuilder.getI64Type());
    } else {
      argumentTypes.push_back(workloadType);
    }
  }
  SmallVector<Type> resultTypes;
  SmallVector<Type> resultDimTypes;
  SmallVector<int64_t> tiedArguments;
  int64_t numTensorArgs = 0;
  for (auto argType : exportedFunc.getArgumentTypes()) {
    if (auto flowTensorType = dyn_cast<Flow::DispatchTensorType>(argType)) {
      RankedTensorType tensorType = flowTensorType.asRankedTensorType();
      // TODO: Support generating dynamic dispatches. Properly tying tensor
      // sizes to workload values is tricky. Alternatively we could require
      // users to pass values for all dynamic sizes manually, but there is
      // likely a better solution that doesn't involve trying to dispatch
      // the executable directly.
      if (!tensorType.hasStaticShape()) {
        return;
      }
      TensorAccess accessType = flowTensorType.getAccess();
      if (accessType != TensorAccess::WriteOnly) {
        argumentTypes.push_back(flowTensorType.asRankedTensorType());
        numTensorArgs++;
      }
      // Infer results as any writeable tensor.
      if (accessType != TensorAccess::ReadOnly) {
        resultTypes.push_back(flowTensorType.asRankedTensorType());
        tiedArguments.push_back(IREE::Util::TiedOpInterface::kUntiedIndex);
      }
      if (accessType == TensorAccess::ReadWrite) {
        tiedArguments.back() = numTensorArgs - 1;
      }
      continue;
    }
    if (!argType.isIndex()) {
      // Assume 64 bit index to be safe.
      argumentTypes.push_back(moduleBuilder.getI64Type());
    } else {
      argumentTypes.push_back(argType);
    }
  }

  // Create a function that runs the dispatches based on signature of the
  // exported function.
  auto funcType = moduleBuilder.getFunctionType(argumentTypes, resultTypes);
  auto funcOp =
      moduleBuilder.create<IREE::Util::FuncOp>(loc, baseName, funcType);
  funcOp.setVisibility(SymbolTable::Visibility::Public);

  // Build the function that runs the dispatch.
  auto *entryBlock = funcOp.addEntryBlock();
  OpBuilder funcBuilder = OpBuilder::atBlockBegin(entryBlock);
  IndexSet indexSet(loc, funcBuilder);

  int64_t numWorkgroupArgs = exportOp.getWorkgroupCount().getNumArguments();
  auto workloadArgs = entryBlock->getArguments().slice(0, numWorkgroupArgs);
  SmallVector<Value> workloadIndexArgs;
  for (auto [workloadType, workloadBlockArg] : llvm::zip_equal(
           exportOp.getWorkgroupCount().getArgumentTypes(), workloadArgs)) {
    if (workloadType.isIndex()) {
      auto workloadIndexArg = funcBuilder.create<arith::IndexCastOp>(
          loc, funcBuilder.getIndexType(), workloadBlockArg);
      workloadIndexArgs.push_back(workloadIndexArg);
    } else {
      workloadIndexArgs.push_back(workloadBlockArg);
    }
  }

  SmallVector<Value> arguments;
  // Slices off the first |numWorkgroupArgs| block arguments; the remaining
  // block arguments are the arguments to the dispatch.
  for (auto blockArg : entryBlock->getArguments().slice(numWorkgroupArgs)) {
    if (blockArg.getType().isIndex()) {
      arguments.push_back(funcBuilder.create<arith::IndexCastOp>(
          loc, funcBuilder.getIndexType(), blockArg));
      continue;
    }
    arguments.push_back(blockArg);
  }

  SmallVector<Attribute> tiedAttrs =
      llvm::map_to_vector<8>(tiedArguments, [&](int64_t v) -> Attribute {
        return IntegerAttr::get(funcBuilder.getIndexType(), v);
      });

  auto dispatchOp = funcBuilder.create<IREE::Flow::DispatchOp>(
      loc, exportOp, workloadIndexArgs, resultTypes,
      /*resultDims=*/SmallVector<Value>{}, arguments,
      /*argumentDims=*/SmallVector<Value>{},
      ArrayAttr::get(funcBuilder.getContext(), tiedAttrs));

  funcBuilder.create<IREE::Util::ReturnOp>(loc, dispatchOp.getResults());
}

// Builds a module exporting one function for each dispatch configuration
// targeting |sourceExecutableOp|.
static mlir::OwningOpRef<mlir::ModuleOp>
buildExecutableModule(IREE::Flow::ExecutableOp sourceExecutableOp) {
  // Empty module with default name.
  // We could use the original module name here to make tracking nicer.
  mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
      mlir::ModuleOp::create(sourceExecutableOp.getLoc());
  auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp->getBody());

  // Intentionally ignore the device targets for the current module. The
  // exported functions should be compilable for any target.

  // Clone the executable into the new module.
  auto executableOp =
      cast<IREE::Flow::ExecutableOp>(moduleBuilder.clone(*sourceExecutableOp));

  // Add functions to test each entry point with its various dispatch
  // parameters.
  auto exportOps = llvm::to_vector(
      executableOp.getBody().getOps<IREE::Flow::ExecutableExportOp>());
  for (auto exportOp :
       executableOp.getBody().getOps<IREE::Flow::ExecutableExportOp>()) {
    buildDispatchExportFunction(executableOp, exportOp, moduleBuilder);
  }

  // Run CSE and the canonicalizer to pretty up the output.
  PassManager passManager(moduleOp->getContext());
  passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addPass(mlir::createCSEPass());
  if (failed(passManager.run(*moduleOp))) {
    moduleOp->emitError("failed to run canonicalizer; malformed output");
    return {};
  }

  return moduleOp;
}

static void dumpModuleToStream(mlir::ModuleOp moduleOp, StringRef fileName,
                               llvm::raw_ostream &os) {
  OpPrintingFlags flags;
  flags.useLocalScope(); // could use global scope, but IR gets messy fast
  moduleOp.print(os, flags);
  os << "\n"; // newline at end of file
}

//===----------------------------------------------------------------------===//
// --iree-flow-dump-executable-functions
//===----------------------------------------------------------------------===//

/// Pass declaration.
struct DumpExecutableFunctionsPass
    : public DumpExecutableFunctionsPassBase<DumpExecutableFunctionsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, IREE::Util::UtilDialect,
                    tensor::TensorDialect>();
  }
  DumpExecutableFunctionsPass(std::string path) { this->path = path; }
  DumpExecutableFunctionsPass(const DumpExecutableFunctionsPass &pass)
      : DumpExecutableFunctionsPass(pass.path) {}
  void runOnOperation() override;
};

void DumpExecutableFunctionsPass::runOnOperation() {
  auto moduleOp = getOperation();
  auto moduleName = moduleOp.getName().value_or("module");

  // Help people out and mkdir if needed.
  if (!path.empty() && path != "-") {
    llvm::sys::fs::create_directories(path);
  }

  // Produce one file per executable containing all exported entry points.
  for (auto executableOp : moduleOp.getOps<IREE::Flow::ExecutableOp>()) {
    auto benchmarkModuleOp = buildExecutableModule(executableOp);
    if (!benchmarkModuleOp)
      continue;
    auto fileName =
        (moduleName + "_" + executableOp.getName() + "_function.mlir").str();
    if (path.empty() || path == "-") {
      dumpModuleToStream(*benchmarkModuleOp, fileName, llvm::outs());
    } else {
      auto filePath =
          (path + llvm::sys::path::get_separator() + fileName).str();
      std::string error;
      auto file = mlir::openOutputFile(filePath, &error);
      if (!file) {
        executableOp.emitError()
            << "while dumping to " << path << ": " << error;
        return signalPassFailure();
      }
      dumpModuleToStream(*benchmarkModuleOp, fileName, file->os());
      file->keep();
    }
  }
}

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createDumpExecutableFunctionsPass(std::string path) {
  return std::make_unique<DumpExecutableFunctionsPass>(path);
}

} // namespace mlir::iree_compiler::IREE::Flow
