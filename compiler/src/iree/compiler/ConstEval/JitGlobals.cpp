// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ConstEval/PassDetail.h"
#include "iree/compiler/ConstEval/Passes.h"
#include "iree/compiler/ConstEval/Runtime.h"
#include "iree/compiler/Pipelines/Pipelines.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#include <cstdlib>

#define DEBUG_TYPE "iree-const-eval"
using llvm::dbgs;

namespace mlir::iree_compiler::ConstEval {

static llvm::cl::opt<std::string> clJitTargetBackend(
    "iree-consteval-jit-target-backend",
    llvm::cl::desc("Overrides the target backend used for JIT'ing."),
    llvm::cl::init(""));

static llvm::cl::opt<bool> clEnableDebug(
    "iree-consteval-jit-debug",
    llvm::cl::desc(
        "Prints debugging information to stderr (useful since when consteval "
        "has issues, it is often in production on the largest models where we "
        "don't want to run a debug compiler)."),
    llvm::cl::init(false));

namespace {

static bool isDebugEnabled() {
  if (clEnableDebug)
    return true;
  if (std::getenv("IREE_COMPILER_DEBUG_CONSTEVAL"))
    return true;
  return false;
}

// These options structs are not copy-constructable so we have to allocate them
// shared.
// TODO: See if we can make them copyable?
struct CompileOptions {
  BindingOptions bindingOptions;
  InputDialectOptions inputOptions;
  PreprocessingOptions preprocessingOptions;
  GlobalOptimizationOptions globalOptimizationOptions;
  SchedulingOptions schedulingOptions;
  IREE::HAL::TargetOptions executableOptions;
  IREE::VM::TargetOptions targetOptions;
  IREEVMPipelineHooks hooks;
};

// Supported types vary by backend and other factors, so we track them here.
// Types that cross the ABI boundary are configured here.
class SupportedFeatures {
public:
  void addScalarType(Type t) { scalarTypes.insert(t); }
  void addElementType(Type t) { elementTypes.insert(t); }

  bool supportsScalarType(Type t) const { return scalarTypes.contains(t); }

  bool supportsElementType(Type t) const { return elementTypes.contains(t); }

  bool isSupportedAbiType(Type t) const {
    if (auto tensorType = llvm::dyn_cast<TensorType>(t)) {
      return supportsElementType(tensorType.getElementType());
    } else {
      return supportsScalarType(t);
    }
  }

private:
  llvm::DenseSet<Type> scalarTypes;
  llvm::DenseSet<Type> elementTypes;
};

// JIT functions take arguments, generally from the source program. We capture
// them here.
class ArgumentBinding {
public:
  enum class Type {
    // An ElementsAttr.
    ElementsAttr,

    // The value of a GlobalOp. It may not be set at the start of the run
    // if there is a dependency that evaluates first.
    GlobalOp,
  };

  ArgumentBinding(ElementsAttr attr)
      : type(Type::ElementsAttr), elementsAttr(attr) {}
  ArgumentBinding(IREE::Util::GlobalOp globalOp)
      : type(Type::GlobalOp), globalOp(globalOp) {}

  Type getType() { return type; }

  ElementsAttr getElementsAttr() {
    assert(type == Type::ElementsAttr);
    return elementsAttr;
  }

  IREE::Util::GlobalOp getGlobalOp() {
    assert(type == Type::GlobalOp);
    return globalOp;
  }

private:
  Type type;
  ElementsAttr elementsAttr;
  IREE::Util::GlobalOp globalOp;
};

// How to bind results to the original program.
class ResultBinding {
public:
  enum class Type {
    // Set the result on the global op.
    GlobalOp,
  };

  ResultBinding(IREE::Util::GlobalOp globalOp)
      : type(Type::GlobalOp), globalOp(globalOp) {}

  Type getType() { return type; }

  IREE::Util::GlobalOp getGlobalOp() {
    assert(type == Type::GlobalOp);
    return globalOp;
  }

private:
  Type type;
  ElementsAttr elementsAttr;
  IREE::Util::GlobalOp globalOp;
};

// Description of a JIT function that we have created for doing some
// initialization work.
struct JitFunctionDesc {
  JitFunctionDesc(Location loc, std::string name)
      : loc(loc), name(std::move(name)) {}
  Location loc;
  std::string name;
  llvm::SmallVector<ArgumentBinding> argumentBindings;
  llvm::SmallVector<ResultBinding> resultBindings;
};

class ProgramBuilder {
public:
  ProgramBuilder(ModuleOp sourceModuleOp,
                 const SupportedFeatures &supportedFeatures)
      : targetModuleOp(createInnerModule(sourceModuleOp)),
        sourceSymbolTable(sourceModuleOp), targetSymbolTable(targetModuleOp),
        supportedFeatures(supportedFeatures) {}

  llvm::SmallVector<JitFunctionDesc> &getJitFunctions() { return jitFunctions; }
  ModuleOp getTargetModule() { return targetModuleOp; }

  LogicalResult importInitializer(IREE::Util::InitializerOp initOp) {
    //  We convert each initializer into a public FuncOp by converting each:
    //    - Tensor constant into an argument
    //    - util.global_load into an argument
    //    - util.global_store into a result
    //  It is considered an eval'able initializer if it contains stores
    //  into immutable global(s). In the future, we will also want to
    //  condition this on an attribute so as to not try to statically
    //  compile dynamic initializers.
    // Build it into a new function.
    if (!initOp.getBody().hasOneBlock()) {
      // It would be possible to support these in theory but unclear if
      // worth it in practice.
      emitWarning(initOp.getLoc())
          << "skipping consteval initializer: initializers with >1 block not "
             "yet supported";
      return failure();
    }

    OpBuilder moduleBuilder = OpBuilder::atBlockEnd(targetModuleOp.getBody());
    auto funcOp = moduleBuilder.create<func::FuncOp>(
        initOp.getLoc(), "jit_eval", moduleBuilder.getFunctionType({}, {}));
    targetSymbolTable.insert(funcOp);
    IRMapping unusedMapping;
    initOp.getBody().cloneInto(&funcOp.getBody(), unusedMapping);
    if (failed(transformToJitFunction(funcOp))) {
      funcOp.erase();
      return failure();
    }

    return success();
  }

private:
  static ModuleOp createInnerModule(ModuleOp sourceModuleOp) {
    OpBuilder builder = OpBuilder::atBlockEnd(sourceModuleOp.getBody());
    auto m = builder.create<ModuleOp>(sourceModuleOp.getLoc());
    m->setAttr("iree.consteval", builder.getUnitAttr());
    return m;
  }

  LogicalResult transformToJitFunction(func::FuncOp funcOp) {
    JitFunctionDesc desc(funcOp.getLoc(), funcOp.getName().str());
    llvm::SmallVector<Type> argumentTypes;
    llvm::SmallVector<Type> returnTypes;
    llvm::SmallVector<Value> returns;
    llvm::SmallVector<Operation *> eraseOps;

    Block *entryBlock = &funcOp.getBody().front();

    // Find immutable loads.
    for (auto loadOp : funcOp.getOps<IREE::Util::GlobalLoadOp>()) {
      auto globalOp = llvm::dyn_cast_or_null<IREE::Util::GlobalOp>(
          sourceSymbolTable.lookup(loadOp.getGlobalAttr().getAttr()));
      if (!globalOp || globalOp.getIsMutable()) {
        emitWarning(loadOp.getLoc()) << "skipping consteval initializer: load "
                                        "from mutable globals not supported";
        return failure();
      }
      Type t = loadOp.getResult().getType();
      if (!supportedFeatures.isSupportedAbiType(t)) {
        emitWarning(funcOp.getLoc())
            << "skipping consteval initializer: unsupported type for current "
               "jit configuration: "
            << t;
        return failure();
      }
      argumentTypes.push_back(t);
      BlockArgument entryArg = entryBlock->addArgument(t, loadOp.getLoc());
      loadOp.getResult().replaceAllUsesWith(entryArg);
      eraseOps.push_back(loadOp);
      desc.argumentBindings.emplace_back(globalOp);
    }

    // And loose tensor constants.
    for (auto constantOp : funcOp.getOps<arith::ConstantOp>()) {
      auto tensorType = constantOp.getResult().getType().dyn_cast<TensorType>();
      auto elementsAttr = constantOp.getValue().dyn_cast<ElementsAttr>();
      if (!tensorType || !elementsAttr)
        continue;
      if (!supportedFeatures.isSupportedAbiType(tensorType)) {
        emitWarning(funcOp.getLoc())
            << "skipping consteval initializer: unsupported type for current "
               "jit configuration: "
            << tensorType;
        return failure();
      }
      argumentTypes.push_back(tensorType);
      BlockArgument entryArg =
          entryBlock->addArgument(tensorType, constantOp.getLoc());
      constantOp.getResult().replaceAllUsesWith(entryArg);
      eraseOps.push_back(constantOp);
      desc.argumentBindings.emplace_back(elementsAttr);
    }

    // Find immutable stores, early exiting if not supported.
    // The consumers must come after rewrites of the producers above.
    for (auto storeOp : funcOp.getOps<IREE::Util::GlobalStoreOp>()) {
      auto globalOp = llvm::dyn_cast_or_null<IREE::Util::GlobalOp>(
          sourceSymbolTable.lookup(storeOp.getGlobalAttr().getAttr()));
      if (!globalOp || globalOp.getIsMutable()) {
        emitWarning(storeOp.getLoc()) << "skipping consteval initializer: stor "
                                         "to mutable globals not supported";
        return failure();
      }
      Type t = storeOp.getValue().getType();
      if (!supportedFeatures.isSupportedAbiType(t)) {
        emitWarning(funcOp.getLoc())
            << "skipping consteval initializer: unsupported type for current "
               "jit configuration: "
            << t;
        return failure();
      }

      returns.push_back(storeOp.getValue());
      returnTypes.push_back(t);
      eraseOps.push_back(storeOp);
      desc.resultBindings.emplace_back(globalOp);
    }

    // Cleanup.
    for (auto *op : eraseOps) {
      op->erase();
    }

    // Rewrite the terminator and the function type.
    entryBlock->getTerminator()->erase();
    OpBuilder termBuilder = OpBuilder::atBlockEnd(entryBlock);
    termBuilder.create<func::ReturnOp>(funcOp.getLoc(), returns);
    funcOp.setType(termBuilder.getFunctionType(argumentTypes, returnTypes));

    jitFunctions.push_back(std::move(desc));
    return success();
  }

  ModuleOp targetModuleOp;
  SymbolTable sourceSymbolTable;
  SymbolTable targetSymbolTable;
  llvm::SmallVector<JitFunctionDesc> jitFunctions;
  const SupportedFeatures &supportedFeatures;
};

struct JitGlobalsPass : public JitGlobalsBase<JitGlobalsPass> {
  JitGlobalsPass(const IREE::HAL::TargetBackendRegistry &targetRegistry)
      : options(std::make_shared<CompileOptions>()),
        compilePipeline("builtin.module") {
    // Detect backend.
    requestedTargetBackend = resolveTargetBackend(targetRegistry);
    hasRequestedTargetBackend =
        targetRegistry.getTargetBackend(requestedTargetBackend) != nullptr;
    options->executableOptions.targets.push_back(requestedTargetBackend);
    options->targetOptions.f32Extension = true;
    options->targetOptions.f64Extension = true;
    options->targetOptions.truncateUnsupportedFloats = false;
    if (requestedTargetBackend == "vmvx" || !hasRequestedTargetBackend) {
      targetBackend = targetRegistry.getTargetBackend("vmvx");
    } else {
      targetBackend = targetRegistry.getTargetBackend(requestedTargetBackend);
    }

    // Disable constant evaluation for our Jit compilation pipeline.
    // It would make no sense to recursively do constant evaluation, and since
    // we omit the necessary hooks, it is unsupported anyway.
    options->globalOptimizationOptions.constExprHoisting = false;
    options->globalOptimizationOptions.constEval = false;

    buildIREEVMTransformPassPipeline(
        targetRegistry, options->bindingOptions, options->inputOptions,
        options->preprocessingOptions, options->globalOptimizationOptions,
        options->schedulingOptions, options->executableOptions,
        options->targetOptions, options->hooks, compilePipeline);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    compilePipeline.getDependentDialects(registry);
  }

  static std::string
  resolveTargetBackend(const IREE::HAL::TargetBackendRegistry &targetRegistry) {
    if (clJitTargetBackend.empty()) {
      // Default - choose something we have.
      // First llvm-cpu then vmvx.
      if (targetRegistry.getTargetBackend("llvm-cpu")) {
        return std::string("llvm-cpu");
      } else {
        return std::string("vmvx");
      }
    }

    return clJitTargetBackend;
  }

  const SupportedFeatures getSupportedFeatures(MLIRContext *context) {
    SupportedFeatures s;
    Builder b(context);
    s.addScalarType(b.getIntegerType(8));
    s.addScalarType(b.getIntegerType(16));
    s.addScalarType(b.getIntegerType(32));
    s.addScalarType(b.getIntegerType(64));
    s.addScalarType(b.getF32Type());

    s.addElementType(b.getIntegerType(1));
    s.addElementType(b.getIntegerType(8));
    s.addElementType(b.getIntegerType(16));
    s.addElementType(b.getIntegerType(32));
    s.addElementType(b.getIntegerType(64));
    s.addElementType(b.getF32Type());
    if (requestedTargetBackend != "vmvx" && hasRequestedTargetBackend) {
      // The full compilers support additional types.
      // TODO: Enable support for i4 once it is worked out how to
      // transfer to and from ElementsAttr.
      s.addScalarType(b.getF64Type());
      s.addElementType(b.getF16Type());
      s.addElementType(b.getBF16Type());
      s.addElementType(b.getF64Type());
    }
    return s;
  }

  LogicalResult
  processFunctions(CompiledBinary &binary,
                   llvm::SmallVector<JitFunctionDesc> &jitFunctions,
                   ModuleOp module, llvm::TimerGroup &tg) {
    // Process each function through the runtime.
    for (JitFunctionDesc &jitFunction : jitFunctions) {
      std::optional<llvm::Timer> invokeTimer;
      if (debugEnabled) {
        std::string timerName("Invoke ");
        timerName.append(jitFunction.name);
        invokeTimer.emplace(timerName, timerName, tg);
        invokeTimer->startTimer();
        dbgs() << "::: Invoking " << jitFunction.name << "\n";
      }

      FunctionCall call(binary, jitFunction.argumentBindings.size(),
                        jitFunction.resultBindings.size());

      // Convert arguments.
      for (ArgumentBinding &arg : jitFunction.argumentBindings) {
        switch (arg.getType()) {
        case ArgumentBinding::Type::ElementsAttr:
          if (failed(call.addArgument(jitFunction.loc, arg.getElementsAttr())))
            return failure();
          break;

        case ArgumentBinding::Type::GlobalOp: {
          auto globalValue = arg.getGlobalOp().getInitialValue();
          if (!globalValue) {
            return emitError(jitFunction.loc)
                   << "internal error: jit global source initialization order. "
                      "global "
                   << arg.getGlobalOp().getSymName() << " has no value";
          }
          if (failed(
                  call.addArgument(arg.getGlobalOp().getLoc(), *globalValue)))
            return failure();
        } break;
        }
      }

      if (failed(call.invoke(jitFunction.loc, jitFunction.name))) {
        return failure();
      }

      // Process results.
      for (auto it : llvm::enumerate(jitFunction.resultBindings)) {
        ResultBinding &resultBinding = it.value();
        switch (resultBinding.getType()) {
        case ResultBinding::Type::GlobalOp: {
          TypedAttr attr;
          if (failed(call.getResultAsAttr(
                  resultBinding.getGlobalOp().getLoc(), it.index(),
                  resultBinding.getGlobalOp().getType(), attr)))
            return failure();
          resultBinding.getGlobalOp().setInitialValueAttr(attr);
          break;
        }
        }
      }

      if (debugEnabled) {
        invokeTimer->stopTimer();
      }
    }

    return success();
  }

  void runOnOperation() override {
    llvm::TimerGroup tg("iree-consteval-jit", "Consteval Jit");
    auto outerModule = getOperation();
    auto supportedFeatures = getSupportedFeatures(&getContext());
    if (!hasRequestedTargetBackend) {
      emitWarning(UnknownLoc::get(&getContext()))
          << "consteval jit requested with " << requestedTargetBackend
          << " backend, but it is not available. Falling back to vmvx";
    }
    if (!targetBackend) {
      emitError(UnknownLoc::get(&getContext()))
          << "consteval jit could not find a usable backend (requested '"
          << requestedTargetBackend << "')";
      signalPassFailure();
      return;
    }

    llvm::SmallVector<IREE::Util::InitializerOp> initOps;
    llvm::SmallVector<IREE::Util::InitializerOp> deadInitOps;
    for (auto childOp : outerModule.getOps<IREE::Util::InitializerOp>()) {
      initOps.push_back(childOp);
    }

    // Build the program.
    ProgramBuilder programBuilder(outerModule, supportedFeatures);

    // Set the target.
    std::optional<IREE::HAL::DeviceTargetAttr> targetAttr =
        targetBackend->getHostDeviceTarget(&getContext());
    {
      if (!targetAttr) {
        emitError(UnknownLoc::get(&getContext()))
            << "consteval requested backend " << requestedTargetBackend
            << " cannot target the host";
        signalPassFailure();
        return;
      }
      SmallVector<Attribute> targetAttrs;
      targetAttrs.push_back(*targetAttr);
      programBuilder.getTargetModule()->setAttr(
          "hal.device.targets", ArrayAttr::get(&getContext(), targetAttrs));
    }

    // Iterate over initializers.
    for (auto initOp : initOps) {
      if (!initOp->hasAttr("iree.compiler.consteval"))
        continue;

      if (succeeded(programBuilder.importInitializer(initOp))) {
        deadInitOps.push_back(initOp);
      } else if (debugEnabled) {
        dbgs() << "::: Rejected consteval initializer:\n" << initOp << "\n";
      }
    }
    if (programBuilder.getJitFunctions().empty()) {
      programBuilder.getTargetModule()->erase();
      return;
    }

    std::optional<llvm::Timer> compileTimer;
    if (debugEnabled) {
      dbgs() << "::: COMPILING JIT (" << requestedTargetBackend
             << "): " << programBuilder.getTargetModule() << "\n";
      compileTimer.emplace("iree-consteval-jit-compile", "Compiling", tg);
      compileTimer->startTimer();
    }
    if (failed(
            runPipeline(compilePipeline, programBuilder.getTargetModule()))) {
      return signalPassFailure();
    }
    // Generate a binary.
    InMemoryCompiledBinary binary;
    if (failed(binary.translateFromModule(programBuilder.getTargetModule()))) {
      return signalPassFailure();
    }
    if (debugEnabled) {
      compileTimer->stopTimer();
    }

    // Kill the temporary program.
    programBuilder.getTargetModule()->erase();

    // Process the functions.
    if (failed(processFunctions(binary, programBuilder.getJitFunctions(),
                                outerModule, tg))) {
      signalPassFailure();
      return;
    }

    // Cleanup any initializers we replaced.
    // We do this after running the JIT-ed functions because we have deep
    // references into ops and attributes that need to be converted to
    // arguments.
    for (auto deadOp : deadInitOps) {
      deadOp.erase();
    }
  }

  std::shared_ptr<CompileOptions> options;
  OpPassManager compilePipeline;
  std::string requestedTargetBackend;
  std::shared_ptr<IREE::HAL::TargetBackend> targetBackend;
  bool hasRequestedTargetBackend;
  bool debugEnabled = isDebugEnabled();
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createJitGlobalsPass(const IREE::HAL::TargetBackendRegistry &targetRegistry) {
  return std::make_unique<JitGlobalsPass>(targetRegistry);
}

std::unique_ptr<OperationPass<ModuleOp>> createJitGlobalsPass() {
  return std::make_unique<JitGlobalsPass>(
      IREE::HAL::TargetBackendRegistry::getGlobal());
}

} // namespace mlir::iree_compiler::ConstEval
