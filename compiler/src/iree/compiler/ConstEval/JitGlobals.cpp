// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ConstEval/PassDetail.h"
#include "iree/compiler/ConstEval/Passes.h"
#include "iree/compiler/ConstEval/Runtime.h"
#include "iree/compiler/Dialect/HAL/Target/TargetOptions.h"
#include "iree/compiler/Dialect/Util/Analysis/Constant/ConstExpr.h"
#include "iree/compiler/Dialect/Util/Analysis/Constant/OpOracle.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Pipelines/Pipelines.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#include <cstdlib>

#define DEBUG_TYPE "iree-const-eval"

namespace mlir::iree_compiler::ConstEval {

static llvm::cl::opt<std::string> clJitTargetDevice(
    "iree-consteval-jit-target-device",
    llvm::cl::desc("Overrides the target device used for JIT'ing."),
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

template <typename AccessorTy>
static inline bool isAccessorParameterized(const SymbolTable &moduleSymbols,
                                           AccessorTy op) {
  auto global =
      moduleSymbols.lookup<IREE::Util::GlobalOpInterface>(op.getGlobalName());
  if (!global)
    return true;
  auto attr = global.getGlobalInitialValue();
  if (!attr)
    return false;
  return !isa<IntegerAttr>(attr) && !isa<FloatAttr>(attr) &&
         !isa<IREE::Util::SerializableAttrInterface>(
             global.getGlobalInitialValue());
}

// Today the only way to interact with a global is with loads, stores, and
// addresses, and globals are the only way to reference parameters given where
// const-eval is run today. This is a workaround until we have proper dialect
// interfaces for detecting whether something is evaluatable at compile time.
static bool isParameterized(const SymbolTable &moduleSymbols,
                            IREE::Util::InitializerOpInterface initializerOp) {
  WalkResult res = initializerOp->walk([&](Operation *op) {
    const bool parameterized =
        llvm::TypeSwitch<Operation *, bool>(op)
            .Case([=](IREE::Util::GlobalLoadOpInterface accessor) {
              return isAccessorParameterized(moduleSymbols, accessor);
            })
            .Case([=](IREE::Util::GlobalStoreOpInterface accessor) {
              return isAccessorParameterized(moduleSymbols, accessor);
            })
            .Default([=](auto) { return false; });
    if (parameterized)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return res.wasInterrupted();
}

// WIP specialized analysis for tracking initialization order in a module.
// This attempts to provide a "is this value initialized?" query with the
// differentiation of whether that initialization is possible within the
// compiler or if it relies on runtime information.
//
// This is currently fairly limited and bails on many common cases that we don't
// naturally generate in early phases of program compilation. More sophisticated
// analysis is required to use this elsewhere once calls, control flow, and
// more dynamic values are used.
class InitializationAnalysis {
public:
  enum class Availability {
    // Analysis failure, assume runtime.
    Unknown = 0,
    // Can only be evaluated fully at runtime. May depend on runtime-derived
    // values from the HAL, custom modules, or parameters.
    Runtime,
    // Can be entirely evaluated at compile-time.
    Compiler,
  };

  InitializationAnalysis(
      Operation *rootOp, SymbolTable &symbolTable,
      const IREE::Util::ConstExprAnalysis &constExprAnalysis) {
    run(rootOp, symbolTable, constExprAnalysis);
  }

  // Returns the calculated availability of an initializer indicating when it is
  // able to be evaluated.
  Availability
  getInitializerAvailability(IREE::Util::InitializerOpInterface initializerOp) {
    auto it = initializerAvailability.find(initializerOp);
    if (it == initializerAvailability.end())
      return Availability::Unknown;
    return it->second;
  }

private:
  void run(Operation *rootOp, SymbolTable &symbolTable,
           const IREE::Util::ConstExprAnalysis &constExprAnalysis) {
    unsigned nextOpOrdinal = 0;
    for (auto &region : rootOp->getRegions()) {
      for (auto &op : region.getOps()) {
        if (auto globalOp = dyn_cast<IREE::Util::GlobalOpInterface>(op)) {
          // Globals with initial values are initialized in order with where
          // they are in the module.
          auto &timeline = globalTimelines[globalOp.getGlobalName().getValue()];
          assert(timeline.empty() && "out-of-order global store");
          timeline.push_back(
              std::make_pair(nextOpOrdinal++, Availability::Compiler));
        } else if (auto initializerOp =
                       dyn_cast<IREE::Util::InitializerOpInterface>(op)) {
          // Initializer availability depends on all dependent initialized
          // values.
          initializerAvailability[initializerOp] =
              calculateInitializerAvailability(
                  initializerOp, symbolTable, constExprAnalysis, nextOpOrdinal);
        }
      }
    }
  }

  // Returns the availability of |globalName| by the time |opOrdinal| is
  // executed. Note that some globals may be initialized multiple times (yuck,
  // but valid).
  Availability queryGlobalInitializationStatus(StringRef globalName,
                                               unsigned opOrdinal) {
    auto &timeline = globalTimelines[globalName];
    if (timeline.empty())
      return Availability::Unknown;
    for (auto &timepoint : timeline) {
      if (timepoint.first > opOrdinal)
        return timepoint.second;
    }
    return timeline.back().second;
  }

  // Returns true if the given |initializerOp| is a constant expression that is
  // able to be evaluated by this pass.
  Availability calculateInitializerAvailability(
      IREE::Util::InitializerOpInterface initializerOp,
      SymbolTable &symbolTable,
      const IREE::Util::ConstExprAnalysis &constExprAnalysis,
      unsigned &nextOpOrdinal) {
    SmallVector<std::pair<IREE::Util::GlobalStoreOpInterface, unsigned>>
        globalStoreOps;

    // Assume compile-time availability unless we see anything that may prevent
    // it. As we analyze the initializer we may "lower" the availability from
    // the most available (compile-time) to least available (run-time/unknown).
    auto availability = Availability::Compiler;
    auto lowerAvailability = [&](Availability newAvailability,
                                 StringRef reason) {
      auto previousAvailability = availability;
      availability = static_cast<Availability>(
          std::min(static_cast<unsigned>(availability),
                   static_cast<unsigned>(newAvailability)));
      if (previousAvailability != availability)
        emitWarning(initializerOp.getLoc()) << reason;
    };

    if (initializerOp->getRegions().size() != 1 ||
        !initializerOp->getRegion(0).hasOneBlock()) {
      // Skip if multiple blocks. It would be possible to support these in
      // theory but unclear if worth it in practice given the predominance of
      // SCF at the levels we run things. What we'd require is adding a single
      // exit block that stored to the globals unconditionally.
      lowerAvailability(Availability::Unknown,
                        "skipping consteval initializer: initializers with >1 "
                        "block not yet supported");
    } else if (isParameterized(symbolTable, initializerOp)) {
      // We don't allow anything with parameters today. We could handle these by
      // passing in the parameter file for use but would likely also want to
      // bind a writeable parameter file to produce into.
      lowerAvailability(Availability::Runtime,
                        "skipping consteval initializer: uses parameters or "
                        "other runtime-dependent values");
    }

    // Today we require that all values are constant expressions. We could slice
    // out just the ones that are.
    for (auto &op : initializerOp.getInitializerRegion().getOps()) {
      if (op.hasTrait<OpTrait::ConstantLike>() ||
          isa<IREE::Util::ReturnOp>(op)) {
        continue;
      } else if (isa<RegionBranchOpInterface>(op)) {
        // Control flow currently isn't evaluated properly; we'd need much
        // better analysis for things like conditional stores to globals. We
        // could make this more permissive for cases where the globals are
        // stored unconditionally/once but still allow control flow in other
        // places.
        lowerAvailability(
            Availability::Unknown,
            "skipping consteval initializer: has control flow ops");
      } else if (isa<CallOpInterface>(op)) {
        // Calls aren't currently analyzed - we need to rewrite this to use DFX
        // and walk the call graph to do that.
        lowerAvailability(Availability::Unknown,
                          "skipping consteval initializer: has call");
      } else if (isa<IREE::Util::GlobalLoadIndirectOpInterface>(op) ||
                 isa<IREE::Util::GlobalStoreIndirectOpInterface>(op)) {
        // Pessimistic case as we need analysis to know if the global
        // being loaded may potentially be a parameter.
        lowerAvailability(
            Availability::Unknown,
            "skipping consteval initializer: has indirect global accesses");
      } else if (auto loadOp =
                     dyn_cast<IREE::Util::GlobalLoadOpInterface>(op)) {
        // Globals must be initialized prior to this initializer and if they are
        // initialized at runtime it means this initializer must be too.
        auto globalStatus = queryGlobalInitializationStatus(
            loadOp.getGlobalName(), nextOpOrdinal++);
        if (globalStatus != Availability::Compiler) {
          lowerAvailability(globalStatus, "skipping consteval initializer: has "
                                          "runtime-dependent global load");
        }
      } else if (auto storeOp =
                     dyn_cast<IREE::Util::GlobalStoreOpInterface>(op)) {
        // Only allow stores to immutable globals (ones we initialize).
        auto globalOp = symbolTable.lookup<IREE::Util::GlobalOpInterface>(
            storeOp.getGlobalAttr().getAttr());
        if (!globalOp || globalOp.isGlobalMutable()) {
          lowerAvailability(
              Availability::Runtime,
              "skipping consteval initializer: has mutable global store");
        }
        globalStoreOps.push_back(std::make_pair(storeOp, nextOpOrdinal++));
      } else if (!constExprAnalysis.isConstExprOperation(&op)) {
        lowerAvailability(
            Availability::Runtime,
            "skipping consteval initializer: has non-const-expr values");
      }
    }

    // Record global availability produced by this initializer.
    for (auto [storeOp, opOrdinal] : globalStoreOps) {
      auto &timeline = globalTimelines[storeOp.getGlobalName()];
      timeline.push_back(std::make_pair(opOrdinal, availability));
    }
    return availability;
  }

  // An initialization-ordered sequence denoting changes in availability.
  // Example:
  //   * [-1, Compiler]: initialized with a constant primitive at startup
  //   * [2, Runtime]: reinitialized with a value computed at runtime
  //   * [4, Compiler]: reinitialized with a value available at compile time
  //   * [8, Unknown]: reinitialized with a value that failed analysis
  // The timeline can be queried by walking in order looking for any ordinal
  // under the requested point. A query at 3 would return Runtime as it is after
  // the first initialization but prior to the subsequent reinitializations.
  using AvailabilityTimeline = SmallVector<std::pair<unsigned, Availability>>;
  DenseMap<StringRef, AvailabilityTimeline> globalTimelines;
  DenseMap<Operation *, Availability> initializerAvailability;
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
  ArgumentBinding(IREE::Util::GlobalOpInterface globalOp)
      : type(Type::GlobalOp), globalOp(globalOp) {}

  Type getType() { return type; }

  ElementsAttr getElementsAttr() {
    assert(type == Type::ElementsAttr);
    return elementsAttr;
  }

  IREE::Util::GlobalOpInterface getGlobalOp() {
    assert(type == Type::GlobalOp);
    return globalOp;
  }

private:
  Type type;
  ElementsAttr elementsAttr;
  IREE::Util::GlobalOpInterface globalOp;
};

// How to bind results to the original program.
class ResultBinding {
public:
  enum class Type {
    // Set the result on the global op.
    GlobalOp,
  };

  ResultBinding(IREE::Util::GlobalOpInterface globalOp)
      : type(Type::GlobalOp), globalOp(globalOp) {}

  Type getType() { return type; }

  IREE::Util::GlobalOpInterface getGlobalOp() {
    assert(type == Type::GlobalOp);
    return globalOp;
  }

private:
  Type type;
  ElementsAttr elementsAttr;
  IREE::Util::GlobalOpInterface globalOp;
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
                 const SupportedFeatures &supportedFeatures,
                 const IREE::Util::ConstExprAnalysis &constExprAnalysis)
      : targetModuleOp(createInnerModule(sourceModuleOp)),
        sourceSymbolTable(sourceModuleOp), targetSymbolTable(targetModuleOp),
        supportedFeatures(supportedFeatures),
        constExprAnalysis(constExprAnalysis),
        initializationAnalysis(sourceModuleOp, sourceSymbolTable,
                               constExprAnalysis) {}

  llvm::SmallVector<JitFunctionDesc> &getJitFunctions() { return jitFunctions; }
  ModuleOp getTargetModule() { return targetModuleOp; }

  LogicalResult importInitializer(IREE::Util::InitializerOp initializerOp) {
    //  We convert each initializer into a public FuncOp by converting each:
    //    - Tensor constant into an argument
    //    - util.global.load into an argument
    //    - util.global.store into a result
    //  It is considered an eval'able initializer if it contains stores
    //  into immutable global(s). In the future, we will also want to
    //  condition this on an attribute so as to not try to statically
    //  compile dynamic initializers.
    auto availability =
        initializationAnalysis.getInitializerAvailability(initializerOp);
    if (availability != InitializationAnalysis::Availability::Compiler)
      return failure();

    OpBuilder moduleBuilder = OpBuilder::atBlockEnd(targetModuleOp.getBody());
    auto funcOp = moduleBuilder.create<IREE::Util::FuncOp>(
        initializerOp.getLoc(), "jit_eval",
        moduleBuilder.getFunctionType({}, {}));
    targetSymbolTable.insert(funcOp);
    IRMapping unusedMapping;
    initializerOp.getBody().cloneInto(&funcOp.getBody(), unusedMapping);
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

  LogicalResult transformToJitFunction(IREE::Util::FuncOp funcOp) {
    JitFunctionDesc desc(funcOp.getLoc(), funcOp.getName().str());
    llvm::SmallVector<Type> argumentTypes;
    llvm::SmallVector<Type> returnTypes;
    llvm::SmallVector<Value> returns;
    llvm::SmallVector<Operation *> eraseOps;

    Block *entryBlock = &funcOp.getBody().front();

    // Find immutable loads.
    for (auto loadOp : funcOp.getOps<IREE::Util::GlobalLoadOpInterface>()) {
      auto globalOp = llvm::dyn_cast_or_null<IREE::Util::GlobalOpInterface>(
          sourceSymbolTable.lookup(loadOp.getGlobalAttr().getAttr()));
      if (!globalOp || globalOp.isGlobalMutable()) {
        emitWarning(loadOp.getLoc()) << "skipping consteval initializer: load "
                                        "from mutable globals not supported";
        return failure();
      }
      Type t = loadOp.getLoadedGlobalValue().getType();
      if (!supportedFeatures.isSupportedAbiType(t)) {
        emitWarning(funcOp.getLoc())
            << "skipping consteval initializer: unsupported type for current "
               "jit configuration: "
            << t;
        return failure();
      }
      argumentTypes.push_back(t);
      BlockArgument entryArg = entryBlock->addArgument(t, loadOp.getLoc());
      loadOp.getLoadedGlobalValue().replaceAllUsesWith(entryArg);
      eraseOps.push_back(loadOp);
      desc.argumentBindings.emplace_back(globalOp);
    }

    // And loose tensor constants.
    for (auto constantOp : funcOp.getOps<arith::ConstantOp>()) {
      auto tensorType = dyn_cast<TensorType>(constantOp.getResult().getType());
      auto elementsAttr = dyn_cast<ElementsAttr>(constantOp.getValue());
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
    for (auto storeOp : funcOp.getOps<IREE::Util::GlobalStoreOpInterface>()) {
      auto globalOp = llvm::dyn_cast_or_null<IREE::Util::GlobalOpInterface>(
          sourceSymbolTable.lookup(storeOp.getGlobalAttr().getAttr()));
      assert(globalOp && "should have been checked in isConstExpr");

      Type t = storeOp.getStoredGlobalValue().getType();
      if (!supportedFeatures.isSupportedAbiType(t)) {
        emitWarning(funcOp.getLoc())
            << "skipping consteval initializer: unsupported type for current "
               "jit configuration: "
            << t;
        return failure();
      }

      returns.push_back(storeOp.getStoredGlobalValue());
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
    termBuilder.create<IREE::Util::ReturnOp>(funcOp.getLoc(), returns);
    funcOp.setType(termBuilder.getFunctionType(argumentTypes, returnTypes));

    jitFunctions.push_back(std::move(desc));
    return success();
  }

  ModuleOp targetModuleOp;
  SymbolTable sourceSymbolTable;
  SymbolTable targetSymbolTable;
  llvm::SmallVector<JitFunctionDesc> jitFunctions;
  const SupportedFeatures &supportedFeatures;
  const IREE::Util::ConstExprAnalysis &constExprAnalysis;
  InitializationAnalysis initializationAnalysis;
};

struct JitGlobalsPass : public JitGlobalsBase<JitGlobalsPass> {
  JitGlobalsPass(const JitGlobalsOptions &options)
      : compileOptions(std::make_shared<CompileOptions>()),
        compilePipeline("builtin.module") {
    targetRegistry = options.targetRegistry;

    // Detect backend.
    requestedTargetDevice = resolveTargetDevice(*targetRegistry.value);
    hasRequestedTargetDevice =
        targetRegistry->getTargetDevice(requestedTargetDevice) != nullptr;
    compileOptions->executableOptions.targets.push_back(requestedTargetDevice);
    compileOptions->targetOptions.f32Extension = true;
    compileOptions->targetOptions.f64Extension = true;
    compileOptions->targetOptions.truncateUnsupportedFloats = false;
    if (requestedTargetDevice == "vmvx" || !hasRequestedTargetDevice) {
      targetDevice = targetRegistry->getTargetDevice("vmvx");
    } else {
      targetDevice = targetRegistry->getTargetDevice(requestedTargetDevice);
    }

    // Disable constant evaluation for our Jit compilation pipeline.
    // It would make no sense to recursively do constant evaluation, and since
    // we omit the necessary hooks, it is unsupported anyway.
    compileOptions->globalOptimizationOptions.constExprHoisting = false;
    compileOptions->globalOptimizationOptions.constEval = false;

    buildIREEVMTransformPassPipeline(
        *targetRegistry.value, compileOptions->bindingOptions,
        compileOptions->inputOptions, compileOptions->preprocessingOptions,
        compileOptions->globalOptimizationOptions,
        compileOptions->schedulingOptions, compileOptions->executableOptions,
        compileOptions->targetOptions, compileOptions->hooks, compilePipeline);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    compilePipeline.getDependentDialects(registry);
  }

  static std::string
  resolveTargetDevice(const IREE::HAL::TargetRegistry &targetRegistry) {
    if (clJitTargetDevice.empty()) {
      // Default - choose something we have.
      // First llvm-cpu then vmvx.
      if (targetRegistry.getTargetDevice("llvm-cpu")) {
        return std::string("llvm-cpu");
      } else {
        return std::string("vmvx");
      }
    }

    return clJitTargetDevice;
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
    if (requestedTargetDevice != "vmvx" && hasRequestedTargetDevice) {
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
        llvm::dbgs() << "::: Invoking " << jitFunction.name << "\n";
      }

      FunctionCall call(binary, jitFunction.argumentBindings.size(),
                        jitFunction.resultBindings.size());
      if (failed(call.initialize(jitFunction.loc)))
        return failure();

      // Convert arguments.
      for (ArgumentBinding &arg : jitFunction.argumentBindings) {
        switch (arg.getType()) {
        case ArgumentBinding::Type::ElementsAttr: {
          if (failed(call.addArgument(jitFunction.loc, arg.getElementsAttr())))
            return failure();
          break;
        }
        case ArgumentBinding::Type::GlobalOp: {
          auto globalValue = arg.getGlobalOp().getGlobalInitialValue();
          if (!globalValue) {
            return emitError(jitFunction.loc)
                   << "internal error: jit global source initialization order "
                      "invalid: global "
                   << arg.getGlobalOp().getGlobalName() << " has no value";
          }
          if (failed(call.addArgument(arg.getGlobalOp().getLoc(), globalValue)))
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
                  resultBinding.getGlobalOp().getGlobalType(), attr)))
            return failure();
          resultBinding.getGlobalOp().setGlobalInitialValue(attr);
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
    if (!hasRequestedTargetDevice) {
      emitWarning(UnknownLoc::get(&getContext()))
          << "consteval jit requested with " << requestedTargetDevice
          << " backend, but it is not available. Falling back to vmvx";
    }
    if (!targetDevice) {
      emitError(UnknownLoc::get(&getContext()))
          << "consteval jit could not find a usable backend (requested '"
          << requestedTargetDevice << "')";
      signalPassFailure();
      return;
    }

    llvm::SmallVector<IREE::Util::InitializerOp> initializerOps;
    llvm::SmallVector<IREE::Util::InitializerOp> deadInitOps;
    for (auto childOp : outerModule.getOps<IREE::Util::InitializerOp>()) {
      initializerOps.push_back(childOp);
    }

    // Build the program.
    ProgramBuilder programBuilder(outerModule, supportedFeatures,
                                  getAnalysis<IREE::Util::ConstExprAnalysis>());

    // Set the target.
    std::optional<IREE::HAL::DeviceTargetAttr> targetAttr =
        targetDevice->getHostDeviceTarget(&getContext(), *targetRegistry.value);
    {
      if (!targetAttr) {
        emitError(UnknownLoc::get(&getContext()))
            << "consteval requested backend " << requestedTargetDevice
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
    for (auto initializerOp : initializerOps) {
      if (succeeded(programBuilder.importInitializer(initializerOp))) {
        deadInitOps.push_back(initializerOp);
      } else if (debugEnabled) {
        llvm::dbgs() << "::: Rejected consteval initializer:\n"
                     << initializerOp << "\n";
      }
    }
    if (programBuilder.getJitFunctions().empty()) {
      programBuilder.getTargetModule()->erase();
      return;
    }

    std::optional<llvm::Timer> compileTimer;
    if (debugEnabled) {
      llvm::dbgs() << "::: COMPILING JIT (" << requestedTargetDevice
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

  std::shared_ptr<CompileOptions> compileOptions;
  OpPassManager compilePipeline;
  std::string requestedTargetDevice;
  std::shared_ptr<IREE::HAL::TargetDevice> targetDevice;
  bool hasRequestedTargetDevice;
  bool debugEnabled = isDebugEnabled();
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createJitGlobalsPass(const JitGlobalsOptions &options) {
  return std::make_unique<JitGlobalsPass>(options);
}

std::unique_ptr<OperationPass<ModuleOp>> createJitGlobalsPass() {
  return std::make_unique<JitGlobalsPass>(JitGlobalsOptions{});
}

} // namespace mlir::iree_compiler::ConstEval
