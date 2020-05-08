// Copyright 2020 Google LLC
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

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMTarget.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMIRPasses.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Translation/CodegenPasses/Passes.h"
#include "iree/schemas/llvmir_executable_def_generated.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// TODO(ataei): This is written as a stub in LLVM IR. It would be easier to have
// this using MLIR and lower it to LLVM like the dispatch function
// implementation is.
static void createInvocationFunc(const std::string& name,
                                 llvm::Module* module) {
  auto& ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  auto var_func = module->getFunction(name);

  auto new_type = llvm::FunctionType::get(
      builder.getVoidTy(), builder.getInt8PtrTy()->getPointerTo(),
      /*isVarArg=*/false);

  auto new_name = "invoke_" + name;
  auto func_cst = module->getOrInsertFunction(new_name, new_type);
  llvm::Function* interface_func =
      llvm::cast<llvm::Function>(func_cst.getCallee());

  auto bb = llvm::BasicBlock::Create(ctx);
  bb->insertInto(interface_func);
  builder.SetInsertPoint(bb);
  llvm::Value* argList = interface_func->arg_begin();
  llvm::SmallVector<llvm::Value*, 8> args;
  args.reserve(llvm::size(var_func->args()));
  for (auto& indexedArg : llvm::enumerate(var_func->args())) {
    llvm::Value* arg_index = llvm::Constant::getIntegerValue(
        builder.getInt64Ty(), llvm::APInt(64, indexedArg.index()));
    llvm::Value* arg_ptr_ptr = builder.CreateGEP(argList, arg_index);
    llvm::Value* arg_ptr = builder.CreateLoad(arg_ptr_ptr);
    arg_ptr = builder.CreateBitCast(
        arg_ptr, indexedArg.value().getType()->getPointerTo());
    llvm::Value* arg = builder.CreateLoad(arg_ptr);
    args.push_back(arg);
  }
  builder.CreateCall(var_func, args);
  builder.CreateRetVoid();
}

namespace {

/// Returns true if the given function contains interface related operations
/// that are used by other ops.
bool containsUsedInterfaceOp(FuncOp funcOp) {
  for (Block& block : funcOp.getBlocks()) {
    for (Operation& op : block) {
      if (!op.getUses().empty() &&
          (isa<IREE::PlaceholderOp>(op) ||
           isa<IREE::HAL::InterfaceLoadConstantOp>(op))) {
        return true;
      }
    }
  }
  return false;
}

/// Returns true if `aOp` has a desciptor (set, binding) pair smaller than
/// `bOp`. Note that this ignores the offset.
bool operator<(InterfaceBindingOp aOp, InterfaceBindingOp bOp) {
  if (aOp.set().getZExtValue() == bOp.set().getZExtValue())
    return aOp.binding().getZExtValue() < bOp.binding().getZExtValue();
  return aOp.set().getZExtValue() < bOp.set().getZExtValue();
}

/// A pattern to process function interface. It replaces interface related ops
/// with function arguments to match LLVM's CodeGen's ABI contract.
///
/// IREE scheduler passes interface ABI information via hal.interface.* ops to
/// all backends. We create iree.placeholder ops to represent buffers behind
/// those hal.interface.* ops. However the LLVM CodeGen uses function parameters
/// and memref descriptors for ABI. So we need to bridge the gap somewhere.
///
/// This pass finds all interface buffers used in the function, sort them
/// according to the descriptor (set, binding) pair, and put unique ones as
/// function parameters in order.
/// Note: This should be kept consistent with LLVM's HAL backend.
struct ProcessFuncInterfacePattern : public OpConversionPattern<FuncOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      FuncOp funcOp, ArrayRef<Value> Operands,
      ConversionPatternRewriter& rewriter) const override {
    // Only process entry functions.
    if (SymbolTable::getSymbolVisibility(funcOp) !=
        SymbolTable::Visibility::Public)
      return failure();

    FunctionType fnType = funcOp.getType();
    if (fnType.getNumInputs() != 0)
      return rewriter.notifyMatchFailure(
          funcOp, "entry function should not have inputs");

    // Get interface buffers from all the blocks.
    // TODO: Also handle hal.interface.load.constant for dynamic shape.
    SmallVector<IREE::PlaceholderOp, 8> bufferOps;
    for (Block& block : funcOp.getBlocks()) {
      for (Operation& op : block)
        if (auto phOp = dyn_cast<IREE::PlaceholderOp>(op))
          bufferOps.push_back(phOp);
    }

    if (bufferOps.empty()) return failure();

    // A map from buffer ops to their corresponding interface binding ops.
    llvm::DenseMap<Operation*, IREE::HAL::InterfaceBindingOp> bufferBindingMap;
    for (auto bufferOp : bufferOps) {
      auto symbol = SymbolTable::lookupNearestSymbolFrom(
          bufferOp, bufferOp.getAttrOfType<SymbolRefAttr>("binding"));
      bufferBindingMap[bufferOp] = cast<IREE::HAL::InterfaceBindingOp>(symbol);
    }

    // Sort buffers according to their descriptor (set, binding) pair.
    llvm::sort(bufferOps, [&bufferBindingMap](IREE::PlaceholderOp aBuffer,
                                              IREE::PlaceholderOp bBuffer) {
      return bufferBindingMap[aBuffer] < bufferBindingMap[bBuffer];
    });

    // Create a function argument for each of the unique binding pointed by the
    // buffer ops.
    TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
    // A map from buffer ops to their corresponding function argument indices.
    llvm::DenseMap<Operation*, unsigned> bufferArgMap;
    // A map from binding ops to their corresponding function argument indices.
    llvm::DenseMap<Operation*, unsigned> bindingArgMap;
    unsigned argIndex = 0;
    for (auto bufferOp : bufferOps) {
      auto binding = bufferBindingMap[bufferOp];
      auto it = bindingArgMap.find(binding);
      if (it != bindingArgMap.end()) {
        bufferArgMap[bufferOp] = it->second;
      } else {
        bindingArgMap[binding] = argIndex;
        bufferArgMap[bufferOp] = argIndex;
        signatureConverter.addInputs(bufferOp.getType());
        ++argIndex;
      }
    }

    // Create the new function's signature.
    Location loc = funcOp.getLoc();
    auto newFuncOp = rewriter.create<FuncOp>(
        loc, funcOp.getName(),
        rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                                 llvm::None),
        ArrayRef<NamedAttribute>());
    newFuncOp.setAttr("llvm.emit_c_interface",
                      mlir::UnitAttr::get(funcOp.getContext()));

    // Move all ops in the old function's region to the new function.
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    rewriter.applySignatureConversion(&newFuncOp.getBody(), signatureConverter);

    // Replace all buffer ops' uses with the newly created function arguments
    // and erase them.
    for (auto bufferOp : bufferOps) {
      bufferOp.replaceAllUsesWith(
          newFuncOp.getArgument(bufferArgMap[bufferOp]));
      rewriter.eraseOp(bufferOp);
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct RemoveInterfaceOpPattern
    : public OpRewritePattern<IREE::HAL::InterfaceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::HAL::InterfaceOp interfaceOp,
                                PatternRewriter& rewriter) const override {
    rewriter.eraseOp(interfaceOp);
    return success();
  }
};

/// Converting from Linalg to LLVM needs to run on a module and since it
/// applies a full conversion, make a module with jst the impl function.
struct PrepareForLLVMLoweringPass
    : PassWrapper<PrepareForLLVMLoweringPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    MLIRContext& context = getContext();

    OwningRewritePatternList patterns;
    patterns.insert<ProcessFuncInterfacePattern>(&context);
    patterns.insert<RemoveInterfaceOpPattern>(&context);

    ConversionTarget target(context);
    // Convert the interface related ops away.
    target.addDynamicallyLegalOp<FuncOp>(
        [](FuncOp funcOp) { return !containsUsedInterfaceOp(funcOp); });
    target.addIllegalOp<IREE::PlaceholderOp>();
    target.addIllegalDialect<IREE::HAL::HALDialect>();
    // Allow the rest.
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

    if (failed(applyFullConversion(getOperation(), target, patterns, nullptr)))
      return signalPassFailure();
  }
};

}  // namespace

class LLVMIRTargetBackend final : public TargetBackend {
 public:
  LLVMIRTargetBackend(LLVMTargetOptions options)
      : options_(std::move(options)) {}

  // NOTE: we could vary this based on the options, such as by arch/etc.
  std::string name() const override { return "llvm*"; }

  // Adds a sequence of passess to a given pass manager that progressively lower
  // from HLO to LLVM throught linalg dialect.
  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpPassManager& passManager) override {
    passManager.addPass(createInlinerPass());

    // HLO -> Linalg on buffers.
    passManager.addPass(createDecomposeHLOClampPass());
    addHLOToLinalgOnBuffersPasses(passManager);

    // Linalg -> Loops
    passManager.addPass(createConvertLinalgToLoopsPass());
    passManager.addPass(createCanonicalizerPass());
    passManager.addPass(createCSEPass());

    // Loops -> STD
    passManager.addPass(createLowerToCFGPass());
    passManager.addPass(createCanonicalizerPass());
    passManager.addPass(createCSEPass());

    // (Linalg, STD) -> LLVM
    passManager.addPass(std::make_unique<PrepareForLLVMLoweringPass>());
    // OpPassManager& llvmPassManager = passManager.nest<ModuleOp>();
    passManager.addPass(createConvertLinalgToLLVMPass());
    passManager.addPass(createCanonicalizerPass());
    passManager.addPass(createCSEPass());
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder& executableBuilder) override {
    // LLVM is not thread safe and currently translation shares an LLVMContext.
    // Since we serialize executables from multiple threads we have to take a
    // global lock here.
    static llvm::sys::SmartMutex<true> mutex;
    llvm::sys::SmartScopedLock<true> lock(mutex);

    // At this moment we are leaving MLIR LLVM dialect land translating module
    // into target independent LLVMIR.
    auto llvmModule = mlir::translateModuleToLLVMIR(targetOp.getInnerModule());

    // Create invocation function an populate entry_points.
    iree::LLVMIRExecutableDefT llvmIrExecutableDef;
    auto executableOp = cast<IREE::HAL::ExecutableOp>(targetOp.getParentOp());
    auto entryPointOps =
        executableOp.getBlock().getOps<IREE::HAL::ExecutableEntryPointOp>();
    const bool addCInterface = true;
    for (auto entryPointOp : entryPointOps) {
      std::string funcName =
          addCInterface ? "_mlir_ciface_" + std::string(entryPointOp.sym_name())
                        : std::string(entryPointOp.sym_name());
      llvmIrExecutableDef.entry_points.push_back(funcName);
      createInvocationFunc(funcName, llvmModule.get());
    }

    // LLVMIR opt passes.
    auto targetMachine = createTargetMachine(options_);
    if (!targetMachine) {
      targetOp.emitError("Can't create target machine for target triple: " +
                         options_.targetTriple);
      return failure();
    }
    if (failed(runLLVMIRPasses(options_, std::move(targetMachine),
                               llvmModule.get()))) {
      return targetOp.emitError(
          "Can't build LLVMIR opt passes for ExecutableOp module");
    }

    // Serialize LLVM module.
    std::string bufferString;
    llvm::raw_string_ostream ostream(bufferString);
    llvmModule->print(ostream, nullptr);
    ostream.flush();

    // Creates executable bytes.
    llvmIrExecutableDef.llvmir_module = {bufferString.begin(),
                                         bufferString.end()};

    ::flatbuffers::FlatBufferBuilder fbb;
    auto executableOffset =
        iree::LLVMIRExecutableDef::Pack(fbb, &llvmIrExecutableDef);
    iree::FinishLLVMIRExecutableDefBuffer(fbb, executableOffset);
    std::vector<uint8_t> bytes;
    bytes.resize(fbb.GetSize());
    std::memcpy(bytes.data(), fbb.GetBufferPointer(), bytes.size());

    // Add the binary data to the target executable.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(),
        static_cast<uint32_t>(IREE::HAL::ExecutableFormat::LLVM),
        std::move(bytes));

    return success();
  }

 private:
  LLVMTargetOptions options_;
};

void registerLLVMTargetBackends(
    std::function<LLVMTargetOptions()> queryOptions) {
  getLLVMTargetOptionsFromFlags();
  static TargetBackendRegistration registration("llvm-ir", [=]() {
    // Initalize registered targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return std::make_unique<LLVMIRTargetBackend>(queryOptions());
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
