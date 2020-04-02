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
#include "iree/compiler/Dialect/HAL/Target/LegacyUtil.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Translation/CodegenPasses/Passes.h"
#include "iree/compiler/Translation/CodegenUtils/CodegenUtils.h"
#include "iree/schemas/llvmir_executable_def_generated.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Mutex.h"
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

LLVMTargetOptions getLLVMTargetOptionsFromFlags() {
  LLVMTargetOptions targetOptions;
  return targetOptions;
}

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

/// Clones the dispatch function implementation into a module to be later passed
/// onto LLVM lowering.
struct DispatchFnImplRewritePattern : public OpRewritePattern<FuncOp> {
  using OpRewritePattern<FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(FuncOp funcOp,
                                PatternRewriter& rewriter) const override {
    if (isDispatchFuncImpl(funcOp)) {
      rewriter.updateRootInPlace(funcOp, [&funcOp]() {
        funcOp.setAttr("llvm.emit_c_interface",
                       mlir::UnitAttr::get(funcOp.getContext()));
        funcOp.setName(getDispatchFuncName(funcOp).getValue());
      });
    } else {
      rewriter.eraseOp(funcOp);
    }
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
    OwningRewritePatternList patterns;
    MLIRContext* context = &getContext();
    patterns.insert<DispatchFnImplRewritePattern>(context);
    patterns.insert<RemoveInterfaceOpPattern>(context);
    applyPatternsAndFoldGreedily(getOperation(), patterns);
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
    // Convert IREE's hal.interface accesses to memrefs.
    passManager.addPass(createHALInterfaceToMemrefPass());

    // HLO -> Linalg on buffers.
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
    return std::make_unique<LLVMIRTargetBackend>(queryOptions());
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
