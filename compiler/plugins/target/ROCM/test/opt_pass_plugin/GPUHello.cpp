#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace {

struct GpuHello final : llvm::PassInfoMixin<GpuHello> {
  llvm::PreservedAnalyses run(llvm::Module &module,
                              llvm::ModuleAnalysisManager &) {
    bool modifiedCodeGen = runOnModule(module);
    if (!modifiedCodeGen)
      return llvm::PreservedAnalyses::all();

    return llvm::PreservedAnalyses::all();
  }
  bool runOnModule(llvm::Module &module);
  // isRequired being set to true keeps this pass from being skipped
  // if it has the optnone LLVM attribute
  static bool isRequired() { return true; }
};

bool GpuHello::runOnModule(llvm::Module &module) {
  bool modifiedCodeGen = false;
  for (llvm::Function &function : module) {
    if (function.isIntrinsic())
      continue;

    llvm::StringRef functionName = function.getName();
    if (function.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL ||
        function.getCallingConv() == llvm::CallingConv::PTX_Kernel ||
        functionName.contains("kernel")) {

      for (llvm::BasicBlock &basicBlock : function) {
        for (llvm::Instruction &inst : basicBlock) {
          llvm::DILocation *debugLocation = inst.getDebugLoc();
          std::string sourceInfo;
          if (!debugLocation) {
            sourceInfo = function.getName().str();
          } else {
            // Using formatv
            sourceInfo = llvm::formatv("{0}\t{1}:{2}:{3}", function.getName(),
                                       debugLocation->getFilename(),
                                       debugLocation->getLine(),
                                       debugLocation->getColumn())
                             .str();
          }

          llvm::errs() << "Hello From First Instruction of GPU Kernel: "
                       << sourceInfo << "\n";
          return modifiedCodeGen;
        }
      }
    }
  }
  return false;
}

} // end anonymous namespace

llvm::PassPluginLibraryInfo getPassPluginInfo() {
  const auto callback = [](llvm::PassBuilder &pb) {
    pb.registerOptimizerLastEPCallback([&](llvm::ModulePassManager &mpm, auto) {
      mpm.addPass(GpuHello());
      return true;
    });
  };
  return {LLVM_PLUGIN_API_VERSION, "gpu-hello", LLVM_VERSION_STRING, callback};
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}
