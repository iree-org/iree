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

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMAOTTarget.h"

#include <cstdlib>

#include "iree/compiler/Conversion/CodegenUtils/GetNumWorkgroups.h"
#include "iree/compiler/Conversion/Common/Attributes.h"
#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMIRPasses.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LinkerTool.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/dylib_executable_def_builder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {

llvm::Optional<FileLineColLoc> findFirstFileLoc(Location baseLoc) {
  if (auto loc = baseLoc.dyn_cast<FusedLoc>()) {
    for (auto &childLoc : loc.getLocations()) {
      auto childResult = findFirstFileLoc(childLoc);
      if (childResult) return childResult;
    }
  } else if (auto loc = baseLoc.dyn_cast<FileLineColLoc>()) {
    return loc;
  }
  return llvm::None;
}

std::string guessModuleName(mlir::ModuleOp moduleOp) {
  std::string moduleName =
      moduleOp.getName().hasValue() ? moduleOp.getName().getValue().str() : "";
  if (!moduleName.empty()) return moduleName;
  auto loc = findFirstFileLoc(moduleOp.getLoc());
  if (loc.hasValue()) {
    return llvm::sys::path::stem(loc.getValue().getFilename()).str();
  } else {
    return "llvm_module";
  }
}

}  // namespace

class LLVMAOTTargetBackend final : public TargetBackend {
 public:
  explicit LLVMAOTTargetBackend(LLVMTargetOptions options)
      : options_(std::move(options)) {}

  // NOTE: we could vary these based on the options, such as by arch/etc.
  std::string name() const override { return "llvm_aot"; }
  std::string filter_pattern() const override { return "dylib*"; }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    buildLLVMTransformPassPipeline(passManager);
  }

  LogicalResult linkExecutables(mlir::ModuleOp moduleOp) override {
    OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());

    auto sourceExecutableOps =
        llvm::to_vector<8>(moduleOp.getOps<IREE::HAL::ExecutableOp>());
    if (sourceExecutableOps.size() <= 1) return success();

    // Guess a module name, if needed, to make the output files readable.
    auto moduleName = guessModuleName(moduleOp);

    // Create our new "linked" hal.executable.
    std::string linkedExecutableName =
        llvm::formatv("{0}_linked_{1}", moduleName, name());
    auto linkedExecutableOp = builder.create<IREE::HAL::ExecutableOp>(
        moduleOp.getLoc(), linkedExecutableName);
    linkedExecutableOp.setVisibility(
        sourceExecutableOps.front().getVisibility());

    // Add our hal.executable.target with an empty module.
    builder.setInsertionPointToStart(linkedExecutableOp.getBody());
    auto linkedTargetOp = builder.create<IREE::HAL::ExecutableTargetOp>(
        moduleOp.getLoc(), name(), filter_pattern());
    builder.setInsertionPoint(&linkedTargetOp.getBlock().back());
    builder.create<ModuleOp>(moduleOp.getLoc());

    // Try linking together all executables in moduleOp.
    return linkExecutablesInto(
        moduleOp, sourceExecutableOps, linkedExecutableOp, linkedTargetOp,
        [](mlir::ModuleOp moduleOp) { return moduleOp; }, builder);
  }

  LogicalResult recordDispatch(Location loc, DispatchState dispatchState,
                               DeviceSwitchRewriter &switchRewriter) override {
    // TODO(#4140): remove this legacy path when linalg-on-tensors is used.
    // In the linalg-on-tensors world where we are performing the tiling logic
    // in the flow dialect we don't even really need the ability to override
    // dispatch recording at all - just a way to allow targets to map workgroup
    // counts from the N-dimensional flow workgroup counts to the 3D hal counts.
    if (dispatchState.workgroupCount.size() == 3) {
      return TargetBackend::recordDispatch(loc, dispatchState, switchRewriter);
    }

    IREE::HAL::ExecutableOp executableOp = dispatchState.executableOp;
    ModuleOp llvmIRModuleOp;
    for (auto executableTargetOp :
         executableOp.getBlock().getOps<IREE::HAL::ExecutableTargetOp>()) {
      if (matchPattern(executableTargetOp.target_backend_filter(),
                       filter_pattern())) {
        ModuleOp innerModuleOp = executableTargetOp.getInnerModule();
        llvmIRModuleOp = innerModuleOp;
        break;
      }
    }
    if (!llvmIRModuleOp)
      return executableOp.emitError("unable to find executable llvmIR module");

    SmallVector<LLVM::LLVMFuncOp, 2> entryPointFns;
    for (LLVM::LLVMFuncOp funcOp : llvmIRModuleOp.getOps<LLVM::LLVMFuncOp>()) {
      if (funcOp.isPublic()) {
        entryPointFns.push_back(funcOp);
      }
    }

    auto *region = switchRewriter.addConditionRegion(
        IREE::HAL::DeviceMatchIDAttr::get(filter_pattern(), loc.getContext()),
        {
            dispatchState.workgroupCount[0],
            dispatchState.commandBuffer,
        });
    auto &entryBlock = region->front();
    ConversionPatternRewriter &rewriter = switchRewriter.getRewriter();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(&entryBlock);

    auto commandBuffer = entryBlock.getArgument(1);
    for (auto it : llvm::enumerate(entryPointFns)) {
      LLVM::LLVMFuncOp funcOp = it.value();
      FlatSymbolRefAttr numWorkgroupsFnAttr =
          funcOp->getAttrOfType<FlatSymbolRefAttr>(
              getNumWorkgroupsFnAttrName());
      if (!numWorkgroupsFnAttr) {
        auto constantOne = rewriter.createOrFold<mlir::ConstantIndexOp>(loc, 1);
        rewriter.create<IREE::HAL::CommandBufferDispatchSymbolOp>(
            loc, commandBuffer, dispatchState.entryPointOp, constantOne,
            constantOne, constantOne);
        rewriter.create<IREE::HAL::ReturnOp>(loc);
        return success();
      }
      std::array<Value, 3> workgroupCount = {nullptr, nullptr, nullptr};
      FuncOp numWorkgroupsFn = dyn_cast<FuncOp>(SymbolTable::lookupSymbolIn(
          funcOp->getParentOfType<ModuleOp>(), numWorkgroupsFnAttr));
      if (!numWorkgroupsFn) {
        return funcOp.emitError("unable to find function ")
               << numWorkgroupsFnAttr
               << " that computes the number of workgroups to use";
      }
      workgroupCount =
          iree_compiler::calculateWorkgroupCountFromNumWorkgroupsFn(
              loc, numWorkgroupsFn, dispatchState.interfaceOp,
              dispatchState.operands, dispatchState.results, rewriter);

      if (llvm::any_of(workgroupCount,
                       [](Value v) -> bool { return v == nullptr; })) {
        auto constantOne = rewriter.createOrFold<mlir::ConstantIndexOp>(loc, 1);
        rewriter.create<IREE::HAL::CommandBufferDispatchSymbolOp>(
            loc, commandBuffer, dispatchState.entryPointOp, constantOne,
            constantOne, constantOne);
      } else {
        rewriter.create<IREE::HAL::CommandBufferDispatchSymbolOp>(
            loc, commandBuffer, dispatchState.entryPointOp, workgroupCount[0],
            workgroupCount[1], workgroupCount[2]);
      }
    }
    rewriter.create<IREE::HAL::ReturnOp>(loc);
    return success();
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableTargetOp targetOp,
                                    OpBuilder &executableBuilder) override {
    // Perform the translation in a separate context to avoid any
    // multi-threading issues.
    llvm::LLVMContext context;

    // We name our files after the executable name so that they are easy to
    // track both during compilation (logs/artifacts/etc), as outputs (final
    // intermediate code/binary files), and at runtime (loaded
    // libraries/symbols/etc).
    auto libraryName =
        targetOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    // TODO(#3737): don't add functions we don't want to serialize to the
    // module. Right now workgroup count calculation functions end up in here
    // as std.func ops and not just the llvm.func ops we expect.
    auto illegalFuncOps =
        llvm::to_vector<4>(targetOp.getInnerModule().getOps<FuncOp>());
    for (auto funcOp : illegalFuncOps) {
      funcOp.erase();
    }

    llvm::Triple targetTriple(options_.targetTriple);
    targetOp.getInnerModule()->setAttr(
        LLVM::LLVMDialect::getTargetTripleAttrName(),
        executableBuilder.getStringAttr(targetTriple.str()));

    // At this moment we are leaving MLIR LLVM dialect land translating module
    // into target independent LLVMIR.
    auto llvmModule = mlir::translateModuleToLLVMIR(targetOp.getInnerModule(),
                                                    context, libraryName);
    if (!llvmModule) {
      return targetOp.emitError() << "failed to translate the MLIR LLVM "
                                     "dialect to the native llvm::Module";
    }

    // Try to grab a linker tool based on the options (and target environment).
    auto linkerTool = LinkerTool::getForTarget(targetTriple, options_);
    if (!linkerTool) {
      return mlir::emitError(targetOp.getLoc())
             << "failed to find a target linker for the given target triple '"
             << options_.targetTriple << "'";
    }

    // Configure the module with any code generation options required later by
    // linking (such as initializer functions).
    auto entryPointNames = llvm::to_vector<8>(
        llvm::map_range(targetOp.getBlock().getOps<ExecutableEntryPointOp>(),
                        [&](auto op) { return op.getName(); }));
    if (failed(
            linkerTool->configureModule(llvmModule.get(), entryPointNames))) {
      return targetOp.emitError()
             << "failed to configure LLVM module for target linker";
    }

    // LLVM opt passes that perform code generation optimizations/transformation
    // similar to what a frontend would do before passing to linking.
    auto targetMachine = createTargetMachine(options_);
    if (!targetMachine) {
      return mlir::emitError(targetOp.getLoc())
             << "failed to create target machine for target triple '"
             << options_.targetTriple << "'";
    }
    llvmModule->setDataLayout(targetMachine->createDataLayout());
    llvmModule->setTargetTriple(targetMachine->getTargetTriple().str());
    if (failed(
            runLLVMIRPasses(options_, targetMachine.get(), llvmModule.get()))) {
      return targetOp.emitError()
             << "failed to run LLVM-IR opt passes for IREE::HAL::ExecutableOp "
                "targeting '"
             << options_.targetTriple << "'";
    }

    // Emit object files.
    SmallVector<Artifact, 4> objectFiles;
    {
      // NOTE: today we just use a single object file, however if we wanted to
      // scale code generation and linking we'd want to generate one per
      // function (or something like that).
      std::string objectData;
      if (failed(runEmitObjFilePasses(targetMachine.get(), llvmModule.get(),
                                      &objectData))) {
        return targetOp.emitError()
               << "failed to compile LLVM-IR module to an object file";
      }
      auto objectFile = Artifact::createTemporary(libraryName, "obj");
      auto &os = objectFile.outputFile->os();
      os << objectData;
      os.flush();
      os.close();
      objectFiles.push_back(std::move(objectFile));
    }

    // Link the generated object files into a dylib.
    auto linkArtifactsOr =
        linkerTool->linkDynamicLibrary(libraryName, objectFiles);
    if (!linkArtifactsOr.hasValue()) {
      return mlir::emitError(targetOp.getLoc())
             << "failed to link executable and generate target dylib using "
                "linker toolchain "
             << linkerTool->getToolPath();
    }
    auto &linkArtifacts = linkArtifactsOr.getValue();
    if (options_.keepLinkerArtifacts) {
      mlir::emitRemark(targetOp.getLoc())
          << "Linker artifacts for " << targetOp.getName() << " preserved:\n"
          << "    " << linkArtifacts.libraryFile.path;
      linkArtifacts.keepAllFiles();
    }

    // Embed debug symbols at the end of the flatbuffer by adding first in the
    // bottoms-up builder.
    FlatbufferBuilder builder;
    flatbuffers_uint8_vec_ref_t debugDatabaseRef = 0;
    flatbuffers_string_ref_t debugDatabaseFilenameRef = 0;
    if (options_.debugSymbols && linkArtifacts.debugFile.outputFile) {
      debugDatabaseRef = builder.streamUint8Vec([&](raw_ostream &stream) {
        return linkArtifacts.debugFile.readInto(stream);
      });
      debugDatabaseFilenameRef = builder.createString(
          llvm::sys::path::filename(linkArtifacts.debugFile.path));
    }

    // Embed entire dynamic library output.
    flatbuffers_uint8_vec_ref_t libraryEmbeddedRef =
        builder.streamUint8Vec([&](raw_ostream &stream) {
          return linkArtifacts.libraryFile.readInto(stream);
        });
    if (!libraryEmbeddedRef) {
      return targetOp.emitError() << "failed to read back dylib temp file at "
                                  << linkArtifacts.libraryFile.path;
    }

    // Entry point names up from.
    // TODO(#3580): these won't be needed in the executable_library world.
    auto entryPointsRef = builder.createStringVec(llvm::map_range(
        targetOp.getBlock().getOps<ExecutableEntryPointOp>(),
        [&](ExecutableEntryPointOp op) { return op.getName(); }));

    iree_DyLibExecutableDef_start_as_root(builder);
    iree_DyLibExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_DyLibExecutableDef_library_embedded_add(builder, libraryEmbeddedRef);
    iree_DyLibExecutableDef_debug_database_filename_add(
        builder, debugDatabaseFilenameRef);
    iree_DyLibExecutableDef_debug_database_embedded_add(builder,
                                                        debugDatabaseRef);
    iree_DyLibExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(), targetOp.sym_name(),
        static_cast<uint32_t>(IREE::HAL::ExecutableFormat::DyLib),
        builder.getBufferAttr(executableBuilder.getContext()));
    return success();
  }

 private:
  LLVMTargetOptions options_;
};

void registerLLVMAOTTargetBackends(
    std::function<LLVMTargetOptions()> queryOptions) {
  getLLVMTargetOptionsFromFlags();
  static TargetBackendRegistration registration("dylib-llvm-aot", [=]() {
#define INIT_LLVM_TARGET(TargetName)        \
  LLVMInitialize##TargetName##Target();     \
  LLVMInitialize##TargetName##TargetMC();   \
  LLVMInitialize##TargetName##TargetInfo(); \
  LLVMInitialize##TargetName##AsmPrinter(); \
  LLVMInitialize##TargetName##AsmParser();
    INIT_LLVM_TARGET(X86)
    INIT_LLVM_TARGET(ARM)
    INIT_LLVM_TARGET(AArch64)
    INIT_LLVM_TARGET(RISCV)
    return std::make_unique<LLVMAOTTargetBackend>(queryOptions());
  });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
