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

#include "iree/compiler/Dialect/HAL/Target/LLVM/AOT/LLVMAOTTarget.h"

#include <cstdlib>

#include "iree/compiler/Dialect/HAL/Target/LLVM/AOT/LinkerTool.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMBaseTarget.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMIRPasses.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/dylib_executable_def_builder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

class LLVMAOTTargetBackend final : public LLVMBaseTargetBackend {
 public:
  explicit LLVMAOTTargetBackend(LLVMTargetOptions options)
      : LLVMBaseTargetBackend(options) {}

  // NOTE: we could vary these based on the options, such as by arch/etc.
  std::string name() const override { return "llvm_aot"; }
  std::string filter_pattern() const override { return "dylib*"; }

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
        targetOp.getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    // TODO(#3737): don't add functions we don't want to serialize to the
    // module. Right now workgroup count calculation functions end up in here
    // as std.func ops and not just the llvm.func ops we expect.
    auto illegalFuncOps =
        llvm::to_vector<4>(targetOp.getInnerModule().getOps<FuncOp>());
    for (auto funcOp : illegalFuncOps) {
      funcOp.erase();
    }

    llvm::Triple targetTriple(options_.targetTriple);
    targetOp.getInnerModule().setAttr(
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
        targetOp.getLoc(),
        static_cast<uint32_t>(IREE::HAL::ExecutableFormat::DyLib),
        builder.getBufferAttr(executableBuilder.getContext()));
    return success();
  }
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
