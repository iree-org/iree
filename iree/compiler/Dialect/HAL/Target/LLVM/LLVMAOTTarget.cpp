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

#include "iree/compiler/Conversion/Common/Attributes.h"
#include "iree/compiler/Conversion/LinalgToLLVM/LLVMCodeGenOptions.h"
#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMIRPasses.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LibraryBuilder.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LinkerTool.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/dylib_executable_def_builder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

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
  std::string filter_pattern() const override {
    llvm::Triple targetTriple(options_.targetTriple);
    if (targetTriple.isWasm()) {
      return "wasm*";
    } else {
      return "dylib*";
    }
  }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    auto codeGenOptions = getLLVMCodegenOptionsFromClOptions();
    // Set target specific options.
    // TODO(ataei): This is temporary here, should move when target specific
    // overrides options grows.
    llvm::Triple triple(options_.targetTriple);
    if (triple.isWasm()) {
      // WebAssembly does not (yet) support FMA ops natively, so unfuse them.
      codeGenOptions.unfuseFMAOps = true;
    }
    buildLLVMTransformPassPipeline(passManager, codeGenOptions);
  }

  LogicalResult linkExecutables(mlir::ModuleOp moduleOp) override {
    mlir::registerLLVMDialectTranslation(*moduleOp->getContext());

    OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());

    auto sourceExecutableOps =
        llvm::to_vector<8>(moduleOp.getOps<IREE::HAL::ExecutableOp>());
    if (sourceExecutableOps.size() <= 1) return success();

    // Private symbols (i.e. llvm dialect private symbols) get deduped
    // incorrectly by the link executables pass even though they should be
    // treated as different symbols. For now just change the names of the
    // private symbols to avoid conflicts.
    unsigned moduleNumber = 0;
    for (auto sourceExecutableOp : enumerate(sourceExecutableOps)) {
      auto targetOps = llvm::to_vector<4>(
          sourceExecutableOp.value().getOps<IREE::HAL::ExecutableTargetOp>());
      for (auto targetOp : targetOps) {
        if (!matchPattern(targetOp.target_backend_filter(), filter_pattern())) {
          continue;
        }

        auto sourceModuleOp = targetOp.getInnerModule();
        for (auto globalOp : sourceModuleOp.getOps<LLVM::GlobalOp>()) {
          if (globalOp.linkage() != LLVM::Linkage::Private) {
            continue;
          }
          auto disambiguateName =
              llvm::formatv("{0}_{1}", globalOp.sym_name(), moduleNumber).str();
          SymbolTableCollection symbolTable;
          SymbolUserMap symbolUsers(symbolTable, sourceModuleOp);
          symbolUsers.replaceAllUsesWith(globalOp, disambiguateName);
          SymbolTable::setSymbolName(globalOp, disambiguateName);
        }
        moduleNumber++;
      }
    }

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

    // Specialize the module to the target triple.
    // The executable will have been cloned into other ExecutableTargetOps for
    // other triples so it's fine to mutate in-place.
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

    // Configure the functions in the module. This may override defaults set
    // during the MLIR->LLVM conversion.
    for (auto &func : *llvmModule) {
      // Enable frame pointers to ensure that stack unwinding works, e.g. in
      // Tracy. In principle this could also be achieved by enabling unwind
      // tables, but we tried that and that didn't work in Tracy (which uses
      // libbacktrace), while enabling frame pointers worked.
      // https://github.com/google/iree/issues/3957
      func.addFnAttr("frame-pointer", "all");

      // -ffreestanding-like behavior.
      func.addFnAttr("no-builtins");
    }

    // Build the IREE HAL executable library metadata. The runtime uses this to
    // find the entry point functions and their information.
    LibraryBuilder libraryBuilder(
        llvmModule.get(), LibraryBuilder::Mode::INCLUDE_REFLECTION_ATTRS,
        LibraryBuilder::Version::V_0);
    switch (options_.sanitizerKind) {
      case SanitizerKind::kNone: {
        libraryBuilder.setSanitizerKind(LibraryBuilder::SanitizerKind::NONE);
        break;
      }
      case SanitizerKind::kAddress: {
        libraryBuilder.setSanitizerKind(LibraryBuilder::SanitizerKind::ADDRESS);
        for (auto &function : llvmModule->getFunctionList()) {
          function.addFnAttr(llvm::Attribute::SanitizeAddress);
        }
      } break;
    }
    for (auto entryPointOp :
         targetOp.getBlock().getOps<ExecutableEntryPointOp>()) {
      auto *llvmFunc = llvmModule->getFunction(entryPointOp.getName());
      llvmFunc->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
      llvmFunc->setDSOLocal(true);
      libraryBuilder.addEntryPoint(entryPointOp.getName(), "", llvmFunc);
    }
    auto *queryLibraryFunc =
        libraryBuilder.build("iree_hal_executable_library_query");

    // The query function must be exported for dynamic libraries.
    queryLibraryFunc->setVisibility(
        llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
    queryLibraryFunc->setLinkage(
        llvm::GlobalValue::LinkageTypes::ExternalLinkage);

    // Try to grab a linker tool based on the options (and target environment).
    auto linkerTool = LinkerTool::getForTarget(targetTriple, options_);
    if (!linkerTool) {
      return mlir::emitError(targetOp.getLoc())
             << "failed to find a target linker for the given target triple '"
             << options_.targetTriple << "'";
    }

    // Configure the module with any code generation options required later by
    // linking (such as initializer functions).
    if (failed(linkerTool->configureModule(llvmModule.get(),
                                           {queryLibraryFunc}))) {
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

    iree_DyLibExecutableDef_start_as_root(builder);
    iree_DyLibExecutableDef_library_embedded_add(builder, libraryEmbeddedRef);
    iree_DyLibExecutableDef_debug_database_filename_add(
        builder, debugDatabaseFilenameRef);
    iree_DyLibExecutableDef_debug_database_embedded_add(builder,
                                                        debugDatabaseRef);
    iree_DyLibExecutableDef_end_as_root(builder);

    uint32_t executableFormat =
        targetTriple.isWasm()
            ? static_cast<uint32_t>(IREE::HAL::ExecutableFormat::WASM)
            : static_cast<uint32_t>(IREE::HAL::ExecutableFormat::DyLib);

    // Add the binary data to the target executable.
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        targetOp.getLoc(), targetOp.sym_name(), executableFormat,
        builder.getBufferAttr(executableBuilder.getContext()));
    return success();
  }

 private:
  LLVMTargetOptions options_;
};

void registerLLVMAOTTargetBackends(
    std::function<LLVMTargetOptions()> queryOptions) {
  getLLVMTargetOptionsFromFlags();

#define INIT_LLVM_TARGET(TargetName)        \
  LLVMInitialize##TargetName##Target();     \
  LLVMInitialize##TargetName##TargetMC();   \
  LLVMInitialize##TargetName##TargetInfo(); \
  LLVMInitialize##TargetName##AsmPrinter(); \
  LLVMInitialize##TargetName##AsmParser();

  static TargetBackendRegistration dylibRegistration("dylib-llvm-aot", [=]() {
    INIT_LLVM_TARGET(X86)
    INIT_LLVM_TARGET(ARM)
    INIT_LLVM_TARGET(AArch64)
    INIT_LLVM_TARGET(RISCV)
    return std::make_unique<LLVMAOTTargetBackend>(queryOptions());
  });
  static TargetBackendRegistration wasmRegistration("wasm-llvm-aot", [=]() {
    INIT_LLVM_TARGET(WebAssembly)
    return std::make_unique<LLVMAOTTargetBackend>(queryOptions());
  });

#undef INIT_LLVM_TARGET
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
