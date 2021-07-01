// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMAOTTarget.h"

#include <cstdlib>

#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMIRPasses.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LibraryBuilder.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LinkerTool.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/StaticLibraryGenerator.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/dylib_executable_def_builder.h"
#include "llvm/Bitcode/BitcodeWriter.h"
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

constexpr char kQueryFunctionName[] = "iree_hal_executable_library_query";

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
  std::string name() const override { return "llvm"; }

  std::string deviceID() const override { return "dylib"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::registerLLVMDialectTranslation(registry);
  }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    auto targetMachine = createTargetMachine(options_);
    if (!targetMachine) {
      llvm::errs() << "failed to create target machine for target triple '"
                   << options_.targetTriple << "'";
      return;
    }
    passManager.addPass(createLLVMCPULowerExecutableTargetPass());
    // Set target specific options.
    LLVMCPUCodegenPassPipelineOptions codeGenOptions;
    codeGenOptions.targetTriple = options_.targetTriple;
    codeGenOptions.targetDataLayout =
        targetMachine->createDataLayout().getStringRepresentation();

    // TODO(ataei): This is temporary here, should move when target specific
    // overrides options grows.
    if (targetMachine->getTargetTriple().isWasm()) {
      codeGenOptions.unfuseFMAOps = true;
    }

    buildLLVMCPUCodegenPassPipeline(passManager, codeGenOptions);
  }

  LogicalResult linkExecutables(mlir::ModuleOp moduleOp) override {
    OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());

    auto sourceExecutableOps =
        llvm::to_vector<8>(moduleOp.getOps<IREE::HAL::ExecutableOp>());
    if (sourceExecutableOps.size() <= 1) return success();

    // Ensure any LLVM symbol names we define are unique prior to linking.
    //
    // The link executables pass requires that there be no name conflicts
    // between symbols with public MLIR Symbol visibility. LLVM dialect symbols
    // use a different visibility mechanism, defaulting to public for MLIR
    // Symbol visibility.
    unsigned moduleNumber = 0;
    for (auto sourceExecutableOp : enumerate(sourceExecutableOps)) {
      auto variantOps = llvm::to_vector<4>(
          sourceExecutableOp.value().getOps<IREE::HAL::ExecutableVariantOp>());
      for (auto variantOp : variantOps) {
        if (variantOp.target() != name()) continue;

        auto sourceModuleOp = variantOp.getInnerModule();
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

    // Add our hal.executable.variant with an empty module.
    builder.setInsertionPointToStart(linkedExecutableOp.getBody());
    auto linkedTargetOp = builder.create<IREE::HAL::ExecutableVariantOp>(
        moduleOp.getLoc(), name(), name());
    builder.setInsertionPoint(&linkedTargetOp.getBlock().back());
    builder.create<ModuleOp>(moduleOp.getLoc());

    // Try linking together all executables in moduleOp.
    return linkExecutablesInto(
        moduleOp, sourceExecutableOps, linkedExecutableOp, linkedTargetOp,
        [](mlir::ModuleOp moduleOp) { return moduleOp; }, builder);
  }

  LogicalResult serializeExecutable(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    // Perform the translation in a separate context to avoid any
    // multi-threading issues.
    llvm::LLVMContext context;

    // We name our files after the executable name so that they are easy to
    // track both during compilation (logs/artifacts/etc), as outputs (final
    // intermediate code/binary files), and at runtime (loaded
    // libraries/symbols/etc).
    auto libraryName =
        variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    // Validate flags for output mode.
    if (options_.linkEmbedded && !options_.staticLibraryOutput.empty()) {
      return variantOp.emitError()
             << "cannot embed ELF and produce static library simultaneously";
    }

    // Specialize the module to the target triple.
    // The executable will have been cloned into other ExecutableVariantOps for
    // other triples so it's fine to mutate in-place.
    llvm::Triple targetTriple(options_.targetTriple);
    variantOp.getInnerModule()->setAttr(
        LLVM::LLVMDialect::getTargetTripleAttrName(),
        executableBuilder.getStringAttr(targetTriple.str()));

    // At this moment we are leaving MLIR LLVM dialect land translating module
    // into target independent LLVMIR.
    auto llvmModule = mlir::translateModuleToLLVMIR(variantOp.getInnerModule(),
                                                    context, libraryName);
    if (!llvmModule) {
      return variantOp.emitError() << "failed to translate the MLIR LLVM "
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

      // Our dispatches are all hot - that's kind of the point.
      // This may favor more aggressive optimizations.
      func.addFnAttr("hot");
    }

    // Build the IREE HAL executable library metadata. The runtime uses this to
    // find the entry point functions and their information.
    // TODO(benvanik): add a flag for this (adds a few KB/binary).
    LibraryBuilder::Mode libraryBuilderMode =
        LibraryBuilder::Mode::INCLUDE_REFLECTION_ATTRS;
    LibraryBuilder libraryBuilder(llvmModule.get(), libraryBuilderMode,
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
         variantOp.getBlock().getOps<ExecutableEntryPointOp>()) {
      // Find the matching function in the LLVM module.
      auto *llvmFunc = llvmModule->getFunction(entryPointOp.getName());
      llvmFunc->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
      llvmFunc->setDSOLocal(true);

      // Optionally entry points may specify that they require workgroup local
      // memory. We fetch that value here and plumb it through so the runtime
      // knows how much memory to reserve and pass in.
      int64_t localMemorySize = entryPointOp.workgroup_local_memory()
                                    .getValueOr(APInt(64, 0))
                                    .getSExtValue();

      libraryBuilder.addExport(entryPointOp.getName(), "",
                               LibraryBuilder::DispatchAttrs{localMemorySize},
                               llvmFunc);
    }

    auto queryFunctionName = std::string(kQueryFunctionName);
    if (!options_.staticLibraryOutput.empty()) {
      // Static library query functions must be unique to support multiple
      // libraries in the same namespace.
      queryFunctionName = libraryName + "_library_query";
    }
    auto *queryLibraryFunc = libraryBuilder.build(queryFunctionName);

    // The query function must be exported for dynamic libraries.
    queryLibraryFunc->setVisibility(
        llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
    queryLibraryFunc->setLinkage(
        llvm::GlobalValue::LinkageTypes::ExternalLinkage);

    // Try to grab a linker tool based on the options (and target environment).
    auto linkerTool = LinkerTool::getForTarget(targetTriple, options_);
    if (!linkerTool) {
      return mlir::emitError(variantOp.getLoc())
             << "failed to find a target linker for the given target triple '"
             << options_.targetTriple << "'";
    }

    // Configure the module with any code generation options required later by
    // linking (such as initializer functions).
    if (failed(linkerTool->configureModule(llvmModule.get(),
                                           {queryLibraryFunc}))) {
      return variantOp.emitError()
             << "failed to configure LLVM module for target linker";
    }

    // LLVM opt passes that perform code generation optimizations/transformation
    // similar to what a frontend would do before passing to linking.
    auto targetMachine = createTargetMachine(options_);
    if (!targetMachine) {
      return mlir::emitError(variantOp.getLoc())
             << "failed to create target machine for target triple '"
             << options_.targetTriple << "'";
    }
    llvmModule->setDataLayout(targetMachine->createDataLayout());
    llvmModule->setTargetTriple(targetMachine->getTargetTriple().str());
    if (failed(
            runLLVMIRPasses(options_, targetMachine.get(), llvmModule.get()))) {
      return variantOp.emitError()
             << "failed to run LLVM-IR opt passes for IREE::HAL::ExecutableOp "
                "targeting '"
             << options_.targetTriple << "'";
    }

    // Emit object files.
    SmallVector<Artifact, 4> objectFiles;
    {
      // NOTE: today we just use a single object file, however if we wanted to
      // scale code generation and linking we'd want to generate one per
      // function (or something like that). A single object file is also
      // instrumental to static library generation (which only supports one
      // object file per library).
      std::string objectData;
      if (failed(runEmitObjFilePasses(targetMachine.get(), llvmModule.get(),
                                      &objectData))) {
        return variantOp.emitError()
               << "failed to compile LLVM-IR module to an object file";
      }
      auto objectFile = Artifact::createTemporary(libraryName, "o");
      auto &os = objectFile.outputFile->os();
      os << objectData;
      os.flush();
      os.close();
      objectFiles.push_back(std::move(objectFile));
    }

    // If we are keeping artifacts then let's also add the bitcode for easier
    // debugging (vs just the binary object file).
    if (options_.keepLinkerArtifacts) {
      auto bitcodeFile =
          Artifact::createVariant(objectFiles.front().path, "bc");
      auto &os = bitcodeFile.outputFile->os();
      llvm::WriteBitcodeToFile(*llvmModule, os);
      os.flush();
      os.close();
      bitcodeFile.outputFile->keep();
    }

    if (!options_.staticLibraryOutput.empty()) {
      if (objectFiles.size() != 1) {
        // Static library output only supports single object libraries.
        return variantOp.emitError()
               << "generating static libraries from "
                  "multiple object files is not supported";
      }

      // Copy the static object file to the specified output along with
      // generated header file.
      const std::string &libraryPath = options_.staticLibraryOutput;
      const auto library_name = objectFiles[0].path;
      if (!outputStaticLibrary(libraryName, queryFunctionName, libraryPath,
                               objectFiles[0].path)) {
        return variantOp.emitError() << "static library generation failed";
      }
    }

    // Link the generated object files into a dylib.
    auto linkArtifactsOr =
        linkerTool->linkDynamicLibrary(libraryName, objectFiles);
    if (!linkArtifactsOr.hasValue()) {
      return mlir::emitError(variantOp.getLoc())
             << "failed to link executable and generate target dylib using "
                "linker toolchain "
             << linkerTool->getToolPath();
    }
    auto &linkArtifacts = linkArtifactsOr.getValue();
    if (options_.keepLinkerArtifacts) {
      mlir::emitRemark(variantOp.getLoc())
          << "linker artifacts for " << variantOp.getName() << " preserved:\n"
          << "    " << linkArtifacts.libraryFile.path;
      linkArtifacts.keepAllFiles();
      for (auto &objectFile : objectFiles) {
        objectFile.outputFile->keep();
      }
    }

    if (options_.linkEmbedded) {
      // Load the linked ELF file and pack into an attr.
      auto elfFile = linkArtifacts.libraryFile.read();
      if (!elfFile.hasValue()) {
        return variantOp.emitError()
               << "failed to read back dylib temp file at "
               << linkArtifacts.libraryFile.path;
      }
      auto bufferAttr = DenseIntElementsAttr::get(
          VectorType::get({static_cast<int64_t>(elfFile->size())},
                          IntegerType::get(executableBuilder.getContext(), 8)),
          std::move(elfFile.getValue()));

      // Add the binary to the parent hal.executable.
      auto executableFormatAttr = executableBuilder.getStringAttr("EX_ELF");
      auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
          variantOp.getLoc(), variantOp.sym_name(), executableFormatAttr,
          bufferAttr);
      binaryOp.mime_typeAttr(
          executableBuilder.getStringAttr("application/x-elf"));
    } else if (!options_.staticLibraryOutput.empty()) {
      // Embed the library name in the executable binary op. This informs the
      // loader which static library to load for the target binary.
      std::vector<uint8_t> libraryNameVector(libraryName.begin(),
                                             libraryName.end());
      auto executableFormatAttr = std::string("static");

      auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
          variantOp.getLoc(), variantOp.sym_name(), executableFormatAttr,
          libraryNameVector);
    } else {
      FlatbufferBuilder builder;
      iree_DyLibExecutableDef_start_as_root(builder);

      // Embed debug symbols at the end of the flatbuffer by adding first in the
      // bottoms-up builder.
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
        return variantOp.emitError()
               << "failed to read back dylib temp file at "
               << linkArtifacts.libraryFile.path;
      }

      iree_DyLibExecutableDef_library_embedded_add(builder, libraryEmbeddedRef);
      iree_DyLibExecutableDef_debug_database_filename_add(
          builder, debugDatabaseFilenameRef);
      iree_DyLibExecutableDef_debug_database_embedded_add(builder,
                                                          debugDatabaseRef);
      iree_DyLibExecutableDef_end_as_root(builder);

      auto executableFormatAttr = targetTriple.isWasm()
                                      ? executableBuilder.getStringAttr("WASM")
                                      : executableBuilder.getStringAttr("DLIB");

      // Add the binary data to the target executable.
      auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
          variantOp.getLoc(), variantOp.sym_name(), executableFormatAttr,
          builder.getBufferAttr(executableBuilder.getContext()));
      binaryOp.mime_typeAttr(
          executableBuilder.getStringAttr("application/x-flatbuffers"));
    }
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

  auto backendFactory = [=]() {
    INIT_LLVM_TARGET(X86)
    INIT_LLVM_TARGET(ARM)
    INIT_LLVM_TARGET(AArch64)
    INIT_LLVM_TARGET(RISCV)
    INIT_LLVM_TARGET(WebAssembly)
    return std::make_unique<LLVMAOTTargetBackend>(queryOptions());
  };
  static TargetBackendRegistration registration0("llvm", backendFactory);
  static TargetBackendRegistration registration1("dylib-llvm-aot",
                                                 backendFactory);

#undef INIT_LLVM_TARGET
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
