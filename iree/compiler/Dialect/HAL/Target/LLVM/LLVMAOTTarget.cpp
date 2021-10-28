// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMAOTTarget.h"

#include <cstdlib>

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMIRPasses.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LibraryBuilder.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LinkerTool.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/StaticLibraryGenerator.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/librt/librt.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#define DEBUG_TYPE "iree-llvmaot-target"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

static constexpr char kQueryFunctionName[] =
    "iree_hal_executable_library_query";

static llvm::Optional<FileLineColLoc> findFirstFileLoc(Location baseLoc) {
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

static std::string guessModuleName(mlir::ModuleOp moduleOp) {
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

// Appends the |debugDatabase| to the end of |baseFile| and writes the footer
// so the runtime can find it.
static LogicalResult appendDebugDatabase(std::vector<int8_t> &baseFile,
                                         Artifact &debugFileArtifact) {
  auto debugFileOr = debugFileArtifact.read();
  if (!debugFileOr.hasValue()) {
    return failure();
  }
  auto debugFile = std::move(debugFileOr).getValue();

  // NOTE: we align the sizes so that the files all start at nice offsets.
  auto baseFileSize = IREE::Util::align(baseFile.size(), 16);
  auto debugFileSize = IREE::Util::align(debugFile.size(), 16);

  // Matches iree_hal_system_executable_footer_t.
  struct Footer {
    uint8_t magic[8];  // IREEDBG\0
    uint32_t version;
    uint32_t flags;
    uint64_t libraryOffset;
    uint64_t librarySize;
    uint64_t debugOffset;
    uint64_t debugSize;
  } footer = {{0}};
  std::memcpy(footer.magic, "IREEDBG\0", sizeof(footer.magic));
  footer.version = 0;
  footer.librarySize = baseFile.size();
  footer.debugOffset = baseFileSize;
  footer.debugSize = debugFile.size();

  baseFile.resize(baseFileSize + debugFileSize + sizeof(footer));
  std::memcpy(baseFile.data() + baseFileSize, debugFile.data(),
              debugFile.size());
  std::memcpy(baseFile.data() + baseFileSize + debugFileSize, &footer,
              sizeof(footer));
  return success();
}

class LLVMAOTTargetBackend final : public TargetBackend {
 public:
  explicit LLVMAOTTargetBackend(LLVMTargetOptions options)
      : options_(std::move(options)) {
    initConfiguration();
  }

  std::string name() const override { return "llvm"; }

  std::string deviceID() const override { return "cpu"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::registerLLVMDialectTranslation(registry);
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(b.getIdentifier("executable_targets"),
                             getExecutableTargets(context));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    buildLLVMCPUCodegenPassPipeline(passManager);
  }

  LogicalResult linkExecutables(mlir::ModuleOp moduleOp) override {
    OpBuilder builder = OpBuilder::atBlockBegin(moduleOp.getBody());

    auto sourceExecutableOps =
        llvm::to_vector<8>(moduleOp.getOps<IREE::HAL::ExecutableOp>());
    if (sourceExecutableOps.size() <= 1) return success();

    // TODO(benvanik): rework linking to support multiple formats.
    auto sharedTargetAttr = getExecutableTarget(builder.getContext());

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
        moduleOp.getLoc(), sharedTargetAttr.getSymbolNameFragment(),
        sharedTargetAttr);
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
    if (options_.linkEmbedded && options_.linkStatic) {
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
    if (options_.linkStatic) {
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

    SmallVector<Artifact> objectFiles;

    // Emit the base object file containing the bulk of our code.
    // This must come first such that we have the proper library linking order.
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

    // Optionally append additional object files that provide functionality that
    // may otherwise have been runtime-dynamic (like libc/libm calls).
    // For now we only do this for embedded uses.
    if (options_.linkEmbedded) {
      if (failed(buildLibraryObjects(variantOp.getLoc(), targetMachine.get(),
                                     objectFiles, context))) {
        return variantOp.emitError() << "failed generating library objects";
      }
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

    if (options_.linkStatic) {
      // Embed the library name in the executable binary op. This informs the
      // loader which static library to load for the target binary.
      std::vector<uint8_t> libraryNameVector(libraryName.begin(),
                                             libraryName.end());
      executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
          variantOp.getLoc(), variantOp.sym_name(), "static",
          libraryNameVector);
    } else if (options_.linkEmbedded) {
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
      auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
          variantOp.getLoc(), variantOp.sym_name(),
          variantOp.target().getFormat(), bufferAttr);
      binaryOp.mime_typeAttr(
          executableBuilder.getStringAttr("application/x-elf"));
    } else {
      // Load the linked system library and optionally tag on the debug
      // database. This debug database sits at the tail of the file and is
      // ignored by system loaders and tools but still accessible to the runtime
      // loader. Not all platforms have separate debug databases and need this.
      auto libraryFileOr = linkArtifacts.libraryFile.read();
      if (!libraryFileOr.hasValue()) {
        return variantOp.emitError()
               << "failed to read back dylib temp file at "
               << linkArtifacts.libraryFile.path;
      }
      auto libraryFile = std::move(libraryFileOr).getValue();
      if (options_.debugSymbols && linkArtifacts.debugFile.outputFile) {
        if (failed(appendDebugDatabase(libraryFile, linkArtifacts.debugFile))) {
          return variantOp.emitError()
                 << "failed to append debug database to dylib file";
        }
      }
      auto bufferAttr = DenseIntElementsAttr::get(
          VectorType::get({static_cast<int64_t>(libraryFile.size())},
                          IntegerType::get(executableBuilder.getContext(), 8)),
          std::move(libraryFile));

      // Add the binary to the parent hal.executable.
      auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
          variantOp.getLoc(), variantOp.sym_name(),
          variantOp.target().getFormat(), bufferAttr);
      const char *mimeType = nullptr;
      switch (targetTriple.getObjectFormat()) {
        case llvm::Triple::ObjectFormatType::COFF:
          mimeType = "application/x-msdownload";
          break;
        case llvm::Triple::ObjectFormatType::ELF:
          mimeType = "application/x-elf";
          break;
        case llvm::Triple::ObjectFormatType::MachO:
          mimeType = "application/x-dylib";
          break;
        case llvm::Triple::ObjectFormatType::Wasm:
          mimeType = "application/wasm";
          break;
        default:
          mimeType = "application/octet-stream";
          break;
      }
      binaryOp.mime_typeAttr(executableBuilder.getStringAttr(mimeType));
    }
    return success();
  }

 private:
  ArrayAttr getExecutableTargets(MLIRContext *context) const {
    SmallVector<Attribute> targetAttrs;
    // This is where we would multiversion.
    targetAttrs.push_back(getExecutableTarget(context));
    return ArrayAttr::get(context, targetAttrs);
  }

  IREE::HAL::ExecutableTargetAttr getExecutableTarget(
      MLIRContext *context) const {
    std::string format;
    if (options_.linkStatic) {
      // Static libraries are just string references when serialized so we don't
      // need to specify the target architecture.
      format += "static";
    } else {
      // Construct the [loader]-[format]-[arch] triple.
      llvm::Triple targetTriple(options_.targetTriple);
      if (options_.linkEmbedded) {
        // Using the IREE embedded ELF format/loader.
        format += "embedded-elf-";
      } else {
        // System-specific shared library format.
        format += "system-";
        switch (targetTriple.getObjectFormat()) {
          case llvm::Triple::ObjectFormatType::COFF:
            format += "dll-";
            break;
          case llvm::Triple::ObjectFormatType::ELF:
            format += "elf-";
            break;
          case llvm::Triple::ObjectFormatType::MachO:
            format += "dylib-";
            break;
          case llvm::Triple::ObjectFormatType::Wasm:
            format += "wasm-";
            break;
          default:
            format += "unknown-";
            break;
        }
      }
      switch (targetTriple.getArch()) {
        case llvm::Triple::ArchType::arm:
          format += "arm_32";
          break;
        case llvm::Triple::ArchType::aarch64:
          format += "arm_64";
          break;
        case llvm::Triple::ArchType::riscv32:
          format += "riscv_32";
          break;
        case llvm::Triple::ArchType::riscv64:
          format += "riscv_64";
          break;
        case llvm::Triple::ArchType::wasm32:
          format += "wasm_32";
          break;
        case llvm::Triple::ArchType::wasm64:
          format += "wasm_64";
          break;
        case llvm::Triple::ArchType::x86:
          format += "x86_32";
          break;
        case llvm::Triple::ArchType::x86_64:
          format += "x86_64";
          break;
        default:
          format += "unknown";
          break;
      }
    }

    // Add some configurations to the `hal.executable.target` attribute.
    SmallVector<NamedAttribute> config;
    auto addConfig = [&](StringRef name, Attribute value) {
      config.emplace_back(
          std::make_pair(Identifier::get(name, context), value));
    };

    // Set target triple.
    addConfig("target_triple", StringAttr::get(context, options_.targetTriple));

    // Set data layout
    addConfig("data_layout", StringAttr::get(context, config_.dataLayoutStr));

    // Set the native vector size. This creates a dummy llvm module just to
    // build the TTI the right way.
    addConfig("native_vector_size",
              IntegerAttr::get(IndexType::get(context), config_.vectorSize));

    return IREE::HAL::ExecutableTargetAttr::get(
        context, StringAttr::get(context, "llvm"),
        StringAttr::get(context, format), DictionaryAttr::get(context, config));
  }

  static void overridePlatformGlobal(llvm::Module &module, StringRef globalName,
                                     uint32_t newValue) {
    // NOTE: the global will not be defined if it is not used in the module.
    auto *globalValue = module.getNamedGlobal(globalName);
    if (!globalValue) return;
    globalValue->setLinkage(llvm::GlobalValue::PrivateLinkage);
    globalValue->setDSOLocal(true);
    globalValue->setConstant(true);
    globalValue->setInitializer(llvm::ConstantInt::get(
        globalValue->getValueType(), APInt(32, newValue)));
  }

  // Builds an object file for the librt embedded runtime library.
  // This is done per link operation so that we can match the precise target
  // configuration. Since we (mostly) link once per user-level compilation
  // this is fine today. If in the future we invoke the compiler for thousands
  // of modules we'd want to (carefully) cache this.
  LogicalResult buildLibraryObjects(Location loc,
                                    llvm::TargetMachine *targetMachine,
                                    SmallVector<Artifact> &objectFiles,
                                    llvm::LLVMContext &context) {
    assert(!objectFiles.empty() && "libraries must come after the base object");

    // Load the generic bitcode file contents.
    llvm::MemoryBufferRef bitcodeBufferRef(
        llvm::StringRef(iree_compiler_librt_create()->data,
                        iree_compiler_librt_create()->size),
        "librt.bc");
    auto bitcodeModuleValue = llvm::parseBitcodeFile(bitcodeBufferRef, context);
    if (!bitcodeModuleValue) {
      return mlir::emitError(loc)
             << "failed to parse librt bitcode: "
             << llvm::toString(bitcodeModuleValue.takeError());
    }
    auto bitcodeModule = std::move(bitcodeModuleValue.get());
    bitcodeModule->setDataLayout(targetMachine->createDataLayout());
    bitcodeModule->setTargetTriple(targetMachine->getTargetTriple().str());

    // Inject target-specific flags.
    // TODO(benvanik): move this entire function to another file that can do
    // more complex logic cleanly. This is just an example.
    overridePlatformGlobal(*bitcodeModule, "librt_platform_example_flag", 0u);

    // Run the LLVM passes to optimize it for the current target.
    if (failed(runLLVMIRPasses(options_, targetMachine, bitcodeModule.get()))) {
      return mlir::emitError(loc)
             << "failed to run librt LLVM-IR opt passes targeting '"
             << options_.targetTriple << "'";
    }

    // Emit an object file we can pass to the linker.
    std::string objectData;
    if (failed(runEmitObjFilePasses(targetMachine, bitcodeModule.get(),
                                    &objectData))) {
      return mlir::emitError(loc)
             << "failed to compile librt LLVM-IR module to an object file";
    }

    // Write the object file to disk with a similar name to the base file.
    auto objectFile =
        Artifact::createVariant(objectFiles.front().path, ".librt.o");
    auto &os = objectFile.outputFile->os();
    os << objectData;
    os.flush();
    os.close();
    objectFiles.push_back(std::move(objectFile));

    return success();
  }

  void initConfiguration() {
    auto targetMachine = createTargetMachine(options_);

    // Data layout
    llvm::DataLayout DL = targetMachine->createDataLayout();
    config_.dataLayoutStr = DL.getStringRepresentation();

    // Set the native vector size. This creates a dummy llvm module just to
    // build the TTI the right way.
    llvm::LLVMContext llvmContext;
    auto llvmModule =
        std::make_unique<llvm::Module>("dummy_module", llvmContext);
    llvm::Type *voidType = llvm::Type::getVoidTy(llvmContext);
    llvmModule->setDataLayout(DL);
    llvm::Function *dummyFunc = llvm::Function::Create(
        llvm::FunctionType::get(voidType, false),
        llvm::GlobalValue::ExternalLinkage, "dummy_func", *llvmModule);
    llvm::TargetTransformInfo tti =
        targetMachine->getTargetTransformInfo(*dummyFunc);
    config_.vectorSize = tti.getRegisterBitWidth(
                             llvm::TargetTransformInfo::RGK_FixedWidthVector) /
                         8;
    LLVM_DEBUG({
      llvm::dbgs() << "CPU : " << targetMachine->getTargetCPU() << "\n";
      llvm::dbgs() << "Target Triple : "
                   << targetMachine->getTargetTriple().normalize() << "\n";
      llvm::dbgs() << "Target Feature string : "
                   << targetMachine->getTargetFeatureString() << "\n";
      llvm::dbgs() << "Data Layout : " << config_.dataLayoutStr << "\n";
      llvm::dbgs() << "Vector Width : " << config_.vectorSize << "\n";
    });
  }

  LLVMTargetOptions options_;

  // Configuration to be set on each `hal.executable.variant` that only depend
  // on the `options_`.
  struct ConfigurationValues {
    std::string dataLayoutStr;
    int64_t vectorSize;
  } config_;
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
    return std::make_shared<LLVMAOTTargetBackend>(queryOptions());
  };

  // #hal.device.target<"cpu", ...
  static TargetBackendRegistration registration0("cpu", backendFactory);
  // #hal.executable.target<"llvm", ...
  static TargetBackendRegistration registration1("llvm", backendFactory);

  // TODO(benvanik): remove legacy dylib name.
  static TargetBackendRegistration registration2("dylib", backendFactory);
  static TargetBackendRegistration registration3("dylib-llvm-aot",
                                                 backendFactory);

#undef INIT_LLVM_TARGET
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
