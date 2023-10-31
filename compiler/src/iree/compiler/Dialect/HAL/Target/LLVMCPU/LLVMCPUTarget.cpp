// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LLVMCPUTarget.h"

#include <cstdlib>
#include <unordered_set>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/Builtins/Device.h"
#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/Builtins/Musl.h"
#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/Builtins/UKernel.h"
#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LLVMIRPasses.h"
#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LibraryBuilder.h"
#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LinkerTool.h"
#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/StaticLibraryGenerator.h"
#include "iree/compiler/Dialect/HAL/Target/LLVMLinkerUtils.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#define DEBUG_TYPE "iree-llvm-cpu-target"
using llvm::dbgs;

// TODO(ravishankarm): This is redundant w.r.t `iree-vmvx-enable-microkernels`
// flag. Fold these into either a single flag, or not have the flag at all.
static llvm::cl::opt<bool> clEnableCPUMicrokernels(
    "iree-llvmcpu-enable-microkernels",
    llvm::cl::desc(
        "Enables microkernel lowering for llvmcpu backend (experimental)"),
    llvm::cl::init(false));

static llvm::cl::opt<unsigned> clNativeVectorWidthInBytes(
    "iree-llvmcpu-native-vector-width-in-bytes",
    llvm::cl::desc("sets the native vector register width of the hardware. It "
                   "overrides any inferred vector register width"),
    llvm::cl::init(0));

// Default native vector width when target or specific native vector width are
// not provided.
constexpr unsigned defaultNativeVectorWidth = 16;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

static constexpr char kQueryFunctionName[] =
    "iree_hal_executable_library_query";

static void dumpBitcodeToPath(StringRef path, StringRef baseName,
                              StringRef suffix, StringRef extension,
                              llvm::Module &module) {
  llvm::SmallVector<char, 0> data;
  llvm::raw_svector_ostream ostream(data);
  llvm::WriteBitcodeToFile(module, ostream);
  dumpDataToPath(path, baseName, suffix, extension,
                 StringRef(data.data(), data.size()));
}

static void fixupVisibility(llvm::Module &module,
                            const SetVector<llvm::Function *> &preserveFuncs) {
  for (auto &func : module) {
    if (preserveFuncs.contains(&func) || func.getName() == "iree_dll_main") {
      // Leave our library query function as public/external so that it is
      // exported from shared objects and available for linking in static
      // objects.
      continue;
    } else if (func.isDeclaration()) {
      // Declarations must have their original visibility/linkage; they most
      // often come from declared llvm builtin ops (llvm.memcpy/etc).
      continue;
    }
    func.setDSOLocal(true);
    func.setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
  }
  for (auto &global : module.globals()) {
    global.setDSOLocal(true);
    global.setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
  }
}

// Appends the |debugDatabase| to the end of |baseFile| and writes the footer
// so the runtime can find it.
static LogicalResult appendDebugDatabase(std::vector<int8_t> &baseFile,
                                         Artifact &debugFileArtifact) {
  auto debugFileOr = debugFileArtifact.read();
  if (!debugFileOr.has_value()) {
    return failure();
  }
  auto debugFile = std::move(debugFileOr).value();

  // NOTE: we align the sizes so that the files all start at nice offsets.
  auto baseFileSize = IREE::Util::align(baseFile.size(), 16);
  auto debugFileSize = IREE::Util::align(debugFile.size(), 16);

  // Matches iree_hal_system_executable_footer_t.
  struct Footer {
    uint8_t magic[8]; // IREEDBG\0
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

/// Helper method to check if the variant op has a `ukernel` attribute
/// in its `hal.executable.target`. If so, load the ukernel library
/// for that target and link.
// Note: This is duplicate of a similar function in Codegen/. For
// now duplicating this to avoid false linking issues. Eventually
// presence of this attribute in the `hal.executable.target` should
// drive everything.
static bool hasMicrokernel(IREE::HAL::ExecutableVariantOp variantOp) {
  IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.getTarget();
  if (!targetAttr)
    return false;
  auto config = targetAttr.getConfiguration();
  if (!config)
    return false;
  auto attr = config.getAs<BoolAttr>("ukernels");
  return attr && attr.getValue();
}

class LLVMCPUTargetBackend final : public TargetBackend {
public:
  struct AdditionalConfigurationValues {
    std::string dataLayoutStr;
    int64_t vectorSize;
  };

  explicit LLVMCPUTargetBackend(LLVMTargetOptions options)
      : defaultOptions_(std::move(options)) {
    initializeAdditionalConfiguration(defaultOptions_);
  }

  std::string name() const override { return "llvm-cpu"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    // TODO: make inclusion of ArmNeon conditional?
    // clang-format off
    registry.insert<IREE::Codegen::IREECodegenDialect,
                    IREE::LinalgExt::IREELinalgExtDialect,
                    mlir::transform::TransformDialect,
                    pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect,
                    arm_neon::ArmNeonDialect>();
    // clang-format on
  }

  IREE::HAL::DeviceTargetAttr getDeviceTargetFromTarget(
      MLIRContext *context, const LLVMTarget &target,
      const AdditionalConfigurationValues &addlConfig) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(b.getStringAttr("executable_targets"),
                             getExecutableTargets(context, target, addlConfig));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context) const override {
    return getDeviceTargetFromTarget(context, defaultOptions_.target,
                                     defaultAddlConfig_);
  }

  std::optional<IREE::HAL::DeviceTargetAttr>
  getHostDeviceTarget(MLIRContext *context) const override {
    return getDeviceTargetFromTarget(context, LLVMTarget::getForHost(),
                                     defaultAddlConfig_);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpPassManager &passManager) override {
    buildLLVMCPUCodegenPassPipeline(passManager);
  }

  void buildLinkingPassPipeline(OpPassManager &passManager) override {
    buildLLVMCPULinkingPassPipeline(passManager);
  }

  // Gets the LLVM target from |variantOp|.
  // This will differ from the default options specified by command line flags
  // whenever multi-targeting.
  // Returns none and emits on failure.
  std::optional<LLVMTarget>
  getVariantTarget(IREE::HAL::ExecutableVariantOp variantOp) {
    auto configAttr = variantOp.getTarget().getConfiguration();
    return LLVMTarget::loadFromConfigAttr(variantOp.getLoc(), configAttr,
                                          defaultOptions_.target);
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    // Perform the translation in a separate context to avoid any
    // multi-threading issues.
    llvm::LLVMContext context;
    auto maybeTarget = getVariantTarget(variantOp);
    if (!maybeTarget)
      return failure();
    const LLVMTarget &target = *maybeTarget;
    LLVM_DEBUG(dbgs() << "LLVM-CPU SerializeExecutable:\n"
                      << "-----------------------------\n";
               target.print(dbgs()));

    // For debugging effective options in live builds, uncomment the following.
    // dbgs() << "LLVM-CPU ";
    // target.print(dbgs());

    // We name our files after the executable name so that they are easy to
    // track both during compilation (logs/artifacts/etc), as outputs (final
    // intermediate code/binary files), and at runtime (loaded
    // libraries/symbols/etc).
    auto libraryName =
        variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    // Validate flags for output mode.
    if (target.getLinkEmbedded() && target.linkStatic) {
      return variantOp.emitError()
             << "cannot embed ELF and produce static library simultaneously";
    }

    // Try to create the LLVM target machine interface for the variant target.
    auto targetMachine = createTargetMachine(target);
    if (!targetMachine) {
      return mlir::emitError(variantOp.getLoc())
             << "failed to create target machine for target triple '"
             << target.getTriple() << "'";
    }

    // Specialize the module to the target triple.
    // The executable will have been cloned into other ExecutableVariantOps for
    // other triples so it's fine to mutate in-place.
    const llvm::Triple &targetTriple = targetMachine->getTargetTriple();
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
      // https://github.com/openxla/iree/issues/3957
      func.addFnAttr("frame-pointer", "all");

      // -ffreestanding-like behavior.
      func.addFnAttr("no-builtins");

      // Our dispatches are all hot - that's kind of the point.
      // This may favor more aggressive optimizations.
      func.addFnAttr("hot");
    }

    // Build the IREE HAL executable library metadata. The runtime uses this to
    // find the entry point functions and their information.
    LibraryBuilder::Mode libraryBuilderMode =
        target.debugSymbols ? LibraryBuilder::Mode::INCLUDE_REFLECTION_ATTRS
                            : LibraryBuilder::Mode::NONE;
    LibraryBuilder libraryBuilder(llvmModule.get(), libraryBuilderMode,
                                  LibraryBuilder::Version::LATEST);

    switch (target.sanitizerKind) {
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
    case SanitizerKind::kThread: {
      libraryBuilder.setSanitizerKind(LibraryBuilder::SanitizerKind::THREAD);
      for (auto &function : llvmModule->getFunctionList()) {
        function.addFnAttr(llvm::Attribute::SanitizeThread);
      }
    } break;
    }

    // Declare dynamically imported functions.
    auto importsAttrName =
        StringAttr::get(variantOp.getContext(), "hal.executable.imports");
    if (auto importsAttr =
            variantOp->getAttrOfType<ArrayAttr>(importsAttrName)) {
      for (auto importAttr : importsAttr.getAsValueRange<ArrayAttr>()) {
        auto nameAttr = llvm::cast<StringAttr>(importAttr[0]);
        auto weakAttr = llvm::cast<BoolAttr>(importAttr[1]);
        libraryBuilder.addImport(nameAttr.getValue(), weakAttr.getValue());
      }
      variantOp->removeAttr(importsAttrName);
    }

    // Declare exported entry points.
    auto align16 = llvm::Attribute::getWithAlignment(context, llvm::Align(16));
    for (auto exportOp : variantOp.getBlock().getOps<ExecutableExportOp>()) {
      // Find the matching function in the LLVM module.
      auto *llvmFunc = llvmModule->getFunction(exportOp.getName());
      if (!llvmFunc)
        continue;
      llvmFunc->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
      llvmFunc->setDSOLocal(true);

      // Tag the function parameters in case they got removed during conversion.
      // (%arg0: environment, %arg1: dispatch_state, %arg2: workgroup_state)
      for (unsigned i = 0; i <= 2; ++i) {
        llvmFunc->addParamAttr(i, llvm::Attribute::NonNull);
        llvmFunc->addParamAttr(i, llvm::Attribute::NoAlias);
        llvmFunc->addParamAttr(i, align16);
      }

      // Optionally entry points may specify that they require workgroup local
      // memory. We fetch that value here and plumb it through so the runtime
      // knows how much memory to reserve and pass in.
      int64_t localMemorySize = exportOp.getWorkgroupLocalMemory()
                                    .value_or(APInt(64, 0))
                                    .getSExtValue();

      std::string sourceFile = "";
      int sourceLine = 0;
      if (options.debugLevel >= 1) {
        if (auto loc = findFirstFileLoc(exportOp.getLoc())) {
          sourceFile = loc->getFilename().str();
          sourceLine = loc->getLine();
        }
      }
      libraryBuilder.addExport(
          exportOp.getName(), sourceFile, sourceLine, /*tag=*/"",
          LibraryBuilder::DispatchAttrs{localMemorySize}, llvmFunc);
    }

    auto queryFunctionName = std::string(kQueryFunctionName);
    if (target.linkStatic) {
      // Static library query functions must be unique to support multiple
      // libraries in the same namespace.
      queryFunctionName = libraryName + "_library_query";
    }
    auto *queryLibraryFunc = libraryBuilder.build(queryFunctionName);

    // The query function must be exported for dynamic libraries.
    queryLibraryFunc->setDSOLocal(false);
    queryLibraryFunc->setVisibility(
        llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
    queryLibraryFunc->setLinkage(
        llvm::GlobalValue::LinkageTypes::ExternalLinkage);
    queryLibraryFunc->setDLLStorageClass(
        llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);

    // If linking dynamically, find a suitable linker tool and configure the
    // module with any options that tool requires.
    std::unique_ptr<LinkerTool> linkerTool;
    if (!target.linkStatic) {
      // Grab a linker tool based on the options (and target environment).
      // This uses the defaultOptions_ in order to get paths and such, which
      // are environmental, but replace the target with the actual one.
      LLVMTargetOptions options = defaultOptions_;
      options.target = target;
      linkerTool = LinkerTool::getForTarget(targetTriple, options);
      if (!linkerTool) {
        return mlir::emitError(variantOp.getLoc())
               << "failed to find a target linker for the given target triple '"
               << targetTriple.str() << "'";
      }

      // Configure the module with any code generation options required later by
      // linking (such as initializer functions).
      if (failed(linkerTool->configureModule(llvmModule.get(),
                                             {queryLibraryFunc}))) {
        return variantOp.emitError()
               << "failed to configure LLVM module for target linker";
      }
    }

    // Specialize the module to our target machine.
    llvmModule->setDataLayout(targetMachine->createDataLayout());
    llvmModule->setTargetTriple(targetMachine->getTargetTriple().str());

    // Dump just the codegen bitcode before linking and optimization.
    if (!options.dumpIntermediatesPath.empty()) {
      dumpBitcodeToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                        variantOp.getName(), ".codegen.bc", *llvmModule);
    }

    // Statically link libraries into our module prior to LLVM optimizations.
    // This approximates LTO.
    llvm::Linker moduleLinker(*llvmModule);

    // Link any bitcode files specified on the command line.
    if (failed(linkCmdlineBitcodeFiles(variantOp.getLoc(), moduleLinker,
                                       llvm::Linker::OverrideFromSrc,
                                       *targetMachine, context))) {
      return failure();
    }

    // Link any bitcode objects specified in executable.object attributes and
    // specialize them for the current config.
    if (failed(linkBitcodeObjects(variantOp.getLoc(), moduleLinker,
                                  llvm::Linker::LinkOnlyNeeded, *targetMachine,
                                  variantOp.getObjectsAttr(), context))) {
      return failure();
    }

    // Link our libdevice after all codegen and user objects as they may
    // reference it. Some of the functions in here are only known used after
    // we perform LLVM ISel and need to be pulled in whether they are used or
    // not.
    if (failed(linkBitcodeModule(
            variantOp.getLoc(), moduleLinker, llvm::Linker::OverrideFromSrc,
            *targetMachine, "libdevice",
            loadDeviceBitcode(targetMachine.get(), context),
            [&](llvm::Module &module) {
              specializeDeviceModule(variantOp, module, *targetMachine);
            }))) {
      return mlir::emitError(variantOp.getLoc())
             << "failed linking in builtin library for target triple '"
             << targetTriple.str() << "'";
    }

    if (target.getLinkEmbedded()) {
      // Link musl last and pull in all of it - this is sad but LLVM will take
      // IR intrinsics and generate calls out to libc during code generation and
      // we have no control over that - if we don't provide the symbols here
      // then linking with ld will fail.
      if (failed(linkBitcodeModule(
              variantOp.getLoc(), moduleLinker, llvm::Linker::OverrideFromSrc,
              *targetMachine, "libmusl",
              loadMuslBitcode(targetMachine.get(), context)))) {
        return mlir::emitError(variantOp.getLoc())
               << "failed linking in builtin library for target triple '"
               << targetTriple.str() << "'";
      }
    }

    // Tracks ukernel functions, in order to set their linkage to internal
    // after ukernel bitcode modules are linked but before runLLVMIRPasses, so
    // that unused ukernel code paths get DCE'd. Notes:
    // 1. We can't rely on fixupVisibility to do this, because fixupVisibility
    //    is called after runLLVMIRPasses, which is what performs DCE. The
    //    reason why fixupVisibility can't be moved before runLLVMIRPasses is
    //    that causes all math functions to be DCE'd, as references to them get
    //    introduced only later down. The basic difference here between ukernel
    //    functions and math functions is that any references to ukernel
    //    functions already exist at this point.
    // 2. We can't just set internal linkage right away upon loading ukernel
    //    bitcode modules, because some ukernel symbols have to override weak
    //    symbols, and that's disabled when linkage is set to internal.
    std::unordered_set<std::string> ukernelFunctions;

    // Link in ukernel bitcode.
    if (hasMicrokernel(variantOp)) {
      auto setAlwaysInline = [&](llvm::Module &module) {
        for (auto &func : module.getFunctionList()) {
          func.addFnAttr(llvm::Attribute::AlwaysInline);
        }
      };
      auto addUkernelFunctions = [&](const llvm::Module &module) {
        for (auto &func : module.getFunctionList()) {
          if (func.isDeclaration()) {
            continue;
          }
          ukernelFunctions.insert(func.getName().str());
        }
      };

      llvm::Expected<std::unique_ptr<llvm::Module>> archBitcode =
          loadUKernelArchBitcode(targetMachine.get(), context);
      if (!archBitcode) {
        return mlir::emitError(variantOp.getLoc())
               << "failed to load architecture-specific ukernel bitcode: "
               << llvm::toString(archBitcode.takeError());
      }

      llvm::Expected<std::unique_ptr<llvm::Module>> archEntryPointsBitcode =
          loadUKernelArchEntryPointsBitcode(targetMachine.get(), context);
      if (!archEntryPointsBitcode) {
        return mlir::emitError(variantOp.getLoc())
               << "failed to load architecture-specific ukernel entry points "
                  "bitcode: "
               << llvm::toString(archEntryPointsBitcode.takeError());
      }

      // archBitcode and archEntryPointsBitcode are optional, may be null if
      // there is none for the target architecture. However, they should
      // simultaneously be null or non-null.
      if ((archBitcode.get() == nullptr) !=
          (archEntryPointsBitcode.get() == nullptr)) {
        return mlir::emitError(variantOp.getLoc())
               << "there should be architecture-specific ukernel bit code if, "
                  "and only if there is architecture-specific ukernels entry "
                  "points bitcode.";
      }

      if (archBitcode.get()) {
        addUkernelFunctions(*archBitcode.get());
        addUkernelFunctions(*archEntryPointsBitcode.get());

        // archEntryPointsBitcode contains overrides for weak symbols that will
        // come in the baseBitcode below. So we link it before baseBitcode, with
        // OverrideFromSrc.
        StringRef archEntryPointsBitcodeName =
            archEntryPointsBitcode.get()->getName();
        if (failed(linkBitcodeModule(variantOp.getLoc(), moduleLinker, 0,
                                     *targetMachine, archEntryPointsBitcodeName,
                                     std::move(archEntryPointsBitcode),
                                     setAlwaysInline))) {
          return mlir::emitError(variantOp.getLoc())
                 << "failed linking in architecture-specific ukernel entry "
                    "points bitcode "
                    "for target triple '"
                 << targetTriple.str() << "'";
        }

        // archEntryPointsBitcode references symbols defined in archBitcode, so
        // we link that now. We can apply LinkOnlyNeeded, since the only purpose
        // of archBitcode is to satisfy references made in
        // archEntryPointsBitcode.
        StringRef archBitcodeName = archBitcode.get()->getName();
        if (failed(linkBitcodeModule(
                variantOp.getLoc(), moduleLinker, llvm::Linker::LinkOnlyNeeded,
                *targetMachine, archBitcodeName, std::move(archBitcode), {}))) {
          return mlir::emitError(variantOp.getLoc())
                 << "failed linking in architecture-specific ukernel bitcode "
                    "for target triple '"
                 << targetTriple.str() << "'";
        }
      }

      // The baseBitcode module contains weak symbols for fallbacks, potentially
      // overridden by symbols defined in archEntryPointsBitcode above. So this
      // must be linked after archEntryPointsBitcode.
      // The baseBitcode module contains the actual ukernel entry points as seen
      // from the MLIR module, and its purpose is to satisfy these references,
      // so we can apply LinkOnlyNeeded here.
      llvm::Expected<std::unique_ptr<llvm::Module>> baseBitcode =
          loadUKernelBaseBitcode(targetMachine.get(), context);
      if (baseBitcode) {
        addUkernelFunctions(*baseBitcode.get());
      }
      // Sequence that access before we std::move(baseBitcode)!
      StringRef baseBitcodeName =
          baseBitcode ? baseBitcode.get()->getName() : "";
      if (failed(linkBitcodeModule(variantOp.getLoc(), moduleLinker,
                                   llvm::Linker::LinkOnlyNeeded, *targetMachine,
                                   baseBitcodeName, std::move(baseBitcode),
                                   setAlwaysInline))) {
        return mlir::emitError(variantOp.getLoc())
               << "failed linking in base ukernel bitcode";
      }
    }

    // Set internal linkage on all ukernel functions. No new references to
    // ukernels will be created past this point, so any unreferenced ukernel
    // symbol is safe to DCE, which will happen below in runLLVMIRPasses, so we
    // need to set internal linkage before that.
    for (auto &func : llvmModule->getFunctionList()) {
      if (ukernelFunctions.count(func.getName().str())) {
        func.setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
      }
    }

    // Strip any compiler identifiers that may have snuck in. We let the linker
    // tag the module.
    auto *llvmIdent = llvmModule->getNamedMetadata("llvm.ident");
    if (llvmIdent)
      llvmIdent->clearOperands();

    // Dump all linked bitcode prior to optimization.
    if (!options.dumpIntermediatesPath.empty()) {
      dumpBitcodeToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                        variantOp.getName(), ".linked.bc", *llvmModule);
    }

    // LLVM opt passes that perform code generation optimizations/transformation
    // similar to what a frontend would do.
    if (failed(
            runLLVMIRPasses(target, targetMachine.get(), llvmModule.get()))) {
      return variantOp.emitError()
             << "failed to run LLVM-IR opt passes for IREE::HAL::ExecutableOp "
                "targeting '"
             << targetTriple.str() << "'";
    }

    // Fixup visibility from any symbols we may link in - we want to hide all
    // but the query entry point.
    // Note: can't move this before runLLVMIRPasses at the moment, as further
    // symbol references may still be created past this point, namely to math
    // functions, e.g. `llvm.frem` lowering to a call to `fmodf`.
    SetVector<llvm::Function *> preservedFuncs;
    preservedFuncs.insert(queryLibraryFunc);
    fixupVisibility(*llvmModule, preservedFuncs);

    // Dump bitcode post-linking and optimization.
    if (!options.dumpIntermediatesPath.empty()) {
      dumpBitcodeToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                        variantOp.getName(), ".optimized.bc", *llvmModule);
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
                                      llvm::CodeGenFileType::ObjectFile,
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

    // Dump assembly listing after optimization, which is just a textual
    // representation of the object file we generate below.
    if (!options.dumpIntermediatesPath.empty()) {
      std::string asmData;
      if (failed(runEmitObjFilePasses(targetMachine.get(), llvmModule.get(),
                                      llvm::CodeGenFileType::AssemblyFile,
                                      &asmData))) {
        return variantOp.emitError()
               << "failed to compile LLVM-IR module to an assembly file";
      }
      dumpDataToPath(options.dumpIntermediatesPath, options.dumpBaseName,
                     variantOp.getName(), ".s", asmData);
    }

    // If custom object files were specified then add those to our artifact set.
    // These will either be combined into the resulting static library or linked
    // statically into the resulting dynamic library.
    SmallVector<IREE::HAL::ExecutableObjectAttr> linkerObjectAttrs;
    IREE::HAL::ExecutableObjectAttr::filterObjects(variantOp.getObjectsAttr(),
                                                   {".o", ".obj", ".a", ".lib"},
                                                   linkerObjectAttrs);
    for (auto [index, attr] : llvm::enumerate(linkerObjectAttrs)) {
      auto objectAttr = llvm::cast<IREE::HAL::ExecutableObjectAttr>(attr);
      if (auto dataAttr = objectAttr.getData()) {
        objectFiles.push_back(Artifact::createTemporary(
            objectFiles.front().path + "_object_" + std::to_string(index),
            llvm::sys::path::extension(objectAttr.getPath())));
      } else {
        auto absolutePath = objectAttr.getAbsolutePath();
        if (failed(absolutePath)) {
          llvm::errs()
              << "ERROR: referenced object file not found on any path; use "
                 "--iree-hal-executable-object-search-path= to add search "
                 "paths: "
              << objectAttr << "\n";
          return failure();
        }
        objectFiles.push_back(Artifact::fromFile(*absolutePath));
      }
    }

    if (target.linkStatic) {
      return serializeStaticLibraryExecutable(options, target, variantOp,
                                              executableBuilder, libraryName,
                                              queryFunctionName, objectFiles);
    } else {
      return serializeDynamicLibraryExecutable(
          options, target, variantOp, executableBuilder, libraryName,
          targetTriple, objectFiles, linkerTool.get());
    }
  }

  LogicalResult serializeStaticLibraryExecutable(
      const SerializationOptions &options, const LLVMTarget &target,
      IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder,
      const std::string &libraryName, const std::string &queryFunctionName,
      const SmallVector<Artifact> &objectFiles) {
    if (objectFiles.size() != 1) {
      // Static library output only supports single object libraries.
      return variantOp.emitError() << "generating static libraries from "
                                      "multiple object files is not supported";
    }

    // Copy the static object file to the specified output along with
    // generated header file.
    if (!outputStaticLibrary(libraryName, queryFunctionName,
                             target.staticLibraryOutput, objectFiles[0].path)) {
      return variantOp.emitError() << "static library generation failed";
    }

    // Embed the library name in the executable binary op. This informs the
    // loader which static library to load for the target binary.
    std::vector<uint8_t> libraryNameVector(libraryName.begin(),
                                           libraryName.end());
    executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(), "static",
        libraryNameVector);

    return success();
  }

  LogicalResult serializeDynamicLibraryExecutable(
      const SerializationOptions &options, const LLVMTarget &target,
      IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder,
      const std::string &libraryName, const llvm::Triple &targetTriple,
      const SmallVector<Artifact> &objectFiles, LinkerTool *linkerTool) {
    // Link the generated object files into a dylib.
    auto linkArtifactsOr =
        linkerTool->linkDynamicLibrary(libraryName, objectFiles);
    if (!linkArtifactsOr.has_value()) {
      return mlir::emitError(variantOp.getLoc())
             << "failed to link executable and generate target dylib (check "
                "above for more specific error messages)";
    }
    auto &linkArtifacts = linkArtifactsOr.value();
    if (defaultOptions_.keepLinkerArtifacts) {
      mlir::emitRemark(variantOp.getLoc())
          << "linker artifacts for " << variantOp.getName() << " preserved:\n"
          << "    " << linkArtifacts.libraryFile.path;
      linkArtifacts.keepAllFiles();
      for (auto &objectFile : objectFiles) {
        objectFile.keep();
      }
    }

    if (target.getLinkEmbedded()) {
      // Load the linked ELF file and pack into an attr.
      auto elfFile = linkArtifacts.libraryFile.read();
      if (!elfFile.has_value()) {
        return variantOp.emitError()
               << "failed to read back dylib temp file at "
               << linkArtifacts.libraryFile.path;
      }
      if (!options.dumpBinariesPath.empty()) {
        dumpDataToPath<int8_t>(options.dumpBinariesPath, options.dumpBaseName,
                               variantOp.getName(), ".so", *elfFile);
      }
      auto bufferAttr = DenseIntElementsAttr::get(
          VectorType::get({static_cast<int64_t>(elfFile->size())},
                          IntegerType::get(executableBuilder.getContext(), 8)),
          std::move(elfFile.value()));

      // Add the binary to the parent hal.executable.
      auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
          variantOp.getLoc(), variantOp.getSymName(),
          variantOp.getTarget().getFormat(), bufferAttr);
      binaryOp.setMimeTypeAttr(
          executableBuilder.getStringAttr("application/x-elf"));
    } else {
      const char *mimeType = nullptr;
      const char *extension = "";
      switch (targetTriple.getObjectFormat()) {
      case llvm::Triple::ObjectFormatType::COFF:
        mimeType = "application/x-msdownload";
        extension = ".dll";
        break;
      case llvm::Triple::ObjectFormatType::ELF:
        mimeType = "application/x-elf";
        extension = ".so";
        break;
      case llvm::Triple::ObjectFormatType::MachO:
        mimeType = "application/x-dylib";
        extension = ".dylib";
        break;
      case llvm::Triple::ObjectFormatType::Wasm:
        mimeType = "application/wasm";
        extension = ".wasm";
        break;
      default:
        mimeType = "application/octet-stream";
        break;
      }

      // Load the linked system library and optionally tag on the debug
      // database. This debug database sits at the tail of the file and is
      // ignored by system loaders and tools but still accessible to the runtime
      // loader. Not all platforms have separate debug databases and need this.
      auto libraryFileOr = linkArtifacts.libraryFile.read();
      if (!libraryFileOr.has_value()) {
        return variantOp.emitError()
               << "failed to read back dylib temp file at "
               << linkArtifacts.libraryFile.path;
      }
      auto libraryFile = std::move(libraryFileOr).value();
      if (target.debugSymbols && linkArtifacts.debugFile.outputFile) {
        if (failed(appendDebugDatabase(libraryFile, linkArtifacts.debugFile))) {
          return variantOp.emitError()
                 << "failed to append debug database to dylib file";
        }
      }
      if (!options.dumpBinariesPath.empty()) {
        dumpDataToPath<int8_t>(options.dumpBinariesPath, options.dumpBaseName,
                               variantOp.getName(), extension, libraryFile);
      }
      auto bufferAttr = DenseIntElementsAttr::get(
          VectorType::get({static_cast<int64_t>(libraryFile.size())},
                          IntegerType::get(executableBuilder.getContext(), 8)),
          std::move(libraryFile));

      // Add the binary to the parent hal.executable.
      auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
          variantOp.getLoc(), variantOp.getSymName(),
          variantOp.getTarget().getFormat(), bufferAttr);
      binaryOp.setMimeTypeAttr(executableBuilder.getStringAttr(mimeType));
    }

    return success();
  }

private:
  ArrayAttr
  getExecutableTargets(MLIRContext *context, const LLVMTarget &target,
                       const AdditionalConfigurationValues &addlConfig) const {
    SmallVector<Attribute> targetAttrs;
    // This is where we would multiversion.
    targetAttrs.push_back(getExecutableTarget(context, target, addlConfig));
    return ArrayAttr::get(context, targetAttrs);
  }

  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(MLIRContext *context, const LLVMTarget &target,
                      const AdditionalConfigurationValues &addlConfig) const {
    // Add some configurations to the `hal.executable.target` attribute.
    Builder b(context);
    SmallVector<NamedAttribute> configAttrs;
    target.storeToConfigAttrs(context, configAttrs);

    // Compute the format.
    std::string format;
    if (target.linkStatic) {
      // Static libraries are just string references when serialized so we don't
      // need to specify the target architecture.
      format += "static";
    } else {
      // Construct the [loader]-[format]-[arch] triple.
      llvm::Triple targetTriple(target.getTriple());
      if (target.getLinkEmbedded()) {
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
      format += getIreeArchNameForTargetTriple(targetTriple);
    }

    // Additional configuration items that are used in various places.
    // Note: Some of these look to not be isolated properly for
    // cross-compilation, likely requiring a closer look.
    // Set data layout
    configAttrs.emplace_back(b.getStringAttr("data_layout"),
                             b.getStringAttr(addlConfig.dataLayoutStr));
    // Set the native vector size. This creates a dummy llvm module just to
    // build the TTI the right way.
    configAttrs.emplace_back(b.getStringAttr("native_vector_size"),
                             b.getIndexAttr(addlConfig.vectorSize));

    // Check if microkernels are to be enabled.
    configAttrs.emplace_back(b.getStringAttr("ukernels"),
                             b.getBoolAttr(clEnableCPUMicrokernels));

    return IREE::HAL::ExecutableTargetAttr::get(
        context, StringAttr::get(context, "llvm-cpu"),
        StringAttr::get(context, format),
        DictionaryAttr::get(context, configAttrs));
  }

  void initializeAdditionalConfiguration(const LLVMTargetOptions &options) {
    auto targetMachine = createTargetMachine(options.target);
    // TODO(#13988): proper error propagation. This is a common user scenario.
    assert(targetMachine && "createTargetMachine failed");

    // Data layout
    llvm::DataLayout DL = targetMachine->createDataLayout();
    defaultAddlConfig_.dataLayoutStr = DL.getStringRepresentation();

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

    // If target supports AVX-512, enforce 512-bit vector registers.
    llvm::StringRef targetFeatures = targetMachine->getTargetFeatureString();
    if (targetFeatures.contains("avx512")) {
      dummyFunc->addFnAttr("prefer-vector-width", "512");
    }

    llvm::TargetTransformInfo tti =
        targetMachine->getTargetTransformInfo(*dummyFunc);

    // Set the native vector width. We prioritize user-specified widths over
    // widths provided by TTI.
    if (clNativeVectorWidthInBytes) {
      defaultAddlConfig_.vectorSize = clNativeVectorWidthInBytes;
    } else {
      unsigned ttiVectorWidth =
          tti.getRegisterBitWidth(
              llvm::TargetTransformInfo::RGK_FixedWidthVector) /
          8;
      defaultAddlConfig_.vectorSize =
          ttiVectorWidth > 1 ? ttiVectorWidth : defaultNativeVectorWidth;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "CPU : " << targetMachine->getTargetCPU() << "\n";
      llvm::dbgs() << "Target Triple : "
                   << targetMachine->getTargetTriple().normalize() << "\n";
      llvm::dbgs() << "Target Feature string : " << targetFeatures << "\n";
      llvm::dbgs() << "Data Layout : " << defaultAddlConfig_.dataLayoutStr
                   << "\n";
      llvm::dbgs() << "Vector Width : " << defaultAddlConfig_.vectorSize
                   << "\n";
    });
  }

  // Default options as registered from the command line. Should not be
  // relied on outside of getDefaultDeviceTarget() since it represents
  // a static "cross compiling" config and would override more specific
  // settings.
  LLVMTargetOptions defaultOptions_;

  // Additional target information besides that is contained in
  // LLVMTargetOptions defaultOptions_.
  AdditionalConfigurationValues defaultAddlConfig_;
};

void registerLLVMCPUTargetBackends(
    std::function<LLVMTargetOptions()> queryOptions) {
  // Make sure flags are registered.
  LLVMTargetOptions::getFromFlags();

// Dynamically do preprocessor dispatch to initialize only targets that we
// care about if they are enabled. Unfortunately, the way the LLVM macros
// for this are set up and the inability to do a conditional within a macro
// means that we have to syntactically have a macro for every possible
// target we care about. There are more robust ways to do this but they all
// require build support, which is a pain to manage across platforms.
//
// See comments below.
#define LLVM_INITIALIZE_GENERIC(TargetName)                                    \
  LLVMInitialize##TargetName##Target();                                        \
  LLVMInitialize##TargetName##TargetMC();                                      \
  LLVMInitialize##TargetName##TargetInfo();                                    \
  LLVMInitialize##TargetName##AsmPrinter();                                    \
  LLVMInitialize##TargetName##AsmParser();

// CPU targets that we care about and have hard-linked against are here.
// They delegate to the generic initialize above. These must all be added
// to the build file or you will get undefined symbol errors at link time.
#define LLVM_INITIALIZE_TARGET_AArch64() LLVM_INITIALIZE_GENERIC(AArch64)
#define LLVM_INITIALIZE_TARGET_ARM() LLVM_INITIALIZE_GENERIC(ARM)
#define LLVM_INITIALIZE_TARGET_RISCV() LLVM_INITIALIZE_GENERIC(RISCV)
#define LLVM_INITIALIZE_TARGET_X86() LLVM_INITIALIZE_GENERIC(X86)
#define LLVM_INITIALIZE_TARGET_WebAssembly()                                   \
  LLVM_INITIALIZE_GENERIC(WebAssembly)

// We must no-op the name of each target we don't care about. This is annoying,
// but targets aren't created every day and isn't the end of the world. The
// error messages when missing are quite clear and you just add a line here.
#define LLVM_INITIALIZE_TARGET_AMDGPU()
#define LLVM_INITIALIZE_TARGET_AVR()
#define LLVM_INITIALIZE_TARGET_BPF()
#define LLVM_INITIALIZE_TARGET_Hexagon()
#define LLVM_INITIALIZE_TARGET_Lanai()
#define LLVM_INITIALIZE_TARGET_LoongArch()
#define LLVM_INITIALIZE_TARGET_Mips()
#define LLVM_INITIALIZE_TARGET_MSP430()
#define LLVM_INITIALIZE_TARGET_NVPTX()
#define LLVM_INITIALIZE_TARGET_PowerPC()
#define LLVM_INITIALIZE_TARGET_Sparc()
#define LLVM_INITIALIZE_TARGET_SystemZ()
#define LLVM_INITIALIZE_TARGET_VE()
#define LLVM_INITIALIZE_TARGET_XCore()

#define LLVM_TARGET(TargetName) LLVM_INITIALIZE_TARGET_##TargetName()
#include "llvm/Config/Targets.def"

  auto backendFactory = [=]() {
    return std::make_shared<LLVMCPUTargetBackend>(queryOptions());
  };

  // #hal.device.target<"llvm-cpu", ...
  // #hal.executable.target<"llvm-cpu", ...
  static TargetBackendRegistration registration("llvm-cpu", backendFactory);
}

} // namespace HAL
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
