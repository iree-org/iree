// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LinkerTool.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "llvm-linker"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Windows linker (MSVC link.exe-like); for DLL files.
class WindowsLinkerTool : public LinkerTool {
 public:
  using LinkerTool::LinkerTool;

  std::string getSystemToolPath() const override {
    // First check for setting the linker explicitly.
    auto toolPath = LinkerTool::getSystemToolPath();
    if (!toolPath.empty()) return toolPath;

    // No explicit linker specified, search the executable directory (i.e. our
    // own build or install directories) for common tools.
    toolPath = findToolFromExecutableDir({"lld-link"});
    if (!toolPath.empty()) return toolPath;

    llvm::errs() << "No Windows linker tool specified or discovered\n";
    return "";
  }

  LogicalResult configureModule(
      llvm::Module *llvmModule,
      ArrayRef<llvm::Function *> exportedFuncs) override {
    auto &ctx = llvmModule->getContext();

    // Create a _DllMainCRTStartup replacement that does not initialize the CRT.
    // This is required to prevent a bunch of CRT junk (locale, errno, TLS, etc)
    // from getting emitted in such a way that it cannot be stripped by LTCG.
    // Since we don't emit code using the CRT (beyond memset/memcpy) this is
    // fine and can reduce binary sizes by 50-100KB.
    //
    // More info:
    // https://docs.microsoft.com/en-us/cpp/build/run-time-library-behavior?view=vs-2019
    {
      auto dwordType = llvm::IntegerType::get(ctx, 32);
      auto ptrType = llvm::PointerType::getUnqual(dwordType);
      auto entry = cast<llvm::Function>(
          llvmModule
              ->getOrInsertFunction("iree_dll_main", dwordType, ptrType,
                                    dwordType, ptrType)
              .getCallee());
      entry->setCallingConv(llvm::CallingConv::X86_StdCall);
      entry->setDLLStorageClass(
          llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);
      entry->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
      auto *block = llvm::BasicBlock::Create(ctx, "entry", entry);
      llvm::IRBuilder<> builder(block);
      auto one = llvm::ConstantInt::get(dwordType, 1, false);
      builder.CreateRet(one);
    }

    // Since all exports are fetched via the executable library query function
    // we don't need to make them publicly exported on the module. This has
    // the downside of making some tooling (disassemblers/decompilers/binary
    // size analysis tools/etc) a bit harder to work with, though, so when
    // compiling in debug mode we export all the functions.
    if (targetOptions.debugSymbols) {
      for (auto *llvmFunc : exportedFuncs) {
        llvmFunc->setVisibility(
            llvm::GlobalValue::VisibilityTypes::DefaultVisibility);
        llvmFunc->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
        llvmFunc->setDLLStorageClass(
            llvm::GlobalValue::DLLStorageClassTypes::DLLExportStorageClass);

        // In debug modes we need unwind tables in order to get proper stacks on
        // all platforms. We don't support exceptions so unless debugging is
        // requested we omit them to reduce code size.
        llvmFunc->setUWTableKind(llvm::UWTableKind::Async);
      }
    }

    return success();
  }

  Optional<Artifacts> linkDynamicLibrary(
      StringRef libraryName, ArrayRef<Artifact> objectFiles) override {
    Artifacts artifacts;

    // Create the shared object name; if we only have a single input object we
    // can just reuse that.
    if (objectFiles.size() == 1) {
      artifacts.libraryFile =
          Artifact::createVariant(objectFiles.front().path, "dll");
    } else {
      artifacts.libraryFile = Artifact::createTemporary(libraryName, "dll");
    }

    // link.exe doesn't like the files being opened. We don't use them as
    // streams so close them all now before running the linker.
    artifacts.libraryFile.close();

    // We need a full path for the PDB and I hate strings in LLVM grumble.
    SmallString<32> pdbPath(artifacts.libraryFile.path);
    llvm::sys::path::replace_extension(pdbPath, "pdb");

    SmallVector<std::string, 8> flags = {
        getSystemToolPath(),

        // Hide the linker banner message printed each time.
        "/nologo",

        // Useful when debugging linking/loading issues:
        // "/verbose",

        // https://docs.microsoft.com/en-us/cpp/build/reference/dll-build-a-dll?view=vs-2019
        // Builds a DLL and exports functions with the dllexport storage class.
        "/dll",

        // Forces a fixed timestamp to ensure files are reproducible across
        // builds. Undocumented but accepted by both link and lld-link.
        // https://blog.conan.io/2019/09/02/Deterministic-builds-with-C-C++.html
        "/Brepro",

        // https://docs.microsoft.com/en-us/cpp/build/reference/nodefaultlib-ignore-libraries?view=vs-2019
        // Ignore any libraries that are specified by the platform as we
        // directly provide the ones we want.
        "/nodefaultlib",

        // https://docs.microsoft.com/en-us/cpp/build/reference/incremental-link-incrementally?view=vs-2019
        // Disable incremental linking as we are only ever linking in one-shot
        // mode to temp files. This avoids additional file padding and ordering
        // restrictions that enable incremental linking. Our other options will
        // prevent incremental linking in most cases, but it doesn't hurt to be
        // explicit.
        "/incremental:no",

        // https://docs.microsoft.com/en-us/cpp/build/reference/guard-enable-guard-checks?view=vs-2019
        // No control flow guard lookup (indirect branch verification).
        "/guard:no",

        // https://docs.microsoft.com/en-us/cpp/build/reference/safeseh-image-has-safe-exception-handlers?view=vs-2019
        // We don't want exception unwind tables in our output.
        "/safeseh:no",

        // https://docs.microsoft.com/en-us/cpp/build/reference/entry-entry-point-symbol?view=vs-2019
        // Use our entry point instead of the standard CRT one; ensures that we
        // pull in no global state from the CRT.
        "/entry:iree_dll_main",

        // https://docs.microsoft.com/en-us/cpp/build/reference/debug-generate-debug-info?view=vs-2019
        // Copies all PDB information into the final PDB so that we can use the
        // same PDB across multiple machines.
        "/debug:full",

        // https://docs.microsoft.com/en-us/cpp/build/reference/pdb-use-program-database
        // Generates the PDB file containing the debug information.
        ("/pdb:" + pdbPath).str(),

        // https://docs.microsoft.com/en-us/cpp/build/reference/pdbaltpath-use-alternate-pdb-path?view=vs-2019
        // Forces the PDB we generate to be referenced in the DLL as just a
        // relative path to the DLL itself. This allows us to move the PDBs
        // along with the build DLLs across machines.
        "/pdbaltpath:%_PDB%",

        // https://docs.microsoft.com/en-us/cpp/build/reference/out-output-file-name?view=vs-2019
        // Target for linker output. The base name of this path will be used for
        // additional output files (like the map and pdb).
        "/out:" + artifacts.libraryFile.path,
    };

    if (targetOptions.optimizerOptLevel.getSpeedupLevel() >= 2 ||
        targetOptions.optimizerOptLevel.getSizeLevel() >= 2) {
      // https://docs.microsoft.com/en-us/cpp/build/reference/opt-optimizations?view=vs-2019
      // Enable all the fancy optimizations.
      flags.push_back("/opt:ref,icf,lbr");
    }

    // SDK and MSVC paths.
    // These rely on the environment variables provided by the
    // vcvarsall or VsDevCmd ("Developer Command Prompt") scripts. They can also
    // be manually be specified.
    //
    // We could also check to see if vswhere is installed and query that in the
    // event of missing environment variables; that would eliminate the need for
    // specifying things from for example IDEs that may not bring in the vcvars.
    //
    /* Example values:
      UCRTVersion=10.0.18362.0
      UniversalCRTSdkDir=C:\Program Files (x86)\Windows Kits\10\
      VCToolsInstallDir=C:\Program Files (x86)\Microsoft Visual
          Studio\2019\Preview\VC\Tools\MSVC\14.28.29304\
      */
    if (!getenv("VCToolsInstallDir") || !getenv("UniversalCRTSdkDir")) {
      llvm::errs() << "required environment for lld-link/link not specified; "
                      "ensure you are building from a shell where "
                      "vcvarsall/VsDevCmd.bat/etc has been used";
      return std::nullopt;
    }
    const char *arch;
    if (targetTriple.isARM() && targetTriple.isArch32Bit()) {
      arch = "arm";
    } else if (targetTriple.isARM()) {
      arch = "arm64";
    } else if (targetTriple.isX86() && targetTriple.isArch32Bit()) {
      arch = "x86";
    } else if (targetTriple.isX86()) {
      arch = "x64";
    } else {
      llvm::errs() << "unsupported Windows target triple (no arch libs): "
                   << targetTriple.str();
      return std::nullopt;
    }
    flags.push_back(
        llvm::formatv("/libpath:\"{0}\\lib\\{1}\"", "%VCToolsInstallDir%", arch)
            .str());
    flags.push_back(llvm::formatv("/libpath:\"{0}\\Lib\\{1}\\ucrt\\{2}\"",
                                  "%UniversalCRTSdkDir%", "%UCRTVersion%", arch)
                        .str());
    flags.push_back(llvm::formatv("/libpath:\"{0}\\Lib\\{1}\\um\\{2}\"",
                                  "%UniversalCRTSdkDir%", "%UCRTVersion%", arch)
                        .str());

    // We need to link against different libraries based on our configuration
    // matrix (dynamic/static and debug/release).
    int libIndex = 0;
    if (targetOptions.optimizerOptLevel.getSpeedupLevel() == 0) {
      libIndex += 0;  // debug
    } else {
      libIndex += 2;  // release
    }
    libIndex += targetOptions.linkStatic ? 1 : 0;

    // The required libraries for linking DLLs:
    // https://docs.microsoft.com/en-us/cpp/c-runtime-library/crt-library-features?view=msvc-160
    //
    // NOTE: there are only static versions of msvcrt as it's the startup code.
    static const char *kMSVCRTLibs[4] = {
        /*   debug/dynamic */ "msvcrtd.lib",
        /*   debug/static  */ "msvcrtd.lib",
        /* release/dynamic */ "msvcrt.lib",
        /* release/static  */ "msvcrt.lib",
    };
    static const char *kVCRuntimeLibs[4] = {
        /*   debug/dynamic */ "vcruntimed.lib",
        /*   debug/static  */ "libvcruntimed.lib",
        /* release/dynamic */ "vcruntime.lib",
        /* release/static  */ "libvcruntime.lib",
    };
    static const char *kUCRTLibs[4] = {
        /*   debug/dynamic */ "ucrtd.lib",
        /*   debug/static  */ "libucrtd.lib",
        /* release/dynamic */ "ucrt.lib",
        /* release/static  */ "libucrt.lib",
    };
    flags.push_back(kMSVCRTLibs[libIndex]);
    flags.push_back(kVCRuntimeLibs[libIndex]);
    flags.push_back(kUCRTLibs[libIndex]);
    flags.push_back("kernel32.lib");

    // Link all input objects. Note that we are not linking whole-archive as we
    // want to allow dropping of unused codegen outputs.
    for (auto &objectFile : objectFiles) {
      flags.push_back(objectFile.path);
    }

    auto commandLine = llvm::join(flags, " ");
    if (failed(runLinkCommand(commandLine))) return std::nullopt;

    // PDB file gets generated wtih the same path + .pdb.
    artifacts.debugFile =
        Artifact::createVariant(artifacts.libraryFile.path, "pdb");

    // We currently discard some of the other file outputs (like the .exp
    // listing the exported symbols) as we don't need them.
    artifacts.otherFiles.push_back(
        Artifact::createVariant(artifacts.libraryFile.path, "exp"));
    artifacts.otherFiles.push_back(
        Artifact::createVariant(artifacts.libraryFile.path, "lib"));

    return artifacts;
  }
};

std::unique_ptr<LinkerTool> createWindowsLinkerTool(
    const llvm::Triple &targetTriple, LLVMTargetOptions &targetOptions) {
  return std::make_unique<WindowsLinkerTool>(targetTriple, targetOptions);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
