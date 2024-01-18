// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LINKERTOOL_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LINKERTOOL_H_

#include <string>

#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LLVMTargetOptions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::IREE::HAL {

struct Artifact {
  // Wraps an existing file on the file system.
  // The file will not be deleted when the artifact is destroyed.
  static Artifact fromFile(StringRef path);

  // Creates an output file path/container pair.
  // By default the file will be deleted when the link completes; callers must
  // use llvm::ToolOutputFile::keep() to prevent deletion upon success (or if
  // leaving artifacts for debugging).
  static Artifact createTemporary(StringRef prefix, StringRef suffix);

  // Creates an output file derived from the given file's path with a new
  // suffix.
  static Artifact createVariant(StringRef basePath, StringRef suffix);

  Artifact() = default;
  Artifact(std::string path, std::unique_ptr<llvm::ToolOutputFile> outputFile)
      : path(std::move(path)), outputFile(std::move(outputFile)) {}

  std::string path;
  std::unique_ptr<llvm::ToolOutputFile> outputFile;

  // Preserves the file contents on disk after the artifact has been destroyed.
  void keep() const;

  // Reads the artifact file contents as bytes.
  std::optional<std::vector<int8_t>> read() const;

  // Reads the artifact file and writes it into the given |stream|.
  bool readInto(raw_ostream &targetStream) const;

  // Closes the ostream of the file while preserving the temporary entry on
  // disk. Use this if files need to be modified by external tools that may
  // require exclusive access.
  void close();
};

struct Artifacts {
  // File containing the linked library (DLL, ELF, etc).
  Artifact libraryFile;

  // Optional file containing associated debug information (if stored
  // separately, such as PDB files).
  Artifact debugFile;

  // Other files associated with linking.
  SmallVector<Artifact> otherFiles;

  // Keeps all of the artifacts around after linking completes. Useful for
  // debugging.
  void keepAllFiles();
};

// Base type for linker tools that can turn object files into shared objects.
class LinkerTool {
public:
  // Gets an instance of a linker tool for the given target options. This may
  // be a completely different toolchain than that of the host.
  static std::unique_ptr<LinkerTool>
  getForTarget(const llvm::Triple &targetTriple,
               LLVMTargetOptions &targetOptions);

  explicit LinkerTool(llvm::Triple targetTriple,
                      LLVMTargetOptions targetOptions)
      : targetTriple(std::move(targetTriple)),
        targetOptions(std::move(targetOptions)) {}

  virtual ~LinkerTool() = default;

  // Returns the path to the system linker tool binary, or empty string if none
  // was discovered.
  virtual std::string getSystemToolPath() const;

  // Configures a module prior to compilation with any additional
  // functions/exports it may need, such as shared object initializer functions.
  virtual LogicalResult
  configureModule(llvm::Module *llvmModule,
                  ArrayRef<llvm::Function *> exportedFuncs) {
    return success();
  }

  // Links the given object files into a dynamically loadable library.
  // The resulting library (and other associated artifacts) will be returned on
  // success.
  virtual std::optional<Artifacts>
  linkDynamicLibrary(StringRef libraryName, ArrayRef<Artifact> objectFiles) = 0;

protected:
  // Runs the given command line on the shell, logging failures.
  LogicalResult runLinkCommand(std::string commandLine, StringRef env = "");

  llvm::Triple targetTriple;
  LLVMTargetOptions targetOptions;
};

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_LINKERTOOL_H_
