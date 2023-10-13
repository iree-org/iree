// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Reducer/Framework/Oracle.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Bytecode/BytecodeWriter.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::Reducer;

#define DEBUG_TYPE "iree-reduce-framework"

bool Oracle::isInteresting(WorkItem &workItem) {
  // Check if the module verifies before running the interestingness script.
  // iree-reduce expects a verifiable program at start, so if the program does
  // not verify anymore, it is not interesting.
  if (failed(workItem.verify())) {
    LLVM_DEBUG(llvm::dbgs() << "Module does not verify\n");
    return false;
  }

  // Print module to a temporary file.
  SmallString<128> filepath;
  int fd;
  std::string extension = useBytecode ? "mlirbc" : "mlir";
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile("oracle", extension, fd, filepath);

  if (ec) {
    llvm::report_fatal_error(llvm::Twine("Failed to create temporary file: ") +
                             ec.message());
  }

  llvm::ToolOutputFile output(filepath, fd);

  if (useBytecode) {
    // Write bytecode to file.
    BytecodeWriterConfig config;
    LogicalResult result =
        writeBytecodeToFile(workItem.getModule(), output.os(), config);
    if (failed(result)) {
      llvm::report_fatal_error(
          llvm::Twine("Failed to write bytecode to file: ") + filepath);
    }
  } else {
    // Write MLIR to file.
    workItem.getModule()->print(output.os());
  }

  output.os().close();

  if (output.os().has_error()) {
    llvm::report_fatal_error(
        llvm::Twine("Failed to write to temporary file: ") +
        output.os().error().message());
  }

  // Run the oracle.
  SmallVector<StringRef> testerArgs;
  testerArgs.push_back(testScript);
  testerArgs.push_back(filepath);

  LLVM_DEBUG(llvm::dbgs() << "Running interestingness test: " << testScript
                          << " " << filepath << "\n");

  std::string errMsg;
  int exitCode = llvm::sys::ExecuteAndWait(testScript, testerArgs, std::nullopt,
                                           std::nullopt, 0, 0, &errMsg);

  if (exitCode < 0) {
    llvm::report_fatal_error(llvm::Twine("Failed to run oracle: ") + errMsg);
  }

  if (exitCode == 0) {
    return true;
  } else {
    return false;
  }
}
