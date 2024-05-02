// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/TransformDialectUtils.h"
#include <string>
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::iree_compiler {

StrategyRunResult
runTransformConfigurationStrategy(Operation *payloadRoot,
                                  StringRef entryPointName,
                                  ModuleOp &transformLibrary) {
  /// If we have a symbol, verify the existence of the symbol within the
  /// transform library.
  Operation *entryPoint = transform::detail::findTransformEntryPoint(
      payloadRoot, transformLibrary, entryPointName);
  if (!entryPoint) {
    return StrategyRunResult::NotFound;
  }

  transform::TransformOptions options;
  if (failed(transform::applyTransformNamedSequence(
          payloadRoot, entryPoint, transformLibrary,
          options.enableExpensiveChecks(true)))) {
    return StrategyRunResult::Failed;
  }
  return StrategyRunResult::Success;
}

FailureOr<std::pair<std::optional<std::string>, std::optional<std::string>>>
parseTransformLibraryFileNameAndEntrySequence(std::string input) {
  SmallVector<StringRef, 2> parts;
  llvm::SplitString(llvm::StringRef(input), parts, "@");
  if (parts.size() > 2) {
    //   funcOp.emitError()
    //       << "Invalid transform library path and sequence name "
    //       << input;
    return failure();
  }
  bool hasTransformLibrary = !parts.empty();
  std::optional<std::string> libraryFileName;
  if (hasTransformLibrary) {
    if (parts[0].empty()) {
      // funcOp.emitError() << "Cannot specify an empty library path";
      return failure();
    }
    libraryFileName = parts[0];
  }
  std::optional<std::string> entrySequenceName;
  // Check if the user specified a custom entry point name.
  if (parts.size() == 2) {
    if (parts[1].empty()) {
      //   funcOp.emitError() << "Cannot specify an empty sequence name";
      return failure();
    }
    entrySequenceName = parts[1];
  }
  return std::make_pair(libraryFileName, entrySequenceName);
}

} // namespace mlir::iree_compiler
