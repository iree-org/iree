// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"

#include <algorithm>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/FileUtilities.h"

namespace mlir::iree_compiler::IREE::HAL {

SmallVector<std::string>
gatherExecutableTargetNames(IREE::HAL::ExecutableOp executableOp) {
  SmallVector<std::string> targetNames;
  llvm::SmallDenseSet<StringRef> targets;
  executableOp.walk([&](IREE::HAL::ExecutableVariantOp variantOp) {
    auto targetName = variantOp.getTarget().getBackend().getValue();
    if (targets.insert(targetName).second) {
      targetNames.push_back(targetName.str());
    }
  });
  llvm::stable_sort(targetNames);
  return targetNames;
}

SmallVector<std::string> gatherExecutableTargetNames(mlir::ModuleOp moduleOp) {
  SmallVector<std::string> targetNames;
  llvm::stable_sort(targetNames);
  llvm::SmallDenseSet<StringRef> targets;
  moduleOp.walk([&](IREE::HAL::ExecutableOp executableOp) {
    executableOp.walk([&](IREE::HAL::ExecutableVariantOp variantOp) {
      auto targetName = variantOp.getTarget().getBackend().getValue();
      if (targets.insert(targetName).second) {
        targetNames.push_back(targetName.str());
      }
    });
  });
  return targetNames;
}

void dumpDataToPath(StringRef path, StringRef baseName, StringRef suffix,
                    StringRef extension, StringRef data) {
  auto fileName = (llvm::join_items("_", baseName, suffix) + extension).str();
  auto fileParts =
      llvm::join_items(llvm::sys::path::get_separator(), path, fileName);
  auto filePath = llvm::sys::path::convert_to_slash(fileParts);
  std::string error;
  auto file = mlir::openOutputFile(filePath, &error);
  if (!file) {
    llvm::errs() << "Unable to dump debug output to " << filePath << "\n";
    return;
  }
  file->os().write(data.data(), data.size());
  file->keep();
}

} // namespace mlir::iree_compiler::IREE::HAL
