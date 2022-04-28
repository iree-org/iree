// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/OptionUtils.h"

#include "llvm/Support/ManagedStatic.h"

namespace mlir {
namespace iree_compiler {

void OptionsBinder::addGlobalOption(std::unique_ptr<llvm::cl::Option> option) {
  static llvm::ManagedStatic<std::vector<std::unique_ptr<llvm::cl::Option>>>
      globalOptions;
  globalOptions->push_back(std::move(option));
}

LogicalResult OptionsBinder::parseArguments(int argc, const char *const *argv,
                                            ErrorCallback onError) {
  assert(scope && "can only parse arguments for local scoped binder");
  for (int i = 0; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    llvm::StringRef nameVal;
    if (arg.startswith("--")) {
      nameVal = arg.drop_front(2);
    } else if (arg.startswith("-")) {
      nameVal = arg.drop_front(1);
    } else {
      // Pure positional options not supported.
      if (onError) {
        onError("pure positional arguments not supported (prefix with '--')");
      }
      return failure();
    }

    // Split name and value.
    llvm::StringRef name;
    llvm::StringRef value;
    size_t eqPos = nameVal.find("=");
    if (eqPos == llvm::StringRef::npos) {
      name = nameVal;
    } else {
      name = nameVal.take_front(eqPos);
      value = nameVal.drop_front(eqPos + 1);
    }

    // Find the option.
    auto foundIt = scope->OptionsMap.find(name);
    if (foundIt == scope->OptionsMap.end()) {
      if (onError) {
        std::string message("option not found: ");
        message.append(name.begin(), name.end());
        onError(message);
      }
      return failure();
    }
    llvm::cl::Option *option = foundIt->second;

    if (llvm::cl::ProvidePositionalOption(option, value, argc)) {
      // Error.
      if (onError) {
        std::string message("option parse error for: ");
        message.append(name.begin(), name.end());
        message.append("=");
        message.append(value.begin(), value.end());
        onError(message);
      }
      return failure();
    }
  }

  return success();
}

llvm::SmallVector<std::string> OptionsBinder::printArguments(
    bool nonDefaultOnly) {
  llvm::SmallVector<std::string> values;
  for (auto &info : localOptions) {
    if (!info.print) continue;
    if (nonDefaultOnly && !info.isChanged()) continue;

    std::string s;
    llvm::raw_string_ostream os(s);
    info.print(os);
    os.flush();
    values.push_back(std::move(s));
  }
  return values;
}

}  // namespace iree_compiler
}  // namespace mlir
